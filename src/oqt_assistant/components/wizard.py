# File: src/oqt_assistant/components/wizard.py
from __future__ import annotations
import asyncio
import json
import os
import textwrap
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st

from oqt_assistant.utils.wizard_state import WizardState, READ_ACROSS_DEFINITION
from oqt_assistant.utils.filters import filter_experimental_records
from oqt_assistant.utils.data_formatter import clean_response_data
from oqt_assistant.utils import data_formatter

# Your API + Agents
from oqt_assistant.utils.qsar_api import QSARToolboxAPI
from oqt_assistant.utils.llm_utils import (
    analyze_chemical_context, analyze_physical_properties,
    analyze_environmental_fate, analyze_profiling_reactivity,
    analyze_experimental_data, analyze_read_across, synthesize_report
)

# Optional: if you have a dedicated PDF builder already, we use it; else we fallback
try:
    from oqt_assistant.components.report_pdf import export_pdf_report  # your existing module, if any
except Exception:
    export_pdf_report = None

WIZ_KEY = "wiz_state"

DEFAULT_ENDPOINTS = [
    "Ecotoxicity (Aquatic/Terrestrial)", "Persistence/Bioaccumulation",
    "Genotoxicity/Mutagenicity", "Carcinogenicity", "Reproductive/Developmental Toxicity",
    "Neurotoxicity", "Endocrine Disruption", "Skin/Eye Irritation/Corrosion", "Sensitization"
]

def _init_state() -> WizardState:
    if WIZ_KEY not in st.session_state:
        st.session_state[WIZ_KEY] = WizardState()
    return st.session_state[WIZ_KEY]

def _header(step_idx: int, total_steps: int):
    st.markdown(
        f"""
        <div style="padding:8px 0;">
          <span style="font-weight:600;">Guided Wizard</span>
          <span style="opacity:0.75;"> — Step {step_idx+1} of {total_steps}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress((step_idx+1)/total_steps)

def _nav(prev_enabled=True, next_enabled=True, next_label="Next", prev_label="Back"):
    cols = st.columns([1,1,6,1,1])
    moved = False
    with cols[1]:
        if st.button(prev_label, use_container_width=True, disabled=not prev_enabled):
            st.session_state[WIZ_KEY].current_step = max(0, st.session_state[WIZ_KEY].current_step - 1)
            moved = True
    with cols[3]:
        if st.button(next_label, use_container_width=True, disabled=not next_enabled):
            st.session_state[WIZ_KEY].current_step = st.session_state[WIZ_KEY].current_step + 1
            moved = True
    return moved

def _cost_estimate_ui(wiz: WizardState):
    st.subheader("Estimated LLM Cost")
    with st.expander("Show estimate details", expanded=False):
        in_tokens = wiz.last_token_budget_in or 8000   # conservative default
        out_tokens = wiz.last_token_budget_out or 3500
        cost_in = (in_tokens/1_000_000) * wiz.llm.cost_per_million_in
        cost_out = (out_tokens/1_000_000) * wiz.llm.cost_per_million_out
        estimate = cost_in + cost_out
        st.info(
            f"Input ~ {in_tokens:,} tok; Output ~ {out_tokens:,} tok.\n\n"
            f"Unit prices (editable): ${wiz.llm.cost_per_million_in:.2f}/M in, "
            f"${wiz.llm.cost_per_million_out:.2f}/M out.\n\n"
            f"**Estimated total: ${estimate:.2f}.**"
        )
        wiz.last_cost_estimate_usd = estimate
        wiz.llm.cost_per_million_in = st.number_input("$/M input tokens", value=wiz.llm.cost_per_million_in, step=0.5)
        wiz.llm.cost_per_million_out = st.number_input("$/M output tokens", value=wiz.llm.cost_per_million_out, step=0.5)

def _ping_api(api: QSARToolboxAPI) -> Tuple[bool, str]:
    try:
        # Light touch “ping” via a tiny search
        api.search_by_name("water")
        return True, "Connected"
    except Exception as e:
        return False, f"Not connected: {e}"

def _run_pipeline(results: Dict[str, Any], identifier: str, context: str, wiz: WizardState) -> Tuple[str, Dict[str, str]]:
    """
    Calls your existing agents in the same order you documented.
    Uses wizard overrides for LLM provider only if they are set.
    """
    # Agent orchestration (mirrors Listing 1; runs sequentially+gathered)
    async def _run():
        # optional: seed is handled by provider backends when supported
        analysis_context = f"Target: {identifier}\n\nUser Goal: {context}\n\nWizard Scope: {json.dumps(wiz.to_recipe(), indent=2)}"
        identity_txt = await analyze_chemical_context(results.get("chemical_data", {}), analysis_context)
        analysis_context = f"{identity_txt}\n\nUser Goal: {context}\n\nWizard Scope: {json.dumps(wiz.to_recipe(), indent=2)}"

        # Parallel cores
        task_phys = asyncio.create_task(analyze_physical_properties(results.get("properties", {}), analysis_context))
        task_env  = asyncio.create_task(analyze_environmental_fate(results.get("properties", {}), analysis_context))
        task_prof = asyncio.create_task(analyze_profiling_reactivity(results.get("profiling", {}), analysis_context))
        task_exp  = asyncio.create_task(analyze_experimental_data({"experimental_results": results.get("experimental_data", [])}, analysis_context))
        phys, env, prof, exp = await asyncio.gather(task_phys, task_env, task_prof, task_exp)

        # Read‑across
        read_across = await analyze_read_across(
            {"raw": results, "wizard_recipe": wiz.to_recipe()},
            [phys, env, prof, exp],
            analysis_context,
        )
        # Synthesizer
        final_report = await synthesize_report(identifier, [phys, env, prof, exp], read_across, context)
        return final_report, {"Physical": phys, "Env Fate": env, "Profiling": prof, "Experimental": exp, "Read-Across": read_across}

    # If your agents internally call get_llm() we already patched to accept an override.
    # Nothing else needed here—just run.
    report_txt, agents_out = asyncio.run(_run())
    return report_txt, agents_out

def _apply_scope_filters(cleaned: Dict[str, Any], wiz: WizardState) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Filter experimental data according to wizard choices; leave other slices untouched."""
    filtered = dict(cleaned)
    dropped_counts = {"dropped_mammalian": 0, "dropped_adme_tk": 0}

    if wiz.scope.include_experimental and cleaned.get("experimental_data"):
        kept, counts = filter_experimental_records(
            cleaned["experimental_data"],
            exclude_adme_tk=wiz.scope.exclude_adme_tk,
            exclude_mammalian=wiz.scope.exclude_mammalian_endpoints,
        )
        filtered["experimental_data"] = kept
        dropped_counts.update(counts)
    return filtered, dropped_counts

def render():
    wiz = _init_state()
    steps = ["Setup", "Chemical", "Scope", "Read‑Across", "Review", "Run"]
    total = len(steps)

    st.markdown("<style>.stRadio > label {font-weight:600}</style>", unsafe_allow_html=True)
    st.sidebar.caption("Wizard • non-destructive, additive to current app")
    st.sidebar.json(wiz.to_recipe(), expanded=False)

    # STEP 0: Setup
    if wiz.current_step == 0:
        _header(0, total)
        st.subheader("1) Setup & Connectivity")

        # API URL (auto-populated from existing sidebar if you expose it; else env var)
        api_url = st.text_input("QSAR Toolbox API URL", value=(wiz.setup.api_url or os.environ.get("QSAR_TOOLBOX_API_URL", "")))
        wiz.setup.api_url = api_url

        # LLM Provider
        st.markdown("**LLM Provider**")
        wiz.llm.provider = st.radio("Provider", options=["openai", "openai-compatible", "none"], index=["openai","openai-compatible","none"].index(wiz.llm.provider), horizontal=True)
        colp = st.columns(2)
        with colp[0]:
            wiz.llm.model = st.text_input("Model ID", value=wiz.llm.model, help="e.g., gpt-4.1-mini or your local model name")
        with colp[1]:
            wiz.llm.temperature = st.slider("Temperature", 0.0, 1.0, wiz.llm.temperature, 0.05)
        wiz.llm.max_output_tokens = st.number_input("Max output tokens", min_value=256, max_value=8192, value=wiz.llm.max_output_tokens, step=128)

        _cost_estimate_ui(wiz)

        # Context & label
        st.subheader("2) Analysis Context")
        wiz.setup.analysis_label = st.text_input("Analysis label / Case ID", value=wiz.setup.analysis_label)
        wiz.setup.context = st.text_area("What should the analysis focus on?", value=wiz.setup.context, height=120,
                                         placeholder="e.g., Regulatory endpoints X/Y, data gap analysis, Parkinson’s-relevant neurotoxicity, etc.")
        wiz.setup.export_pdf = st.checkbox("Export a PDF report after run", value=wiz.setup.export_pdf)

        # Connectivity check
        ok = False
        if wiz.setup.api_url:
            api = QSARToolboxAPI(base_url=wiz.setup.api_url)
            ok, msg = _ping_api(api)
            st.success(f"API: {msg}") if ok else st.error(f"API: {msg}")

        _nav(prev_enabled=False, next_enabled=ok)

    # STEP 1: Chemical
    elif wiz.current_step == 1:
        _header(1, total)
        st.subheader("Chemical Identification")
        wiz.chemical.search_type = st.radio("Search by", ["name", "smiles", "cas"], index=["name","smiles","cas"].index(wiz.chemical.search_type), horizontal=True)
        wiz.chemical.identifier = st.text_input("Identifier", value=wiz.chemical.identifier, placeholder="e.g., Chlorpyrifos or a SMILES string")

        chosen = None
        if st.button("Search and Resolve"):
            api = QSARToolboxAPI(base_url=wiz.setup.api_url)
            try:
                if wiz.chemical.search_type == "name":
                    hits = api.search_by_name(wiz.chemical.identifier)
                elif wiz.chemical.search_type == "smiles":
                    hits = api.search_by_smiles(wiz.chemical.identifier)
                else:  # cas
                    # some installs accept name search for CAS too
                    hits = api.search_by_name(wiz.chemical.identifier)

                if not hits:
                    st.warning("No matches found.")
                else:
                    options = [(h.get("id") or h.get("ID") or str(i), f"{h.get('name') or h.get('Name') or 'Unknown'}  •  {h.get('cas') or h.get('CAS') or ''}") for i,h in enumerate(hits)]
                    labels = [o[1] for o in options]
                    idx = st.selectbox("Pick the intended chemical", range(len(labels)), format_func=lambda i: labels[i])
                    chosen = options[idx][0]
            except Exception as e:
                st.error(f"Search failed: {e}")

        if chosen:
            wiz.chemical.selected_chemical_id = str(chosen)
            wiz.chemical.selected_display_name = wiz.chemical.identifier
            wiz.chemical.resolved = True
            st.success("Chemical resolved.")

        _nav(prev_enabled=True, next_enabled=wiz.chemical.resolved)

    # STEP 2: Scope
    elif wiz.current_step == 2:
        _header(2, total)
        st.subheader("Data Scope & Constraints")

        c1, c2, c3 = st.columns(3)
        with c1:
            wiz.scope.include_properties = st.checkbox("Properties", value=wiz.scope.include_properties)
        with c2:
            wiz.scope.include_experimental = st.checkbox("Experimental data", value=wiz.scope.include_experimental)
        with c3:
            wiz.scope.include_profilers = st.checkbox("Profiling/Alerts", value=wiz.scope.include_profilers)

        wiz.scope.include_metabolism = st.checkbox("Include metabolism simulators (optional)", value=wiz.scope.include_metabolism)

        st.markdown("**Exclusions (reviewer concerns):**")
        ec1, ec2 = st.columns(2)
        with ec1:
            wiz.scope.exclude_adme_tk = st.checkbox("Exclude ADME/TK records", value=wiz.scope.exclude_adme_tk,
                                                    help="Avoids human TK/ADME predictions in the narrative.")
        with ec2:
            wiz.scope.exclude_mammalian_endpoints = st.checkbox("Exclude mammalian toxicity endpoints", value=wiz.scope.exclude_mammalian_endpoints,
                                                                 help="Skips LD50/LC50 etc. to keep focus on non‑mammalian endpoints.")

        st.caption("You can wire calculator/profiler GUIDs later; these lists are passthrough hints to agents.")
        wiz.scope.calculators_selected = st.text_area("Calculators to run (names or GUIDs, comma‑sep)", value=",".join(wiz.scope.calculators_selected)).split(",") if st.session_state.get(WIZ_KEY) else []
        wiz.scope.profilers_selected = st.text_area("Profilers to run (names or GUIDs, comma‑sep)", value=",".join(wiz.scope.profilers_selected)).split(",") if st.session_state.get(WIZ_KEY) else []
        if wiz.scope.include_metabolism:
            wiz.scope.metabolism_sims_selected = st.text_area("Metabolism simulators (GUIDs, comma‑sep)", value=",".join(wiz.scope.metabolism_sims_selected)).split(",")

        _nav(prev_enabled=True, next_enabled=True)

    # STEP 3: Read‑Across
    elif wiz.current_step == 3:
        _header(3, total)
        st.subheader("Read‑Across Configuration")
        wiz.readacross.strategy = st.radio("Strategy", ["analogue", "category", "both"], index=["analogue","category","both"].index(wiz.readacross.strategy), horizontal=True)
        wiz.readacross.include_metabolites = st.checkbox("Include metabolites as potential sources", value=wiz.readacross.include_metabolites)

        st.markdown("**Endpoints of interest**")
        selected = st.multiselect("Select endpoints", DEFAULT_ENDPOINTS, default=wiz.readacross.endpoints_of_interest)
        wiz.readacross.endpoints_of_interest = selected

        st.markdown("**Similarity weights** (used by the RA agent to prioritize candidates)")
        sw1, sw2, sw3 = st.columns(3)
        wiz.readacross.weight_structure = sw1.slider("Structure", 0.0, 1.0, wiz.readacross.weight_structure, 0.05)
        wiz.readacross.weight_mechanism = sw2.slider("Mechanism", 0.0, 1.0, wiz.readacross.weight_mechanism, 0.05)
        wiz.readacross.weight_properties = sw3.slider("Properties", 0.0, 1.0, wiz.readacross.weight_properties, 0.05)

        st.markdown("**What do we mean by *read‑across potential*?**")
        st.info(READ_ACROSS_DEFINITION)

        wiz.readacross.justification_notes = st.text_area("Additional justification notes (optional)",
                                                          value=wiz.readacross.justification_notes, height=140)

        _nav(prev_enabled=True, next_enabled=True, next_label="Review")

    # STEP 4: Review
    elif wiz.current_step == 4:
        _header(4, total)
        st.subheader("Review Your Recipe")
        st.json(wiz.to_recipe())
        st.caption("You can download this recipe and re‑run it later for full traceability.")
        st.download_button("Download recipe.json", data=wiz.to_recipe_json(), file_name="oqt_wizard_recipe.json")

        _nav(prev_enabled=True, next_enabled=True, next_label="Run Analysis")

    # STEP 5: Run
    else:
        _header(5, total)
        st.subheader("Running Analysis")
        run_btn = st.button("Run now", type="primary", use_container_width=True)

        if run_btn:
            # 1) Retrieve data
            api = QSARToolboxAPI(base_url=wiz.setup.api_url)
            try:
                # Minimal non-breaking path: fetch everything, then filter according to scope
                raw_bundle = api.get_all_chemical_data(wiz.chemical.selected_chemical_id)
            except Exception as e:
                st.error(f"Data retrieval failed: {e}")
                return

            cleaned = clean_response_data(raw_bundle)
            filtered, drop_counts = _apply_scope_filters(cleaned, wiz)
            if any(drop_counts.values()):
                st.caption(f"Filtered records: {drop_counts}")

            # 2) Agents (respects your get_llm override if enabled via wizard)
            with st.spinner("Running agents…"):
                final_report, agent_texts = _run_pipeline(filtered, wiz.chemical.identifier or wiz.chemical.selected_display_name or "Target", wiz.setup.context, wiz)

            st.success("Analysis complete.")
            st.markdown("### Synthesized Report")
            st.write(final_report)

            # 3) Show data & agents
            st.markdown("---")
            st.markdown("### Data (cleaned & filtered)")
            st.json(filtered, expanded=False)

            st.markdown("### Specialist agent outputs")
            with st.expander("Open details"):
                for k, v in agent_texts.items():
                    st.markdown(f"**{k}**")
                    st.write(v)
                    st.markdown("---")

            # 4) Exports
            st.download_button("Download cleaned_data.json", data=json.dumps(filtered, indent=2), file_name="cleaned_data.json")
            st.download_button("Download synthesized_report.txt", data=final_report, file_name="synthesized_report.txt")

            # PDF export: use your exporter if present; otherwise fallback
            if wiz.setup.export_pdf:
                try:
                    if export_pdf_report:
                        pdf_path = export_pdf_report(filtered, final_report, wiz.to_recipe())
                    else:
                        # Simple fallback: write a plaintext PDF using reportlab (optional dependency)
                        # Keep it minimal to avoid breaking environments without reportlab
                        pdf_path = None
                    if pdf_path:
                        st.success("PDF report generated.")
                        st.markdown(f"[Download the PDF]({pdf_path})")
                    else:
                        st.info("PDF export is unavailable in this environment.")
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")

        _nav(prev_enabled=True, next_enabled=False, next_label="Done")
