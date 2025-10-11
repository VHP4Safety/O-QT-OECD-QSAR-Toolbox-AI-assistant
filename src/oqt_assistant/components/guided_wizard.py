
# oqt_assistant/components/guided_wizard.py
# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import logging
import streamlit as st

from oqt_assistant.utils.qsar_api import QSARToolboxAPI, SearchOptions, QSARConnectionError, QSARTimeoutError, QSARResponseError
from oqt_assistant.utils.data_formatter import format_chemical_data
from oqt_assistant.utils.hit_selection import rank_hits_by_quality, select_hit_with_properties
from oqt_assistant.components.search import REGULATORY_ENDPOINTS
from oqt_assistant.utils.structure_3d import render_smiles_3d

try:
    from oqt_assistant.components.search import REGULATORY_ENDPOINTS
except ImportError:
    REGULATORY_ENDPOINTS = [
        "Ecotoxicity (Aquatic/Terrestrial)",
        "Environmental Fate (Persistence/Bioaccumulation)",
        "Genotoxicity/Mutagenicity",
        "Carcinogenicity",
        "Reproductive/Developmental Toxicity",
        "Neurotoxicity",
        "Endocrine Disruption",
        "Skin/Eye Irritation/Corrosion",
        "Sensitization (Skin/Respiratory)",
    ]

logger = logging.getLogger(__name__)


def run_guided_wizard(*, ping_qsar=None, estimate_llm_cost=None, on_run_pipeline=None, get_llm_models=None) -> None:
    st.title("🧪 O'QT: Guided Analysis Wizard")
    _init_wizard_state()
    wiz = st.session_state["wiz"]
    _render_stepper_header(wiz["current_step"], wiz["total_steps"], wiz["step_titles"])

    if wiz["current_step"] == 1:
        _step_1_setup_configuration(get_llm_models, ping_qsar)  # pass ping
    elif wiz["current_step"] == 2:
        _step_2_chemical_identification()
    elif wiz["current_step"] == 3:
        _step_3_analysis_context()
    elif wiz["current_step"] == 4:
        _step_4_scope_methodology()
    elif wiz["current_step"] == 5:
        _step_5_read_across_strategy()
    elif wiz["current_step"] == 6:
        _step_6_review_and_run(on_run_pipeline)
    else:
        _goto_step(1)


def _init_wizard_state():
    if "wiz" not in st.session_state:
        st.session_state["wiz"] = {
            "current_step": 1,
            "total_steps": 6,
            "step_titles": {1: "Setup", 2: "Chemical ID", 3: "Context & Goals", 4: "Scope & Methods", 5: "Read-Across", 6: "Review & Run"},
            "data": {
                "llm_temperature_override": None,
                "llm_max_tokens_override": None,
                "reasoning_effort": "medium",
                "chemical_identifier": "",
                "search_type": "name",
                "chemical_resolved": False,
                "resolved_chemical_data": None,
                "case_label": "",
                "analysis_focus": "",
                "endpoints_of_interest": [],
                "include_properties": True,
                "include_experimental": True,
                "include_profiling": True,
                "selected_simulator_guids": [],
                "selected_profiler_guids": [],
                "exclude_adme_tk": False,
                "exclude_mammalian_tox": False,
                "prioritize_read_across": True,
                "rax_strategy": "Hybrid (Analogue and Category)",
                "rax_similarity_basis": "Combined (Structural, Mechanistic, and Properties)",
            },
            "errors": {},
            "search_results": [],
        }


def _get_wiz() -> Dict[str, Any]:
    if "wiz" not in st.session_state:
        _init_wizard_state()
    return st.session_state["wiz"]


def _set_errors(errors: Dict[str, str]):
    _get_wiz()["errors"] = errors


def _clear_errors():
    _get_wiz()["errors"] = {}


def _goto_step(i: int):
    wiz = _get_wiz()
    i = max(1, min(i, wiz["total_steps"]))
    wiz["current_step"] = i
    _clear_errors()


def _render_stepper_header(current: int, total: int, titles: Dict[int, str]):
    style_done = "text-align: center; color: #4CAF50; font-weight: bold;"
    style_current = "text-align: center; color: #2196F3; font-weight: bold; font-size: 1.1em; border-bottom: 3px solid #2196F3; padding-bottom: 5px;"
    style_future = "text-align: center; color: #9E9E9E;"
    cols = st.columns(total)
    for i in range(1, total + 1):
        with cols[i - 1]:
            title = titles.get(i, f"Step {i}")
            if i < current:
                st.markdown(f"<div style='{style_done}'>✅<br/>{title}</div>", unsafe_allow_html=True)
            elif i == current:
                st.markdown(f"<div style='{style_current}'>📍<br/>{title}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='{style_future}'>⏳<br/>{title}</div>", unsafe_allow_html=True)
    st.progress((current - 1) / (total - 1) if total > 1 else 1.0)
    st.header(f"Step {current}: {titles.get(current)}")


def _render_nav(back: bool = True, next: bool = True, next_label="Next →", back_step: Optional[int] = None):
    st.markdown("---")
    c1, _, c3 = st.columns([2, 6, 2])
    with c1:
        if back and st.button("← Back", key=f"wiz_nav_back_{_get_wiz()['current_step']}", width="stretch"):
            _goto_step(back_step or (_get_wiz()["current_step"] - 1))
            st.rerun()
    with c3:
        if next and st.button(next_label, type="primary", key=f"wiz_nav_next_{_get_wiz()['current_step']}", width="stretch"):
            _goto_step(_get_wiz()["current_step"] + 1)
            st.rerun()


def _validate_step_2(d: Dict[str, Any]) -> Dict[str, str]:
    errors = {}
    if not d.get("chemical_identifier", "").strip():
        errors["chemical_identifier"] = "Please enter a chemical identifier."
    if not d.get("chemical_resolved"):
        errors["chemical_resolved"] = "Please search and confirm the chemical selection before proceeding."
    return errors


def _render_chem_identity_card(basic: Dict[str, Any]) -> None:
    """Compact identity card for the currently resolved chemical."""
    if not isinstance(basic, dict):
        st.info("No identification data available.")
        return
    c1, c2, c3 = st.columns([2, 1, 3])
    with c1:
        st.markdown(f"**Name:** {basic.get('Name') or basic.get('IUPACName') or 'N/A'}")
        st.markdown(f"**IUPAC:** {basic.get('IUPACName', 'N/A')}")
    with c2:
        st.markdown(f"**CAS:** `{basic.get('Cas', 'N/A')}`")
        st.markdown(f"**ChemID:** `{basic.get('ChemId', 'N/A')}`")
    with c3:
        st.markdown(f"**SMILES:** `{basic.get('Smiles', 'N/A')}`")


def _step_1_setup_configuration(get_llm_models, ping_qsar):
    wiz = _get_wiz()
    d = wiz["data"]

    st.info("This wizard uses the QSAR Toolbox and LLM configuration defined in the sidebar. Ensure settings are correct before proceeding.")

    qsar_config = st.session_state.get("qsar_config", {})
    llm_config = st.session_state.get("llm_config", {})

    is_qsar_ready = qsar_config.get("config_complete", False)
    is_llm_ready = llm_config.get("config_complete", False)

    st.subheader("Configuration Status (Sidebar)")
    if is_qsar_ready:
        st.success(f"✅ QSAR Toolbox API URL configured: {qsar_config.get('api_url')}")
    else:
        st.error("❌ QSAR Toolbox API URL missing.")

    llm_models = get_llm_models() if callable(get_llm_models) else {}
    provider = llm_config.get("provider")
    model_name = llm_config.get("model_name")
    if is_llm_ready and provider in llm_models and model_name in llm_models.get(provider, {}):
        st.success(f"✅ LLM configured: {provider} / {model_name}")
    else:
        st.error(f"❌ LLM configuration incomplete (provider/model/key).")

    st.subheader("Connection Status")
    connection_status = st.session_state.get("connection_status")
    colA, colB = st.columns([3, 1])
    with colA:
        if connection_status is True:
            st.success("✅ Connected to QSAR Toolbox API.")
        elif connection_status is False:
            st.error("❌ Failed to connect to QSAR Toolbox API.")
        else:
            st.info("ℹ️ Status pending.")
    with colB:
        if st.button("Check Connection", width="stretch") and callable(ping_qsar):
            ok, msg = ping_qsar(qsar_config.get("api_url"))
            st.session_state.connection_status = bool(ok)
            (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")

    st.subheader("LLM Parameter Overrides (Optional)")
    st.info("Adjust LLM parameters for this specific run. Leave unchecked to use sidebar settings.")
    
    # Check if GPT-5 is selected
    model_name = llm_config.get("model_name", "")
    is_gpt5 = "gpt-5" in model_name.lower()
    
    # For non–GPT‑5 models, show Temperature (default 0.15)
    if not is_gpt5:
        use_temp_override = st.checkbox("Override Temperature", value=(d.get("llm_temperature_override") is not None), key="wiz_step1_use_temp_override")
        d["llm_temperature_override"] = st.slider("Temperature (Creativity)", 0.0, 1.0, value=(d.get("llm_temperature_override") or 0.15), step=0.05, key="wiz_step1_temp_override") if use_temp_override else None
    else:
        st.caption("GPT‑5 ignores temperature; using reasoning controls instead.")
        d["llm_temperature_override"] = None  # Clear temperature for GPT-5
        d["reasoning_effort"] = st.select_slider(
            "Reasoning Effort (GPT‑5)",
            options=["minimal", "medium", "high"],
            value=d.get("reasoning_effort", "medium"),
            help="Controls how much thinking the model does before replying; billed as output tokens.",
            key="wiz_step1_reasoning_effort"
        )
    
    use_tokens_override = st.checkbox("Override Max Output Tokens", value=(d.get("llm_max_tokens_override") is not None), key="wiz_step1_use_tokens_override")
    d["llm_max_tokens_override"] = st.number_input("Max Output Tokens", min_value=512, max_value=32000, value=(d.get("llm_max_tokens_override") or 10000), step=512, key="wiz_step1_tokens_override") if use_tokens_override else None
    
    if is_gpt5:
        st.caption("Note: GPT‑5 'reasoning' tokens are charged as **output** tokens.")

    can_proceed = is_qsar_ready and is_llm_ready and (st.session_state.get("connection_status") is True)
    if not can_proceed:
        st.warning("Resolve configuration/connection issues before continuing.")

    _render_nav(back=False, next=can_proceed, next_label="Continue to Chemical Identification →")


def _get_api_client() -> Optional[QSARToolboxAPI]:
    api_url = st.session_state.get("qsar_config", {}).get("api_url")
    if not api_url:
        st.error("QSAR API URL not configured.")
        return None
    return QSARToolboxAPI(base_url=api_url, timeout=(5, 15))


def _perform_chemical_search(identifier: str, search_type: str):
    api_client = _get_api_client()
    if not api_client:
        return
    wiz = _get_wiz()
    wiz["data"]["chemical_resolved"] = False
    wiz["data"]["resolved_chemical_data"] = None
    wiz["search_results"] = []
    wiz["data"]["raw_hits"] = []
    wiz["data"]["best_hit_notes"] = []
    wiz["data"]["best_hit_properties"] = None
    wiz["data"]["validated_hit_index"] = None
    try:
        with st.spinner("Searching for chemical..."):
            if search_type in ("name", "cas"):
                if hasattr(api_client.search_by_name, "cache_clear"):
                    api_client.search_by_name.cache_clear()
                search_result = api_client.search_by_name(identifier, search_option=SearchOptions.EXACT_MATCH)
            else:
                search_result = api_client.search_by_smiles(identifier)
        if not search_result:
            st.warning("No exact match found. Try refining the identifier.")
            return

        hits = search_result if isinstance(search_result, list) else [search_result]
        ranked_hits = rank_hits_by_quality(identifier, hits)
        formatted = [format_chemical_data(r) for r in ranked_hits]
        wiz["search_results"] = formatted
        wiz["data"]["raw_hits"] = ranked_hits

        if not formatted:
            st.warning("No usable matches after applying ranking filters.")
            return

        try:
            best_basic, best_props, best_chem_id, notes = select_hit_with_properties(
                api_client, identifier, ranked_hits, logger=logger
            )
            wiz["data"]["best_hit_properties"] = best_props
            wiz["data"]["best_hit_notes"] = notes
            if len(formatted) == 1:
                wiz["data"]["chemical_resolved"] = True
                wiz["data"]["resolved_chemical_data"] = best_basic
                wiz["data"]["validated_hit_index"] = 0
                st.success("Chemical automatically resolved to a validated QSAR Toolbox record.")
            else:
                st.info("Multiple matches found. Please select the correct entry below.")
        except RuntimeError as exc:
            st.warning(f"Matches were found, but none completed Toolbox calculator validation: {exc}")
    except (QSARConnectionError, QSARTimeoutError, QSARResponseError) as e:
        st.error(f"Search failed: {str(e)}")


def _display_search_results():
    wiz = _get_wiz()
    results = wiz["search_results"]
    raw_hits = wiz["data"].get("raw_hits", [])
    if not results or len(results) <= 1:
        return
    st.subheader(f"Found {len(results)} matches. Please select one:")

    options = {}
    for i, chem in enumerate(results):
        name = chem.get("Name") or chem.get("IUPACName") or f"ChemID {chem.get('ChemId', 'N/A')}"
        cas = chem.get("Cas") or "N/A"
        smi = chem.get("Smiles") or ""
        prefix = "⭐ " if i == 0 else ""
        label = f"{prefix}{name}  •  CAS {cas}{('  •  ' + smi) if smi else ''}"
        options[label] = i

    default_index = wiz["data"].get("validated_hit_index")
    radio_index = default_index if isinstance(default_index, int) and 0 <= default_index < len(results) else 0

    selected_label = st.radio("Select the correct chemical:", list(options.keys()), index=radio_index, key="wiz_step2_selection")
    if selected_label:
        selected_index = options[selected_label]
        if selected_index != wiz["data"].get("validated_hit_index"):
            api_client = _get_api_client()
            if api_client:
                prioritized_hits = []
                if raw_hits and 0 <= selected_index < len(raw_hits):
                    prioritized_hits.append(raw_hits[selected_index])
                    prioritized_hits.extend(hit for idx, hit in enumerate(raw_hits) if idx != selected_index)
                else:
                    prioritized_hits = raw_hits or []
                if prioritized_hits:
                    try:
                        basic, props, chem_id, notes = select_hit_with_properties(
                            api_client,
                            wiz["data"].get("chemical_identifier", ""),
                            prioritized_hits,
                            logger=logger,
                        )
                        validated_index = next(
                            (i for i, hit in enumerate(raw_hits) if hit.get("ChemId") == chem_id),
                            selected_index,
                        )
                        if validated_index != selected_index:
                            st.warning("The chosen entry could not be validated; switched to the closest valid match.")
                        wiz["data"]["resolved_chemical_data"] = basic
                        wiz["data"]["best_hit_properties"] = props
                        wiz["data"]["best_hit_notes"] = notes
                        wiz["data"]["chemical_resolved"] = True
                        wiz["data"]["validated_hit_index"] = validated_index
                    except RuntimeError as exc:
                        st.error(f"Unable to validate the selected entry: {exc}")
                        wiz["data"]["chemical_resolved"] = False
                        wiz["data"]["resolved_chemical_data"] = None
                else:
                    st.warning("No Toolbox hits available to validate selection.")
            else:
                st.error("QSAR Toolbox API is not configured. Please update settings in Step 1.")

        selected_chemical = wiz["data"].get("resolved_chemical_data") or results[selected_index]
        st.info("Preview of selected chemical:")
        _render_chem_identity_card(selected_chemical)


def _step_2_chemical_identification():
    wiz = _get_wiz()
    d = wiz["data"]
    errs = wiz["errors"]

    search_type = st.radio(
        "Search By",
        options=["name", "smiles", "cas"],
        format_func=lambda x: {"name": "Chemical Name (Exact)", "smiles": "SMILES Notation", "cas": "CAS Number"}[x],
        index=["name", "smiles", "cas"].index(d.get("search_type", "name")),
        horizontal=True,
        key="wiz_step2_search_type",
    )
    d["search_type"] = st.session_state.wiz_step2_search_type

    identifier = st.text_input("Identifier", value=d.get("chemical_identifier", ""), placeholder="e.g., retinol or 68-26-8 or a SMILES")
    d["chemical_identifier"] = identifier

    if errs.get("chemical_identifier"):
        st.error(errs["chemical_identifier"])

    if st.button("🔍 Search QSAR Toolbox", key="wiz_step2_search_btn"):
        if not d["chemical_identifier"]:
            st.warning("Please enter an identifier.")
        else:
            _perform_chemical_search(d["chemical_identifier"], d["search_type"])
            st.rerun()

    _display_search_results()

    if d.get("chemical_resolved"):
        chem_data = d.get("resolved_chemical_data", {}) or {}
        st.success("✅ Chemical Confirmed")
        _render_chem_identity_card(chem_data)
        notes = d.get("best_hit_notes") or d.get("selection_notes")
        if notes:
            st.caption("Validation notes: " + " | ".join(notes))

        smiles = chem_data.get("Smiles") or ""
        with st.expander("3D Preview", expanded=True):
            if smiles:
                try:
                    render_smiles_3d(smiles, height=240, width=0) # width=0 lets Streamlit size it
                except Exception as e:
                    st.warning(f"3D preview unavailable: {e}")
            else:
                st.caption("No SMILES available for this record.")

    if errs.get("chemical_resolved"):
        st.error(errs["chemical_resolved"])

    with st.form(key="wiz_step2_nav_form"):
        submitted = st.form_submit_button("Next: Context & Goals →", type="primary", width="stretch")
    if submitted:
        errors = _validate_step_2(d)
        if errors:
            _set_errors(errors)
            st.rerun()
        else:
            _goto_step(3)
            st.rerun()
    _render_nav(back_step=1, next=False)


def _step_3_analysis_context():
    wiz = _get_wiz()
    d = wiz["data"]
    with st.form(key="wiz_step3_form"):
        case_label = st.text_input("Analysis Label / Case ID (Optional)", value=d.get("case_label", ""))
        analysis_focus = st.text_area(
            "Custom Context/Concerns (Optional)",
            value=d.get("analysis_focus", ""),
            placeholder="e.g., neurotoxicity concerns, environmental fate focus...",
        )
        st.markdown("**Regulatory Endpoints of Interest (Optional)**")
        selected_endpoints = st.multiselect("Select endpoints:", options=REGULATORY_ENDPOINTS, default=d.get("endpoints_of_interest", []))
        submitted = st.form_submit_button("Next: Scope & Methods →", type="primary", width="stretch")
    if submitted:
        d["case_label"] = case_label.strip()
        d["analysis_focus"] = analysis_focus.strip()
        d["endpoints_of_interest"] = selected_endpoints
        _goto_step(4)
        st.rerun()
    _render_nav(back_step=2, next=False)


def _step_4_scope_methodology():
    wiz = _get_wiz()
    d = wiz["data"]
    with st.form(key="wiz_step4_form"):
        st.subheader("Data Retrieval Scope")
        st.info("Select the types of data to retrieve from QSAR Toolbox. Deselecting options can speed up analysis for focused queries.")
        include_properties = st.checkbox("Include Physicochemical Properties (Calculated)", value=d.get("include_properties", True))
        include_experimental = st.checkbox("Include Experimental Data (Measured)", value=d.get("include_experimental", True))
        include_profiling = st.checkbox("Include Profiling & Reactivity Alerts", value=d.get("include_profiling", True))

        if include_experimental:
            st.subheader("Experimental Data Filters (Exclusions)")
            st.info("Select data types to exclude from the analysis.")
            exclude_adme_tk = st.checkbox("Exclude ADME/Toxicokinetics (ADME/TK) records", value=d.get("exclude_adme_tk", False))
            exclude_mammalian_tox = st.checkbox("Exclude Mammalian Toxicity endpoints", value=d.get("exclude_mammalian_tox", False))
        else:
            exclude_adme_tk = False
            exclude_mammalian_tox = False

        st.subheader("Metabolism Simulation (Optional)")
        available_simulators = st.session_state.get("available_simulators", [])
        if available_simulators:
            sim_options = {s["Caption"]: s["Guid"] for s in available_simulators}
            current = [cap for cap, gid in sim_options.items() if gid in d.get("selected_simulator_guids", [])]
            selected_caps = st.multiselect("Select Metabolism Simulators", options=list(sim_options.keys()), default=current)
            selected_sim_guids = [sim_options[c] for c in selected_caps]
        else:
            st.warning("No metabolism simulators available. Ensure QSAR Toolbox connection is active.")
            selected_sim_guids = []

        st.subheader("Profiler Selection (Optional)")
        available_profilers = st.session_state.get("available_profilers", [])
        if available_profilers:
            prof_options = {p.get("Caption") or p.get("Name"): p.get("Guid") for p in available_profilers if isinstance(p, dict) and p.get("Guid")}
            current_prof = [cap for cap, gid in prof_options.items() if gid in d.get("selected_profiler_guids", [])]
            selected_prof_caps = st.multiselect(
                "Select profilers to run (leave empty to use the balanced default set)",
                options=list(prof_options.keys()),
                default=current_prof,
                help="Use this to run additional profilers or narrow down to a specific subset.",
            )
            selected_prof_guids = [prof_options[c] for c in selected_prof_caps]
        else:
            st.info("Profiler catalog not available yet (connect in sidebar), defaults will be used.")
            selected_prof_guids = []

        submitted = st.form_submit_button("Next: Read-Across →", type="primary", width="stretch")

    if submitted:
        d["include_properties"] = include_properties
        d["include_experimental"] = include_experimental
        d["include_profiling"] = include_profiling
        d["selected_simulator_guids"] = selected_sim_guids
        d["selected_profiler_guids"] = selected_prof_guids
        d["exclude_adme_tk"] = exclude_adme_tk
        d["exclude_mammalian_tox"] = exclude_mammalian_tox
        if not (include_properties or include_experimental or include_profiling):
            st.error("Please select at least one data retrieval scope.")
        else:
            _goto_step(5)
            st.rerun()
    _render_nav(back_step=3, next=False)


def _step_5_read_across_strategy():
    wiz = _get_wiz()
    d = wiz["data"]
    RAX_STRATEGY_OPTIONS = ["Analogue Approach (One-to-one)", "Category Approach (Group of similar chemicals)", "Hybrid (Analogue and Category)"]
    RAX_BASIS_OPTIONS = ["Structural Similarity", "Mechanistic Similarity (Reactivity/Profiling)", "Physicochemical Properties", "Combined (Structural, Mechanistic, and Properties)"]
    with st.form(key="wiz_step5_form"):
        prioritize_read_across = st.checkbox("Prioritize Read-Across Strategy Development", value=d.get("prioritize_read_across", True))
        if prioritize_read_across:
            st.subheader("Strategy Details")
            rax_strategy = st.selectbox("Select preferred strategy:", options=RAX_STRATEGY_OPTIONS, index=RAX_STRATEGY_OPTIONS.index(d.get("rax_strategy")) if d.get("rax_strategy") in RAX_STRATEGY_OPTIONS else 2)
            rax_similarity_basis = st.selectbox(
                "Select the primary basis for identifying analogues:",
                options=RAX_BASIS_OPTIONS,
                index=RAX_BASIS_OPTIONS.index(d.get("rax_similarity_basis")) if d.get("rax_similarity_basis") in RAX_BASIS_OPTIONS else 3,
            )
        else:
            rax_strategy = None
            rax_similarity_basis = None
        submitted = st.form_submit_button("Next: Review & Run →", type="primary", width="stretch")
    if submitted:
        d["prioritize_read_across"] = prioritize_read_across
        d["rax_strategy"] = rax_strategy
        d["rax_similarity_basis"] = rax_similarity_basis
        _goto_step(6)
        st.rerun()
    _render_nav(back_step=4, next=False)


def _step_6_review_and_run(on_run_pipeline):
    wiz = _get_wiz()
    d = wiz["data"]
    st.info("Review the analysis plan. If correct, click 'Run Analysis'.")

    st.subheader("Analysis Plan Summary")
    chem_raw = d.get("resolved_chemical_data", {}) or {}
    chem_data = chem_raw if chem_raw.get("ChemId") else format_chemical_data(chem_raw)
    st.markdown("**Target Chemical:**")
    _render_chem_identity_card(chem_data)

    st.markdown(f"**Case Label:** {d.get('case_label') or 'N/A'}")
    st.markdown(f"**Analysis Focus:** {d.get('analysis_focus') or 'General hazard assessment'}")
    endpoints = d.get("endpoints_of_interest")
    if endpoints:
        st.markdown(f"**Endpoints of Interest:** {', '.join(endpoints)}")

    st.markdown("**Scope & Filters:**")
    if d.get("include_properties"):
        st.markdown("- Included: Physicochemical Properties")
    if d.get("include_experimental"):
        st.markdown("- Included: Experimental Data")
        if d.get("exclude_adme_tk"):
            st.markdown("  - *Excluded: ADME/TK Data*")
        if d.get("exclude_mammalian_tox"):
            st.markdown("  - *Excluded: Mammalian Toxicity Data*")
    if d.get("include_profiling"):
        st.markdown("- Included: Profiling & Reactivity")

    simulators = d.get("selected_simulator_guids")
    if simulators:
        available_simulators = st.session_state.get("available_simulators", [])
        simulator_map = {s["Guid"]: s["Caption"] for s in available_simulators}
        simulator_names = [simulator_map.get(guid, f"GUID: {guid}") for guid in simulators]
        st.markdown(f"**Metabolism Simulators:** {', '.join(simulator_names)}")
    else:
        st.markdown("**Metabolism Simulation:** Skipped")

    profilers = d.get("selected_profiler_guids")
    if profilers:
        available_profilers = st.session_state.get("available_profilers", [])
        profiler_map = {p.get("Guid"): (p.get("Caption") or p.get("Name") or p.get("Guid")) for p in available_profilers if isinstance(p, dict)}
        profiler_names = [profiler_map.get(guid, f"GUID: {guid}") for guid in profilers]
        st.markdown(f"**Profilers:** {', '.join(profiler_names)}")
    else:
        st.markdown("**Profilers:** Balanced default set")

    st.subheader("Configuration")
    llm_config = st.session_state.get("llm_config", {})
    st.markdown(f"**LLM Provider (Sidebar):** {llm_config.get('provider', 'N/A')}")
    st.markdown(f"**LLM Model (Sidebar):** {llm_config.get('model_name', 'N/A')}")
    if d.get("llm_temperature_override") is not None:
        st.markdown(f"**LLM Temperature (Override):** {d.get('llm_temperature_override')}")
    if d.get("llm_max_tokens_override") is not None:
        st.markdown(f"**LLM Max Tokens (Override):** {d.get('llm_max_tokens_override')}")

    st.markdown("---")
    run_clicked = st.button("🚀 Run Analysis", type="primary", width="stretch", key="wiz_step6_run")

    if run_clicked and callable(on_run_pipeline):
        context_parts = []
        if endpoints:
            context_parts.append(f"Focus the analysis on the following regulatory endpoints: {', '.join(endpoints)}.")
        if d.get("prioritize_read_across"):
            context_parts.append("Critically evaluate the data gaps and develop a detailed read-across strategy.")
        if d.get("analysis_focus"):
            context_parts.append(f"Address these specific user concerns: {d.get('analysis_focus')}")
        final_context = " ".join(context_parts) or "General chemical hazard assessment (Guided Workflow)."

        config = {
            "identifier": d["chemical_identifier"],
            "search_type": d["search_type"],
            "context": final_context,
            "simulator_guids": d["selected_simulator_guids"],
            "selected_profiler_guids": d["selected_profiler_guids"],
            "include_properties": d["include_properties"],
            "include_experimental": d["include_experimental"],
            "include_profiling": d["include_profiling"],
            "exclude_adme_tk": d["exclude_adme_tk"],
            "exclude_mammalian_tox": d["exclude_mammalian_tox"],
            "rax_strategy": d.get("rax_strategy"),
            "rax_similarity_basis": d.get("rax_similarity_basis"),
            "llm_temperature_override": d.get("llm_temperature_override"),
            "llm_max_tokens_override": d.get("llm_max_tokens_override"),
            "reasoning_effort": d.get("reasoning_effort"),
            "resolved_chemical_data": chem_data,
            "case_label": d["case_label"],
        }
        try:
            on_run_pipeline(config)
        except Exception as e:
            st.error(f"Could not start analysis: {e}")
    _render_nav(back_step=5, next=False)
