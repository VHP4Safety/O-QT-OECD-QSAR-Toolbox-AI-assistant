
# oqt_assistant/components/guided_wizard.py
# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import logging
import streamlit as st

from oqt_assistant.utils.qsar_api import (
    QSARToolboxAPI,
    SearchOptions,
    QSARConnectionError,
    QSARTimeoutError,
    QSARResponseError,
    SLOW_PROFILER_GUIDS,
)
from oqt_assistant.utils.data_formatter import format_chemical_data
from oqt_assistant.utils.hit_selection import rank_hits_by_quality, select_hit_with_properties
from oqt_assistant.utils.qsar_models import derive_recommended_qsar_models, format_qsar_model_label
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
    st.title("🧪 O-QT: Guided Analysis Wizard")
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
                "include_qsar": True,
                "selected_simulator_guids": [],
                "selected_profiler_guids": [],
                "selected_qsar_model_guids": [],
                "include_slow_profilers": False,
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
        submitted = st.form_submit_button(
            "Next: Context & Goals →",
            type="primary",
            use_container_width=True,
        )
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
        submitted = st.form_submit_button(
            "Next: Scope & Methods →",
            type="primary",
            use_container_width=True,
        )
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
    selected_qsar_guids = d.get("selected_qsar_model_guids", [])

    with st.form(key="wiz_step4_form"):
        st.subheader("Data Retrieval Scope")
        st.info("Select the types of data to retrieve from QSAR Toolbox. Deselecting options can speed up analysis for focused queries.")
        include_properties = st.checkbox("Include Physicochemical Properties (Calculated)", value=d.get("include_properties", True))
        include_experimental = st.checkbox("Include Experimental Data (Measured)", value=d.get("include_experimental", True))
        include_profiling = st.checkbox("Include Profiling & Reactivity Alerts", value=d.get("include_profiling", True))
        include_qsar = st.checkbox("Run QSAR Model Predictions", value=d.get("include_qsar", True),
                                   help="Runs the QSAR model catalog and filters results to in-domain predictions.")

        if include_experimental:
            st.subheader("Experimental Data Filters (Exclusions)")
            st.info("Select data types to exclude from the analysis.")
            exclude_adme_tk = st.checkbox("Exclude ADME/Toxicokinetics (ADME/TK) records", value=d.get("exclude_adme_tk", False))
            exclude_mammalian_tox = st.checkbox("Exclude Mammalian Toxicity endpoints", value=d.get("exclude_mammalian_tox", False))
        else:
            exclude_adme_tk = False
            exclude_mammalian_tox = False

        st.subheader("Metabolism Simulation")
        available_simulators = st.session_state.get("available_simulators", [])
        if available_simulators:
            simulator_options = {
                sim.get("Caption", sim.get("Guid", f"Simulator {idx}")): sim.get("Guid")
                for idx, sim in enumerate(available_simulators)
                if isinstance(sim, dict) and sim.get("Guid")
            }
            all_labels = list(simulator_options.keys())

            sim_state_key = "wiz_simulator_labels"
            if sim_state_key not in st.session_state:
                stored_sim_guids = d.get("selected_simulator_guids") or list(simulator_options.values())
                initial_labels = [label for label, guid in simulator_options.items() if guid in stored_sim_guids]
                if not initial_labels:
                    initial_labels = all_labels.copy()
                st.session_state[sim_state_key] = initial_labels

            with st.expander("Select simulators", expanded=False):
                col_all, col_clear = st.columns(2)
                if col_all.form_submit_button("Select all simulators"):
                    st.session_state[sim_state_key] = all_labels.copy()
                    st.rerun()
                if col_clear.form_submit_button("Clear simulators"):
                    st.session_state[sim_state_key] = []
                    st.rerun()

                st.multiselect(
                    "Metabolism simulators",
                    options=all_labels,
                    key=sim_state_key,
                    help="Remove simulators to focus on a smaller subset.",
                )

            selected_labels = st.session_state.get(sim_state_key, [])
            if not selected_labels:
                st.warning("At least one simulator is required; reverting to the full set.")
                st.session_state[sim_state_key] = all_labels.copy()
                st.rerun()
            selected_labels = st.session_state.get(sim_state_key, [])

            selected_sim_guids = [simulator_options[label] for label in selected_labels]
            st.caption(f"Selected {len(selected_labels)} simulators.")
            d["selected_simulator_labels"] = selected_labels
        else:
            st.warning("No metabolism simulators available. Ensure QSAR Toolbox connection is active.")
            selected_sim_guids = []

        st.subheader("Profiler Selection")
        available_profilers = st.session_state.get("available_profilers", [])
        if available_profilers:
            prof_options = {
                (p.get("Caption") or p.get("Name") or p.get("Guid")): p.get("Guid")
                for p in available_profilers
                if isinstance(p, dict) and p.get("Guid")
            }
            prof_options = dict(sorted(prof_options.items(), key=lambda item: item[0].lower()))
            slow_set = {guid for guid in prof_options.values() if guid in SLOW_PROFILER_GUIDS}
            slow_labels = [label for label, guid in prof_options.items() if guid in slow_set]
            fast_labels = [label for label in prof_options.keys() if label not in slow_labels]

            labels_key = "wiz_profiler_labels"
            include_key = "wiz_profiler_include_slow"

            if labels_key not in st.session_state:
                stored_labels = d.get("profiler_custom_labels") or []
                initial_labels = [label for label in stored_labels if label in prof_options]
                if not initial_labels:
                    stored_guids = d.get("selected_profiler_guids") or []
                    initial_labels = [label for label, guid in prof_options.items() if guid in stored_guids]
                if not initial_labels:
                    initial_labels = fast_labels.copy()
                st.session_state[labels_key] = initial_labels

            if include_key not in st.session_state:
                st.session_state[include_key] = d.get("include_slow_profilers", False)

            with st.expander("Select profilers", expanded=False):
                col_fast, col_all, col_clear = st.columns(3)
                if col_fast.form_submit_button("Select fast profilers"):
                    st.session_state[labels_key] = fast_labels.copy()
                    st.session_state[include_key] = False
                    st.rerun()
                if col_all.form_submit_button("Select all profilers"):
                    st.session_state[labels_key] = list(prof_options.keys())
                    st.session_state[include_key] = True
                    st.rerun()
                if col_clear.form_submit_button("Clear profilers"):
                    st.session_state[labels_key] = []
                    st.session_state[include_key] = False
                    st.rerun()

                include_slow = st.checkbox(
                    "Include ECHA profilers (~20 s each)",
                    value=st.session_state.get(include_key, False),
                    help="Adds the slower ECHA profilers that typically take ~20 seconds each.",
                    key="wiz_profiler_include_checkbox",
                )
                if include_slow != st.session_state.get(include_key, False):
                    st.session_state[include_key] = include_slow
                    if not include_slow:
                        st.session_state[labels_key] = [label for label in st.session_state[labels_key] if label not in slow_labels]
                    else:
                        combined = st.session_state[labels_key] + [label for label in slow_labels if label not in st.session_state[labels_key]]
                        st.session_state[labels_key] = combined
                    st.rerun()

                st.multiselect(
                    "Profilers",
                    options=list(prof_options.keys()),
                    key=labels_key,
                )

            selected_labels = st.session_state.get(labels_key, [])
            if not selected_labels:
                st.warning("At least one profiler is required; reverting to the fast defaults.")
                st.session_state[labels_key] = fast_labels.copy()
                st.session_state[include_key] = False
                st.rerun()
            selected_labels = st.session_state.get(labels_key, [])

            include_echa = st.session_state.get(include_key, False) or any(label in slow_labels for label in selected_labels)
            d["profiler_custom_labels"] = selected_labels
            d["profiler_custom_include_slow"] = st.session_state.get(include_key, False)
            d["profiler_selection_mode"] = "Custom"

            selected_labels = list(dict.fromkeys(selected_labels))
            selected_prof_guids = [prof_options[label] for label in selected_labels]
            if not include_echa:
                selected_prof_guids = [guid for guid in selected_prof_guids if guid not in slow_set]

            if not selected_prof_guids and slow_set:
                selected_prof_guids = list(slow_set)
                include_echa = True
                selected_labels = [
                    label for label, guid in prof_options.items() if guid in slow_set
                ]

            selected_prof_guids = list(dict.fromkeys(selected_prof_guids))
            d["include_slow_profilers"] = include_echa
            d["selected_profiler_labels"] = selected_labels
            fast_count = sum(1 for label in selected_labels if prof_options[label] not in slow_set)
            slow_count = sum(1 for label in selected_labels if prof_options[label] in slow_set)
            st.caption(f"Selected {len(selected_labels)} profilers ({fast_count} fast · {slow_count} slow).")
            d["selected_profiler_guids"] = selected_prof_guids
        else:
            st.info("Profiler catalog not available yet (connect in sidebar); defaults will be used.")
            selected_prof_guids = []
            d["include_slow_profilers"] = False

        if include_qsar:
            st.subheader("QSAR Model Selection (Optional)")
            qsar_catalog = st.session_state.get("available_qsar_models", [])
            if not qsar_catalog and st.session_state.get("connection_status"):
                api_client = _get_api_client()
                if api_client:
                    try:
                        qsar_catalog = api_client.get_all_qsar_models_catalog()
                        st.session_state.available_qsar_models = qsar_catalog
                        recommended_models = derive_recommended_qsar_models(qsar_catalog)
                        st.session_state.recommended_qsar_models = recommended_models
                        st.session_state.recommended_qsar_model_guids = [
                            entry.get("Guid") for entry in recommended_models if entry.get("Guid")
                        ]
                    except Exception as exc:
                        st.warning(f"Could not load QSAR model catalog: {exc}")
                        qsar_catalog = []
            if qsar_catalog:
                option_map: Dict[str, str] = {}
                for entry in qsar_catalog:
                    guid = entry.get("Guid")
                    if not guid:
                        continue
                    option_map[format_qsar_model_label(entry)] = guid

                labels = list(option_map.keys())
                recommended_guids = st.session_state.get("recommended_qsar_model_guids", [])
                recommended_labels = [label for label, guid in option_map.items() if guid in recommended_guids]

                state_key = "wiz_qsar_model_multiselect"
                if state_key not in st.session_state:
                    base_selection = selected_qsar_guids or recommended_guids
                    initial_labels = [label for label, guid in option_map.items() if guid in base_selection]
                    if not initial_labels and recommended_labels:
                        initial_labels = recommended_labels.copy()
                    if not initial_labels:
                        initial_labels = labels[: min(8, len(labels))]
                    st.session_state[state_key] = initial_labels

                with st.expander("Select QSAR models", expanded=False):
                    col_rec, col_all, col_clear = st.columns(3)
                    if col_rec.form_submit_button("Recommended set"):
                        st.session_state[state_key] = recommended_labels or labels.copy()
                        st.rerun()
                    if col_all.form_submit_button("Select all models"):
                        st.session_state[state_key] = labels.copy()
                        st.rerun()
                    if col_clear.form_submit_button("Clear models"):
                        st.session_state[state_key] = []
                        st.rerun()

                    st.multiselect(
                        "QSAR models",
                        options=labels,
                        key=state_key,
                        help="Choose one or more QSAR models. Leave empty to skip QSAR predictions.",
                    )

                selected_qsar_guids = [option_map[label] for label in st.session_state[state_key] if label in option_map]
                if not selected_qsar_guids:
                    st.info("No QSAR models selected. QSAR predictions will be skipped for this guided run.")
                else:
                    preview = st.session_state[state_key][:3]
                    suffix = f" … (+{len(st.session_state[state_key]) - len(preview)} more)" if len(st.session_state[state_key]) > len(preview) else ""
                    st.caption(f"Selected {len(st.session_state[state_key])} QSAR models: " + "; ".join(preview) + suffix)
                if recommended_labels:
                    rec_preview = recommended_labels[:3]
                    remaining = len(recommended_labels) - len(rec_preview)
                    suffix = f" … (+{remaining} more)" if remaining > 0 else ""
                    st.caption("Recommended fast QSAR set: " + "; ".join(rec_preview) + suffix)
            else:
                st.info("QSAR model catalog is unavailable. Ensure the QSAR Toolbox connection is active in Step 1.")
                selected_qsar_guids = []
        else:
            selected_qsar_guids = []

        next_submitted = st.form_submit_button(
            "Next: Read-Across →",
            type="primary",
            use_container_width=True,
        )

    if next_submitted:
        d["include_properties"] = include_properties
        d["include_experimental"] = include_experimental
        d["include_profiling"] = include_profiling
        d["include_qsar"] = include_qsar
        d["selected_simulator_guids"] = selected_sim_guids
        d["selected_profiler_guids"] = selected_prof_guids
        d["include_slow_profilers"] = d.get("include_slow_profilers", False)
        d["selected_qsar_model_guids"] = selected_qsar_guids
        d["exclude_adme_tk"] = exclude_adme_tk
        d["exclude_mammalian_tox"] = exclude_mammalian_tox
        if not (include_properties or include_experimental or include_profiling or include_qsar):
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
    # MODIFIED: Added definition of Read-Across Potential (Addressing Reviewer #1)
    read_across_help = """
    **Read-Across Potential (Definition):** The likelihood that the toxicity profile of the target chemical can be accurately predicted by using existing data from structurally or mechanistically similar chemicals (analogues).
    This directs the AI agents to prioritize identifying data gaps and developing a detailed strategy based on the selected approach and similarity basis.
    """

    with st.form(key="wiz_step5_form"):
        prioritize_read_across = st.checkbox(
            "Prioritize Read-Across Strategy Development",
            value=d.get("prioritize_read_across", True),
            help=read_across_help
        )
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
        submitted = st.form_submit_button(
            "Next: Review & Run →",
            type="primary",
            use_container_width=True,
        )
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
    if d.get("include_qsar", True):
        qsar_guids = d.get("selected_qsar_model_guids") or []
        if qsar_guids:
            st.markdown(f"- Included: QSAR model predictions ({len(qsar_guids)} models)")
        else:
            st.markdown("- QSAR predictions: Enabled (no models selected; step will be skipped)")
    else:
        st.markdown("- QSAR predictions: Skipped")

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
    if d.get("include_slow_profilers"):
        st.markdown("  - *Includes ECHA profilers (expect ~20 s per profiler)*")
    else:
        st.markdown("  - Skipping ECHA profilers (opt-in available in Scope step)")

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
        final_context = " ".join(context_parts) or "General chemical hazard assessment."

        config = {
            "identifier": d["chemical_identifier"],
            "search_type": d["search_type"],
            "context": final_context,
            "simulator_guids": d["selected_simulator_guids"],
            "selected_profiler_guids": d["selected_profiler_guids"],
            "include_slow_profilers": d.get("include_slow_profilers", False),
            "selected_qsar_model_guids": d.get("selected_qsar_model_guids", []),
            "include_properties": d["include_properties"],
            "include_experimental": d["include_experimental"],
            "include_profiling": d["include_profiling"],
            "include_qsar": d.get("include_qsar", True),
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
