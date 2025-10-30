# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

import streamlit as st
import sys
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import logging # Ensure logging is imported

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file (adjust path for src layout)
# This is kept for fallback/default values if users prefer environment variables
try:
    # Adjust the pathfinding logic to be more robust
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming app.py is in src/oqt_assistant, the root is two levels up from there (outside src)
    # If running from the package installation, the structure might differ slightly, but this aims for the development structure root.
    project_root = os.path.dirname(os.path.dirname(current_dir)) 
    dotenv_path = os.path.join(project_root, '.env')
    
    # Fallback to the original logic if the above assumption is wrong in some environments
    if not os.path.exists(dotenv_path):
         fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
         if os.path.exists(fallback_path):
             dotenv_path = fallback_path

    load_dotenv(dotenv_path)
except Exception:
    print("Could not load .env file (optional).")

# Use absolute imports based on the package structure
from oqt_assistant.utils.qsar_api import (
    QSARToolboxAPI, QSARConnectionError, QSARTimeoutError, QSARResponseError,
    SearchOptions, SLOW_PROFILER_GUIDS
)
# Import new agent functions (Updated signatures)
from oqt_assistant.utils.llm_utils import (
    analyze_chemical_context,
    analyze_physical_properties,
    analyze_environmental_fate,
    analyze_profiling_reactivity,
    analyze_experimental_data,
    analyze_metabolism,
    analyze_qsar_predictions,
    analyze_read_across,
    synthesize_report
)
from oqt_assistant.components.search import render_search_section
from oqt_assistant.components.results import (
    render_results_section,
    render_download_section,
)

# Removed unused import of wizard

# NEW IMPORT
from oqt_assistant.utils.pdf_generator import generate_pdf_report
from oqt_assistant.components.guided_wizard import run_guided_wizard
# Import data formatter and filters needed in execute_analysis
# UPDATED IMPORT: Added format_chemical_data
from oqt_assistant.utils.data_formatter import process_experimental_metadata, process_qsar_predictions
from oqt_assistant.utils.qsar_models import (
    run_qsar_predictions,
    derive_recommended_qsar_models,
    format_qsar_model_label,
)
from oqt_assistant.utils.filters import filter_experimental_records # NEW IMPORT
from oqt_assistant.utils.qprf_enrichment import QPRFEnricher
from oqt_assistant.utils.key_studies import KeyStudyCollector
from oqt_assistant.utils.hit_selection import select_hit_with_properties


# Define the maximum number of experimental records to send to LLM agents
MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA = 500
# Define maximum number of metabolites to process PER SIMULATOR (to prevent overwhelming the LLM and UI)
MAX_METABOLITES_PER_SIMULATOR = 50 

# Define LLM Models and Costs (Updated for transparency)
# Costs are per 1 Million tokens (Input/Output). 'id' is the actual model identifier for the API.
LLM_MODELS = {
    "OpenAI": {
        "gpt-4.1": {"cost_input": 3.00, "cost_output": 12.00, "id": "gpt-4.1"},
        "gpt-4.1-mini": {"cost_input": 0.80, "cost_output": 3.20, "id": "gpt-4.1-mini"},
        "gpt-4.1-nano": {"cost_input": 0.20, "cost_output": 0.80, "id": "gpt-4.1-nano"},
        "gpt-5": {"cost_input": 1.25, "cost_output": 10.00, "id": "gpt-5"},
        "gpt-5-mini": {"cost_input": 0.25, "cost_output": 2.00, "id": "gpt-5-mini"},
        "gpt-5-nano": {"cost_input": 0.05, "cost_output": 0.40, "id": "gpt-5-nano"},
    },
    "OpenRouter": {
        "OpenAI GPT-4.1": {"cost_input": 3.00, "cost_output": 12.00, "id": "openai/gpt-4.1"},
        "OpenAI GPT-4.1 Mini": {"cost_input": 0.80, "cost_output": 3.20, "id": "openai/gpt-4.1-mini"},
        "OpenAI GPT-4.1 Nano": {"cost_input": 0.20, "cost_output": 0.80, "id": "openai/gpt-4.1-nano"},
        "OpenAI GPT-5": {"cost_input": 1.25, "cost_output": 10.00, "id": "openai/gpt-5"},
        "OpenAI GPT-5 Mini": {"cost_input": 0.25, "cost_output": 2.00, "id": "openai/gpt-5-mini"},
        "OpenAI GPT-5 Nano": {"cost_input": 0.05, "cost_output": 0.40, "id": "openai/gpt-5-nano"},
    },
}

# ... (Keep existing Callbacks for Guided Wizard: _get_llm_models, _ping_qsar, _estimate_cost, _on_run_pipeline) ...
# Callbacks for Guided Wizard
def _get_llm_models():
    return LLM_MODELS

def _ping_qsar(api_url: str):
    # Used by wizard for validation step.
    try:
        api_client = QSARToolboxAPI(base_url=api_url, timeout=(5, 10))
        version_info = api_client.get_version()
        if version_info:
            # Return True and a message (e.g., version info)
            # Ensure version_info is a dict before accessing keys
            if isinstance(version_info, dict):
                 return True, f"QSAR Toolbox v{version_info.get('version', 'Unknown')}"
            return True, "Connection successful (Version info format unexpected)."
        return False, "Connection successful but could not retrieve version."
    except Exception as e:
        return False, str(e)

def _estimate_cost(model_id: str, max_output_tokens: int):
    # Stub remains the same
    return {"model": model_id, "output_tokens": max_output_tokens, "note": "Illustrative estimate"}

# UPDATED: Implemented _on_run_pipeline to use execute_analysis (sync wrapper)
def _on_run_pipeline(config: dict):
    # This function is called synchronously by the wizard upon completion.
    
    # Mapping wizard config to execute_analysis parameters.

    # Chemical identification (Wizard ensures these are present)
    # Use the identifier used for the search initially
    identifier = config.get("identifier")
    search_type = config.get("search_type")

    if not identifier:
            st.error("Chemical identifier is missing from wizard configuration.")
            return

    # Context composition (Wizard combines these fields)
    context = config.get("context")
    if not context:
        context = "General chemical hazard assessment."

    simulator_guids = config.get("simulator_guids", [])

    # Use the configuration stored in the main session state (Wizard Step 1 ensures it's valid)
    llm_config = st.session_state.llm_config.copy() # Make a copy to allow overrides
    qsar_config = st.session_state.qsar_config

    # Apply LLM overrides if present (NEW)
    if config.get("llm_temperature_override") is not None:
        llm_config["temperature_override"] = config["llm_temperature_override"]
    
    if config.get("llm_max_tokens_override") is not None:
        llm_config["max_tokens_override"] = config["llm_max_tokens_override"]
        
    if config.get("reasoning_effort") is not None:
        llm_config["reasoning_effort"] = config["reasoning_effort"]

    # Scope Config mapping (Wizard Step 4 and 5), including new exclusions and RA strategy
    scope_config = {
        "include_properties": config.get("include_properties", True),
        "include_experimental": config.get("include_experimental", True),
        "include_profiling": config.get("include_profiling", True),
        "include_qsar": config.get("include_qsar", True),
        # New Exclusions
        "exclude_adme_tk": config.get("exclude_adme_tk", False),
        "exclude_mammalian_tox": config.get("exclude_mammalian_tox", False),
        # NEW: honor wizard profiler picks
        "selected_profiler_guids": config.get("selected_profiler_guids", []),
        "selected_qsar_model_guids": config.get("selected_qsar_model_guids", []),
        "include_slow_profilers": config.get("include_slow_profilers", False),
        # Read-Across details
        "rax_strategy": config.get("rax_strategy", "Hybrid"),
        "rax_similarity_basis": config.get("rax_similarity_basis", "Combined"),
    }

    # Store inputs in session state for persistence in results view
    st.session_state.input_identifier = identifier
    st.session_state.input_search_type = search_type
    st.session_state.input_context = context
    st.session_state.selected_simulator_guids = simulator_guids

    
    # Run execute_analysis (the synchronous wrapper)
    try:
        execute_analysis(
            identifier=identifier,
            search_type=search_type,
            context=context,
            simulator_guids=simulator_guids,
            llm_config=llm_config,
            qsar_config=qsar_config,
            scope_config=scope_config
        )

        # Upon successful execution start/completion, clear the wizard state.
        if "wiz" in st.session_state:
            del st.session_state["wiz"]

        # Force a rerun. The main() function will now detect analysis_results and display them.
        st.rerun()

    except Exception as e:
        st.error(f"Error running analysis pipeline from wizard: {e}")
        logger.error(f"Error running analysis pipeline from wizard: {e}")


# ... (Keep existing helper functions: initialize_session_state, update_progress, check_connection) ...
def initialize_session_state():
    """Initialize or reset session state variables"""
    defaults = {
        'analysis_results': None, # Raw data from QSAR API
        'final_report': None, # Synthesized report from agents
        'specialist_outputs_dict': None, # To store individual agent outputs
        'comprehensive_log': None, # To store the complete analysis log
        'error': None,
        'connection_status': None,
        'progress_value': 0.0,
        'progress_description': '',
        'retry_count': 0,
        'max_retries': 15,
        'download_clicked': False,
        # Store inputs for persistence
        'input_identifier': '',
        'input_search_type': 'name',
        'input_context': '',
        'input_details': {},
        # UPDATED: Metabolism state
        'available_simulators': [],
        'selected_simulator_guids': [], # Changed from single GUID to list
        # NEW: Profiler catalog + selection
        'available_profilers': [],
        'selected_profiler_guids': [],
        'available_qsar_models': [],
        'recommended_qsar_models': [],
        'recommended_qsar_model_guids': [],
        'selected_qsar_model_guids': [],
        'include_slow_profilers': False,
        'include_qsar_models': True,
        # LLM error tracking
        'last_llm_error': None,
    }

    # --- Configuration State Initialization ---
    # Load defaults from environment variables if available, otherwise set to defaults
    if 'llm_config' not in st.session_state:
        # Default configuration
        default_provider = 'OpenAI'
        default_model_name = 'gpt-4.1-nano'
        
        # Attempt to load API key from environment variables
        # Prioritize OPENAI_API_KEY as a common default
        default_api_key = os.getenv('OPENAI_API_KEY', '')
        
        st.session_state.llm_config = {
            'provider': default_provider,
            'model_name': default_model_name, # Display name
            'api_key': default_api_key,
            'api_base': None, # Used for OpenRouter
            'config_complete': bool(default_api_key),
            # Default parameters (used if not overridden)
            'temperature': 0.15,
            'max_tokens': 10000,
        }

    if 'qsar_config' not in st.session_state:
        # Use a common default URL for recent QSAR Toolbox versions
        default_qsar_url = os.getenv('QSAR_TOOLBOX_API_URL', 'http://localhost:5001/api/v6.0')
        st.session_state.qsar_config = {
            'api_url': default_qsar_url,
            'config_complete': bool(default_qsar_url)
        }
    # --- END Configuration State Initialization ---

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def update_progress(value: float, description: str):
    """Update progress bar with value and description"""
    st.session_state.progress_value = value
    st.session_state.progress_description = description
    # Ensure progress bar exists or create it
    if 'progress_bar' not in st.session_state:
        # Check if the current context allows creating UI elements
        try:
            st.session_state.progress_bar = st.progress(0.0)
        except st.errors.StreamlitAPIException:
            # If we cannot create the progress bar (e.g., running outside main streamlit thread), skip update
            logger.warning("Could not create progress bar outside Streamlit context.")
            return

    # Check if progress bar element still exists before updating
    try:
        st.session_state.progress_bar.progress(value, text=f"Status: {description}")
    except Exception: # Handle cases where the element might have been removed
        try:
            st.session_state.progress_bar = st.progress(value,
                                                        text=f"Status: {description}")
        except st.errors.StreamlitAPIException:
            logger.warning("Could not update progress bar outside Streamlit context.")



def check_connection(api_client: QSARToolboxAPI) -> bool:
    """Check if QSAR Toolbox API is accessible and fetch initial data like simulators."""
    try:
        api_client.get_version()
        # Fetch simulators upon successful connection (NEW)
        simulators = api_client.get_simulators()
        st.session_state.available_simulators = simulators
        # NEW: also cache the profiler and QSAR catalogs
        profilers = api_client.get_profilers()
        st.session_state.available_profilers = profilers
        try:
            qsar_catalog = api_client.get_all_qsar_models_catalog()
        except Exception as exc:
            logger.warning(f"Could not fetch QSAR model catalog during connection check: {exc}")
            qsar_catalog = []
        st.session_state.available_qsar_models = qsar_catalog
        recommended_qsar = derive_recommended_qsar_models(qsar_catalog)
        st.session_state.recommended_qsar_models = recommended_qsar
        recommended_guids = [
            entry.get("Guid") for entry in recommended_qsar if entry.get("Guid")
        ]
        st.session_state.recommended_qsar_model_guids = recommended_guids
        if not st.session_state.get('selected_qsar_model_guids'):
            st.session_state.selected_qsar_model_guids = list(recommended_guids)

        catalog_guids = [entry.get("Guid") for entry in qsar_catalog if isinstance(entry, dict)]
        current_qsar_selection = st.session_state.get('selected_qsar_model_guids', [])
        valid_qsar_selection = [guid for guid in current_qsar_selection if guid in catalog_guids]
        if valid_qsar_selection:
            st.session_state.selected_qsar_model_guids = valid_qsar_selection
        elif recommended_guids:
            st.session_state.selected_qsar_model_guids = list(recommended_guids)
        else:
            st.session_state.selected_qsar_model_guids = []
        
        # Validate existing selection against available simulators
        valid_selections = []
        available_guids = [s['Guid'] for s in simulators]
        
        # Ensure selected_simulator_guids exists
        if 'selected_simulator_guids' not in st.session_state:
            st.session_state.selected_simulator_guids = []

        for guid in st.session_state.selected_simulator_guids:
            if guid in available_guids:
                valid_selections.append(guid)
        
        st.session_state.selected_simulator_guids = valid_selections

        # Validate selected profilers against available profilers
        valid_prof = []
        profiler_guids = [p.get('Guid') for p in profilers if isinstance(p, dict)]
        for guid in st.session_state.get('selected_profiler_guids', []):
            if guid in profiler_guids:
                valid_prof.append(guid)
        st.session_state.selected_profiler_guids = valid_prof

        st.session_state.connection_status = True
        return True
    except QSARConnectionError:
        st.session_state.connection_status = False
        # st.error("⚠️ Unable to connect to QSAR Toolbox API. Please check if the QSAR Toolbox is running and the API URL is correct.")
        return False
    except Exception as e:
        st.session_state.connection_status = False
        # st.error(f"⚠️ Unexpected error checking connection: {str(e)}")
        return False

def _estimate_run_cost(cost_input: float, cost_output: float, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD for one run at given token counts."""
    return (input_tokens / 1_000_000) * cost_input + (output_tokens / 1_000_000) * cost_output


# ... (Keep existing render functions: render_reports_section, render_specialist_downloads) ...
# Modify render_reports_section to display the single final report
def render_reports_section(identifier: str):
    """Render the final synthesized report section."""
    st.header("Synthesized Analysis Report")

    if 'final_report' not in st.session_state:
        st.session_state.final_report = None

    if st.session_state.final_report:
        st.markdown(st.session_state.final_report)
        filename = f"{identifier}_synthesized_report.txt"
        st.download_button(
            label="Download Synthesized Report (TXT)",
            data=st.session_state.final_report,
            file_name=filename,
            mime="text/plain",
            key="synthesized_report_download"
        )
    else:
        st.info("Synthesized report is not available or is being generated.")


# UPDATED FUNCTION: Includes PDF Download
def render_specialist_downloads(identifier: str):
    """Render download buttons for individual specialist agent reports and the comprehensive log."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Downloads")

    # --- NEW: Comprehensive PDF Report Download ---
    if st.session_state.get('comprehensive_log') and st.session_state.get('final_report'):
        st.sidebar.markdown("#### Comprehensive Report")
        
        # We generate the PDF on the fly when the download button is utilized.
        # The download button's 'data' argument triggers the generation.
        try:
            # Generate the PDF content (this might take a moment)
            # We rely on the comprehensive_log containing all necessary data (metadata, reports)
            pdf_bytes = generate_pdf_report(st.session_state.comprehensive_log)
            
            st.sidebar.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"{identifier}_OQT_Report.pdf",
                mime="application/pdf",
                key="comprehensive_pdf_download",
                help="Download a comprehensive PDF with identity, data provenance, key studies, and agent narratives."
            )
        except Exception as e:
            st.sidebar.error(f"Error generating PDF: {e}")
            # Optionally log the full traceback
            import traceback
            logger.error(f"Error generating PDF: {e}\n{traceback.format_exc()}")


    # --- Existing Downloads (Log and Specialist TXT) ---

    # Add Comprehensive Log Download (Addressing Transparency)
    if st.session_state.get('comprehensive_log'):
        st.sidebar.markdown("#### Raw Data and Logs")
        log_data = json.dumps(st.session_state.comprehensive_log, indent=2, default=str)
        st.sidebar.download_button(
            label="Download Comprehensive Log (JSON)",
            data=log_data,
            file_name=f"{identifier}_comprehensive_log.json",
            mime="application/json",
            key="comprehensive_log_download",
            help="Download a complete record of the analysis, including configuration, inputs, raw data, and all agent outputs."
        )

    st.sidebar.markdown("#### Specialist Analyses (TXT)")
    if 'specialist_outputs_dict' not in st.session_state or not st.session_state.specialist_outputs_dict:
        st.sidebar.info("Specialist analyses not available.")
    else:
        # Updated list of specialists
        specialist_names = [
            "Chemical_Context",
            "Physical_Properties",
            "Environmental_Fate",
            "Profiling_Reactivity",
            "Experimental_Data",
            "Metabolism",
            "QSAR_Predictions",
            "Read_Across"
        ]

        # Use the names directly as keys
        for name in specialist_names:
            report_content = st.session_state.specialist_outputs_dict.get(name)
            if report_content:
                filename = f"{identifier}_specialist_{name}.txt"
                st.sidebar.download_button(
                    label=f"{name.replace('_', ' ')}",
                    data=report_content,
                    file_name=filename,
                    mime="text/plain",
                    key=f"specialist_{name}_download"
                )


# UPDATED: Increased version number for UI fixes
def generate_comprehensive_log(
    inputs: Dict[str, Any],
    llm_config: Dict[str, Any], qsar_config: Dict[str, Any],
    processed_qsar_data: Dict[str, Any], specialist_analyses: Dict[str, str],
    synthesized_report: str
) -> Dict[str, Any]:
    """Generates a comprehensive JSON log of the entire analysis run for transparency."""

    # Mask the API key before logging
    masked_llm_config = llm_config.copy()
    if 'api_key' in masked_llm_config and masked_llm_config['api_key']:
        key = masked_llm_config['api_key']
        masked_llm_config['api_key'] = f"{key[:4]}***MASKED***{key[-4:]}"

    log = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tool_name": "O-QT Assistant",
            "version": "1.4.3" # Updated version for UI fixes
        },
        "configuration": {
            "llm_configuration": masked_llm_config,
            "qsar_toolbox_configuration": qsar_config
        },
        "inputs": inputs,
        "data_retrieval": {
            # This now contains the processed/filtered data used by the agents
            "processed_qsar_toolbox_data": processed_qsar_data 
        },
        "analysis": {
            "specialist_agent_outputs": specialist_analyses,
            "synthesized_report": synthesized_report
        }
    }
    # Attach QPRF metadata if available
    qprf_meta = processed_qsar_data.get("qprf_metadata", {}) if isinstance(processed_qsar_data, dict) else {}
    if qprf_meta:
        log["metadata"]["qprf_software"] = qprf_meta.get("software", {})
        log["data_retrieval"]["qprf_metadata"] = qprf_meta

    return log


def perform_chemical_analysis(identifier: str, search_type: str, context: str, simulator_guids: List[str], qsar_config: Dict[str, Any], scope_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Perform chemical data retrieval using QSAR Toolbox API (Synchronous)."""
    
    if scope_config is None:
        # Default scope (fetch everything, compatible with Standard mode)
        scope_config = {
            "include_properties": True,
            "include_experimental": True,
            "include_profiling": True,
            "selected_profiler_guids": [],
            "selected_qsar_model_guids": [],
            "include_slow_profilers": False,
            "include_qsar": True,
        }
    else:
        scope_config.setdefault("include_qsar", True)
        scope_config.setdefault("selected_profiler_guids", [])
        scope_config.setdefault("selected_qsar_model_guids", [])
        scope_config.setdefault("include_slow_profilers", False)
        
    try:
        # Initialize API client using provided configuration
        api_url = qsar_config.get('api_url')

        if not api_url:
            raise ValueError("QSAR Toolbox API URL is not configured.")

        api_client = QSARToolboxAPI(
            base_url=api_url,
            timeout=(10, 120),
            max_retries=15  # Use higher retry count for better reliability
        )

        # Use the check_connection utility which also updates session state connection status
        if not check_connection(api_client):
            # Error message handled by check_connection or UI status display
            return None

        # --- Sequential Data Fetching ---
        update_progress(0.05, "🔍 Searching for chemical...")
        try:
            if search_type == 'name':
                # Check if caching is used and clear it if necessary
                if hasattr(api_client.search_by_name, 'cache_clear'):
                    api_client.search_by_name.cache_clear()
                search_result = api_client.search_by_name(
                    identifier,
                    search_option=SearchOptions.EXACT_MATCH   # <- "0"
                )
            elif search_type == 'cas':
                if hasattr(api_client.search_by_cas, 'cache_clear'):
                    api_client.search_by_cas.cache_clear()
                search_result = api_client.search_by_cas(identifier)
            elif search_type == 'smiles': # Explicitly handle smiles
                search_result = api_client.search_by_smiles(identifier)
            else:
                # Fallback: treat unrecognised search types as exact name lookups
                if hasattr(api_client.search_by_name, 'cache_clear'):
                    api_client.search_by_name.cache_clear()
                search_result = api_client.search_by_name(
                    identifier,
                    search_option=SearchOptions.EXACT_MATCH
                )

            if not search_result:
                raise QSARResponseError(f"Chemical not found: {identifier} (using search type: {search_type})")
        except QSARTimeoutError:
             st.warning("Search request timed out, retrying...")
             if st.session_state.retry_count < st.session_state.max_retries:
                 st.session_state.retry_count += 1
                 # Pass all arguments in the recursive call
                 return perform_chemical_analysis(identifier, search_type, context, simulator_guids, qsar_config, scope_config)
             else:
                 raise QSARTimeoutError("Maximum retries exceeded during chemical search")

        # --- Improved selection: pick best usable hit and pre-fetch calculators ---
        hits = search_result if isinstance(search_result, list) else [search_result]
        try:
            chemical_data, precomputed_properties, chem_id, selection_notes = select_hit_with_properties(
                api_client, identifier, hits, logger=logger
            )
        except RuntimeError as exc:
            st.error(f"Failed to select a usable Toolbox entry for '{identifier}': {exc}")
            return None

        if selection_notes:
            logger.debug("Hit selection notes for %s: %s", identifier, " | ".join(selection_notes))

        # --- UPDATED: Metabolism Simulation (Multi-Simulator) ---
        
        # Initialize structure for metabolism data
        metabolism_data = {
            "status": "Pending",
            "note": "",
            "simulations": {} # Dictionary to hold results per simulator
        }

        if not simulator_guids:
            metabolism_data["status"] = "Skipped"
            metabolism_data["note"] = "No metabolism simulators were selected."
        else:
            total_simulators = len(simulator_guids)
            simulators_completed = 0
            
            # Get simulator names for better logging
            # Ensure available_simulators is populated in session state
            if 'available_simulators' not in st.session_state:
                 st.session_state.available_simulators = []
            simulator_map = {s['Guid']: s['Caption'] for s in st.session_state.available_simulators}

            for i, simulator_guid in enumerate(simulator_guids):
                simulator_name = simulator_map.get(simulator_guid, f"GUID: {simulator_guid}")
                progress_start = 0.1
                progress_end = 0.4
                progress_step = (progress_end - progress_start) / total_simulators
                current_progress = progress_start + (i * progress_step)

                update_progress(current_progress, f"🧬 Simulating metabolism ({i+1}/{total_simulators}): {simulator_name} (may take time)...")
                
                simulation_result = {
                    "status": "Pending",
                    "note": "",
                    "simulator_guid": simulator_guid,
                    "simulator_name": simulator_name,
                    "metabolites": []
                }

                try:
                    # This might take time, the timeout is handled within the API client method
                    raw_metabolites = api_client.apply_simulator(simulator_guid, chem_id)
                    
                    # Process results
                    if raw_metabolites:
                        # Truncate metabolites if they exceed the limit PER SIMULATOR
                        if len(raw_metabolites) > MAX_METABOLITES_PER_SIMULATOR:
                            metabolites = raw_metabolites[:MAX_METABOLITES_PER_SIMULATOR]
                            truncation_note = f" (Truncated to first {MAX_METABOLITES_PER_SIMULATOR} metabolites for analysis)"
                        else:
                            metabolites = raw_metabolites
                            truncation_note = ""

                        simulation_result["status"] = "Success"
                        simulation_result["note"] = f"Generated {len(raw_metabolites)} metabolites{truncation_note}."
                        simulation_result["metabolites"] = metabolites
                        simulators_completed += 1
                    else:
                        simulation_result["status"] = "Success"
                        simulation_result["note"] = "Simulation completed but generated no metabolites."
                        simulators_completed += 1

                except QSARResponseError as e:
                    # Handle failure/timeout gracefully for this specific simulator
                    st.warning(f"Metabolism simulation failed for {simulator_name}: {str(e)}. Proceeding to next step.")
                    simulation_result["status"] = "Failed"
                    simulation_result["note"] = f"Simulation failed or timed out: {str(e)}"
                except Exception as e:
                    st.warning(f"Unexpected error during metabolism simulation for {simulator_name}: {str(e)}. Proceeding to next step.")
                    simulation_result["status"] = "Error"
                    simulation_result["note"] = f"An unexpected error occurred: {str(e)}"
                
                # Store the result for this simulator
                metabolism_data["simulations"][simulator_guid] = simulation_result

            # Summarize overall metabolism status
            if simulators_completed == total_simulators:
                metabolism_data["status"] = "Success"
                metabolism_data["note"] = f"All {total_simulators} selected simulators completed successfully."
            elif simulators_completed > 0:
                metabolism_data["status"] = "Partial Success"
                metabolism_data["note"] = f"{simulators_completed} out of {total_simulators} simulators completed successfully."
            else:
                metabolism_data["status"] = "Failed"
                metabolism_data["note"] = "All selected metabolism simulations failed."


        # --- Properties Calculation (Conditional based on scope_config) ---
        if scope_config.get("include_properties"):
            update_progress(0.4, "📊 Calculating chemical properties...")
            properties = precomputed_properties or {}
            if not properties:
                st.warning("No calculable properties were returned for the selected Toolbox entry.")
        else:
            update_progress(0.4, "📊 Skipping chemical properties (per configuration)...")
            properties = {}

        # --- QPRF/RAAF Metadata Enrichment ---
        update_progress(0.45, "📋 Enriching with QPRF/RAAF metadata...")
        software_info: Dict[str, Any] = {}
        try:
            enricher = QPRFEnricher(api_client)
            chemical_data = enricher.enrich_substance_identity(chemical_data)
            if properties:
                properties = enricher.enrich_calculator_results(properties)
            software_info = enricher.get_software_info()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("QPRF enrichment failed (non-critical): %s", exc)
            software_info = {}

        st.session_state.qprf_software_info = software_info


        
        # --- Experimental Data Retrieval (Conditional based on scope_config) ---
        experimental_data = []
        if scope_config.get("include_experimental"):
            update_progress(0.6, "🧪 Retrieving experimental data (with metadata)...") # Adjusted progress
            try:
                # Data retrieval now attempts to include metadata by default
                experimental_data = api_client.get_all_chemical_data(chem_id) or []
            except Exception as e:
                st.warning(f"Error retrieving experimental data: {str(e)}")
                experimental_data = []
        else:
            update_progress(0.6, "🧪 Skipping experimental data (per configuration)...")


        # --- Profiling Data Retrieval (Conditional based on scope_config) ---
        profiling_data = {}
        if scope_config.get("include_profiling"):
            update_progress(0.75, "🔬 Retrieving profiling data...") # Adjusted progress
            try:
                # We profile the parent compound; allow user-selected profilers when provided
                selected_prof_guids = []
                if isinstance(scope_config, dict):
                    selected_prof_guids = scope_config.get("selected_profiler_guids", []) or []
                include_slow = bool(scope_config.get("include_slow_profilers"))
                profiling_data = api_client.get_chemical_profiling(
                    chem_id,
                    # keep default simulator (No metabolism)
                    selected_profiler_guids=tuple(selected_prof_guids) if selected_prof_guids else None,
                    include_slow_profilers=include_slow
                ) or {}
            except Exception as e:
                st.warning(f"Error retrieving profiling data: {str(e)}")
                profiling_data = {'status': 'Error', 'note': f'Error retrieving profiling data: {str(e)}'}
        else:
            update_progress(0.75, "🔬 Skipping profiling data (per configuration)...")


        raw_qsar = {
            "catalog_size": 0,
            "executed_models": 0,
            "predictions": [],
            "summary": {"total": 0},
            "selected_model_guids": []
        }
        qsar_processed = process_qsar_predictions([])
        selected_qsar_guids = scope_config.get("selected_qsar_model_guids", []) or []
        if scope_config.get("include_qsar", True) and selected_qsar_guids:
            update_progress(0.82, "🔮 Running selected QSAR models...")
            try:
                try:
                    api_client.session.get(f"{qsar_config.get('api_url')}/session/open", timeout=(5, 15))
                except Exception as exc:
                    logger.debug(f"Session warm-up encountered an issue (continuing): {exc}")
                raw_qsar = run_qsar_predictions(
                    api_client,
                    chem_id,
                    selected_model_guids=selected_qsar_guids,
                    logger=logger
                )
                qsar_processed = process_qsar_predictions(raw_qsar.get("predictions", []))
            except Exception as e:
                st.warning(f"QSAR model execution failed: {str(e)}")
                raw_qsar = {
                    "catalog_size": 0,
                    "executed_models": 0,
                    "predictions": [],
                    "summary": {"total": 0},
                    "selected_model_guids": selected_qsar_guids
                }
                qsar_processed = process_qsar_predictions([])
            update_progress(0.85, "✅ QSAR data retrieval complete!")
            raw_qsar.setdefault("selected_model_guids", selected_qsar_guids)
        elif scope_config.get("include_qsar", True):
            update_progress(0.82, "🔮 QSAR predictions skipped (no models selected).")
            raw_qsar["selected_model_guids"] = []
        else:
            update_progress(0.82, "🔮 QSAR predictions skipped (disabled).")
            raw_qsar["selected_model_guids"] = []

        # Format results dictionary (without old report generation)
        results = {
            'chemical_data': {
                'basic_info': chemical_data,
                'properties': properties
            },
            'experimental_data': experimental_data,
            'profiling': profiling_data,
            'metabolism': metabolism_data,
            'qsar_models': {
                'raw': raw_qsar,
                'processed': qsar_processed,
            },
            'context': context,
            'qprf_metadata': {
                'software': software_info,
                'enrichment_applied': bool(software_info)
            }
        }
        return results

    except Exception as e:
        st.session_state.error = str(e)
        # Ensure progress bar is removed or reset on error
        if 'progress_bar' in st.session_state:
            try:
                st.session_state.progress_bar.empty()
            except Exception:
                pass # Ignore if already gone
            if 'progress_bar' in st.session_state:
               # Attempt to delete safely
               try:
                   del st.session_state.progress_bar
               except KeyError:
                   pass
        raise # Re-raise the exception to be caught in execute_analysis


# ... (Keep existing functions: render_configuration_ui, render_methodology_and_transparency) ...
# render_configuration_ui (remains largely the same, adding default parameter display)
def render_configuration_ui():
    """Renders the configuration UI in the sidebar for LLM and QSAR API settings."""
    st.sidebar.header("Configuration")

    # --- QSAR Toolbox Configuration ---
    st.sidebar.subheader("QSAR Toolbox API")
    current_qsar_url = st.session_state.qsar_config['api_url']

    qsar_url = st.sidebar.text_input(
        "API URL",
        value=current_qsar_url,
        help="Enter the URL of the running QSAR Toolbox API (e.g., http://localhost:5001/api/v6.0)"
    )

    if qsar_url != current_qsar_url:
        st.session_state.qsar_config['api_url'] = qsar_url
        st.session_state.connection_status = None # Reset connection status on URL change

    st.session_state.qsar_config['config_complete'] = bool(qsar_url)

    # Add a button to manually check connection
    if st.sidebar.button("Check Connection"):
        if qsar_url:
            try:
                api_client = QSARToolboxAPI(base_url=qsar_url, timeout=(5, 10))
                check_connection(api_client)
                # Force rerun to update status display immediately
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Connection attempt failed: {e}")
        else:
            st.sidebar.warning("Please enter a QSAR API URL.")


    # --- LLM Configuration ---
    st.sidebar.subheader("LLM Configuration")

    current_provider = st.session_state.llm_config['provider']
    provider = st.sidebar.selectbox(
        "Provider",
        options=list(LLM_MODELS.keys()),
        index=list(LLM_MODELS.keys()).index(current_provider)
    )

    if provider != current_provider:
        st.session_state.llm_config['provider'] = provider
        # Update API key based on provider change if environment variables exist (convenience)
        if provider == 'OpenAI':
            st.session_state.llm_config['api_key'] = os.getenv('OPENAI_API_KEY', '')
        elif provider == 'OpenRouter':
            st.session_state.llm_config['api_key'] = os.getenv('OPENROUTER_API_KEY', '')
        # Reset model selection to the first one for the new provider
        st.session_state.llm_config['model_name'] = list(LLM_MODELS[provider].keys())[0]

    model_options = list(LLM_MODELS[provider].keys())
    try:
        current_model_index = model_options.index(st.session_state.llm_config['model_name'])
    except ValueError:
        current_model_index = 0 # Handle case where previous model is not in the new provider's list

    model_name = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=current_model_index
    )
    st.session_state.llm_config['model_name'] = model_name

    # Display Cost Information (Transparency)
    model_info = LLM_MODELS[provider].get(model_name, {})
    cost_input = model_info.get("cost_input")
    cost_output = model_info.get("cost_output")

    if cost_input is not None and cost_output is not None:
        if cost_input == 0 and cost_output == 0:
            st.sidebar.info("Cost: Free (via OpenRouter)")
        else:
            st.sidebar.info(
                f"**Price (per 1M tokens)**  \n"
                f"• Input: **${cost_input:.2f}**  \n"
                f"• Output: **${cost_output:.2f}**"
            )
            with st.sidebar.expander("Per‑run estimate (at your defaults)"):
                # Let user set an assumption for prompt size; output defaults to max_tokens
                assumed_input = st.number_input(
                    "Assumed input tokens per run",
                    min_value=0, max_value=300_000, step=500, value=2_000,
                    help="Used only for this estimate. Change based on your typical prompt size."
                )
                output_tokens = int(st.session_state.llm_config.get('max_tokens', 10000) or 10000)
                est = _estimate_run_cost(cost_input, cost_output, assumed_input, output_tokens)
                output_only = _estimate_run_cost(0.0, cost_output, 0, output_tokens)
                st.markdown(
                    f"**Estimated cost / run:** **${est:,.4f}**  \n"
                    f"(input {assumed_input:,} + output {output_tokens:,})"
                )
                st.caption(f"Output‑only (max): ${output_only:,.4f}")
    else:
        st.sidebar.warning("Cost information unavailable.")


    api_key = st.sidebar.text_input(
        f"API Key",
        value=st.session_state.llm_config['api_key'],
        type="password",
        help=f"Enter your API key for {provider}."
    )

    st.session_state.llm_config['api_key'] = api_key

    # Set API Base URL for OpenRouter
    if provider == "OpenRouter":
        st.session_state.llm_config['api_base'] = "https://openrouter.ai/api/v1"
    else:
        st.session_state.llm_config['api_base'] = None

    # Display Default Parameters (with GPT-5 awareness)
    st.sidebar.markdown("---")
    
    # Get selected model info for GPT-5 detection
    selected_model_id = st.session_state.llm_config.get("model_id")
    if not selected_model_id and model_name and provider:
        try:
            selected_model_id = LLM_MODELS[provider][model_name]['id']
        except (KeyError, TypeError):
            selected_model_id = ""
    
    is_gpt5 = "gpt-5" in (selected_model_id or "").lower()
    
    if is_gpt5:
        st.sidebar.markdown("**Default LLM Parameters (GPT‑5):**")
        st.sidebar.info("GPT‑5 uses reasoning mode: temperature is fixed by the provider.")
        
        # Show max completion tokens, not max tokens
        current_max = int(st.session_state.llm_config.get("max_tokens_override") or
                         st.session_state.llm_config.get("max_tokens", 10000))
        new_max = st.sidebar.number_input(
            "Max completion tokens",
            min_value=256,
            max_value=64000,
            value=current_max,
            step=256,
            help="Upper bound on generated tokens (reasoning + final output).",
            key="gpt5_max_tokens_sidebar"
        )
        st.session_state.llm_config["max_tokens_override"] = int(new_max)
        st.sidebar.markdown(f"Max Completion Tokens: {new_max}")
        st.sidebar.caption("Note: GPT‑5 'reasoning' tokens are charged as output tokens.")
    else:
        st.sidebar.markdown("**Default LLM Parameters:**")
        st.sidebar.markdown(f"Temperature: {st.session_state.llm_config['temperature']}")
        st.sidebar.markdown(f"Max Tokens: {st.session_state.llm_config['max_tokens']}")
        st.sidebar.info("These parameters can be overridden during a Guided Wizard run.")

    # Check configuration completeness
    st.session_state.llm_config['config_complete'] = bool(api_key and model_name)

    # Display last LLM error if any
    if st.session_state.get("last_llm_error"):
        st.sidebar.markdown("---")
        st.sidebar.warning(f"⚠️ Last LLM error: {st.session_state['last_llm_error']}")
        if st.sidebar.button("Clear Error", key="clear_llm_error"):
            st.session_state["last_llm_error"] = None
            st.rerun()

    # --- Status Display ---
    st.sidebar.subheader("Status")
    # QSAR Status
    if st.session_state.connection_status is True:
        st.sidebar.success("✅ Connected to QSAR Toolbox")
    elif st.session_state.connection_status is False:
        st.sidebar.error("❌ Failed to connect to QSAR Toolbox.")
    else:
         st.sidebar.info("QSAR connection status pending.")

    # LLM Status
    if st.session_state.llm_config['config_complete']:
        st.sidebar.success(f"✅ LLM Configured ({provider})")
    else:
        st.sidebar.warning("LLM API Key required.")


def render_methodology_and_transparency():
    """Renders information about the tool's methodology, scope, and transparency in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("About O-QT Assistant")

    with st.sidebar.expander("Licensing and Costs"):
        st.markdown("""
        **O-QT Assistant is open source and free to use.**

        The project is released under the Apache 2.0 license and does not bundle access to the OECD QSAR Toolbox or any LLM provider. You supply your own credentials.

        **Usage costs:** Running analyses consumes API credits from your chosen LLM provider (OpenAI or OpenRouter). Enter your key in the configuration area to see current price guidance.
        """)

    with st.sidebar.expander("Methodology and Scope"):
        st.markdown("""
        **Workflow:** O-QT connects to the OECD QSAR Toolbox WebAPI, retrieves chemical information, profilers, metabolism simulations, and QSAR model outputs, and then coordinates specialist LLM agents plus a synthesiser to build the report.

        **Scope:** The community release focuses on core hazard assessment endpoints. Optional toggles allow you to skip ADME/TK and mammalian toxicity data when you need a faster pass.

        **Data Handling:** Agents surface measured values, metadata, and provenance exactly as returned by the Toolbox. Provenance tags in the report indicate whether information is an experimental record or a QSAR estimate.

        **Applicability:** O-QT works best for industrial chemicals, environmental pollutants, and cosmetics that fall within the applicability domains of the selected profilers and models.
        """)

    with st.sidebar.expander("Transparency and Reliability"):
        st.markdown("""
        **LLM Reliability:** Prompts instruct agents to operate strictly on the retrieved Toolbox data. The combined PDF and JSON logs allow you to audit every value that appears in the narrative.

        **Profiler and Simulator Selection:** Standard mode uses a balanced default set. Both Standard mode and Guided mode let you customise profilers, QSAR models, and metabolism simulators before running an analysis.

        **Comprehensive Log:** Every run produces a JSON bundle capturing configuration, raw Toolbox payloads, agent outputs, and the final synthesis so that results remain reproducible.
        """)


# ... (Keep existing functions: execute_analysis_async, execute_analysis, render_standard_mode, main) ...
# NEW: Centralized analysis execution function (Async Core Logic)
async def execute_analysis_async(identifier: str, search_type: str, context: str, simulator_guids: List[str], llm_config: Dict[str, Any], qsar_config: Dict[str, Any], scope_config: Dict[str, Any] = None):
    """
    Executes the chemical analysis pipeline asynchronously: data retrieval, agent analysis, and report synthesis.
    Stores results in st.session_state.
    """
    # Create inputs dict for logging purposes
    inputs = {
        "identifier": identifier,
        "search_type": search_type,
        "context": context,
        "simulator_guids": simulator_guids,
        "scope_config": scope_config if scope_config else "Default (All)",
        "details": {}
    }

    # Reset state for new analysis
    st.session_state.analysis_results = None
    st.session_state.final_report = None
    st.session_state.specialist_outputs_dict = None
    st.session_state.comprehensive_log = None
    st.session_state.error = None
    st.session_state.retry_count = 0
    if 'exp_data_page' in st.session_state:
        st.session_state.exp_data_page = 1

    # Create progress bar placeholder here
    # Use update_progress to handle creation/reset safely
    update_progress(0.0, "Status: Starting analysis...")


    # Prepare LLM configuration
    current_llm_config = llm_config.copy()
    
    # Map display model name to model ID if needed (handles both standard and wizard configs)
    if 'model_id' not in current_llm_config or not current_llm_config.get('model_id'):
        model_display_name = current_llm_config.get('model_name')
        provider = current_llm_config.get('provider')

        if not provider or not model_display_name:
            st.error("LLM configuration is incomplete (missing provider or model name).")
            return
        try:
            model_id = LLM_MODELS[provider][model_display_name]['id']
            current_llm_config['model_id'] = model_id
        except KeyError:
            st.error(f"Error: Could not find Model ID for {model_display_name} under {provider}.")
            return

    # Apply LLM overrides if present (from Guided Wizard, passed via llm_config) (NEW)
    if current_llm_config.get("temperature_override") is not None:
        # Note: The actual override happens in initialize_llm in llm_utils.py.
        # We just log that an override was requested here.
        inputs["llm_temperature_override"] = current_llm_config["temperature_override"]
    
    if current_llm_config.get("max_tokens_override") is not None:
        inputs["llm_max_tokens_override"] = current_llm_config["max_tokens_override"]


    try:
        # --- Step 1: Sequential Data Fetching ---
        # perform_chemical_analysis is sync and handles hit selection + property retrieval.
        results = perform_chemical_analysis(identifier, search_type, context, simulator_guids, qsar_config, scope_config)

        if results:
            # Note: At this point, st.session_state.analysis_results contains the RAW data before processing/filtering.
            # We no longer store a copy of raw results here, as processing happens progressively.

            # --- Step 1.5: Data Processing and Filtering (NEW) ---
            update_progress(0.81, "⚙️ Processing and filtering data...")

            # Process metadata (Properties and Basic Info are already processed in perform_chemical_analysis)
            processed_experimental_data = process_experimental_metadata(results.get('experimental_data', []))

            # Enrich experimental records with provenance metadata
            try:
                api_for_provenance = QSARToolboxAPI(base_url=qsar_config.get('api_url'))
                collector = KeyStudyCollector(api_for_provenance)
                processed_experimental_data = collector.enrich_experimental_records(processed_experimental_data)
                results['experimental_data'] = processed_experimental_data
                logger.info("Enriched experimental records with key study provenance.")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to enrich experimental data with key study info: %s", exc)
            
            # Apply filters based on scope_config
            if scope_config:
                exclude_adme_tk = scope_config.get("exclude_adme_tk", False)
                exclude_mammalian_tox = scope_config.get("exclude_mammalian_tox", False)
                
                if exclude_adme_tk or exclude_mammalian_tox:
                    filtered_data, drop_counts = filter_experimental_records(
                        processed_experimental_data,
                        exclude_adme_tk=exclude_adme_tk,
                        exclude_mammalian_tox=exclude_mammalian_tox
                    )
                    
                    # Update the 'results' dictionary (used by agents) with filtered data
                    results['experimental_data'] = filtered_data
                    
                    # Add filtering information to the log inputs
                    inputs['filtering_info'] = {
                        "note": f"Applied filters. Dropped Mammalian Tox: {drop_counts['dropped_mammalian']}, Dropped ADME/TK: {drop_counts['dropped_adme_tk']}",
                        "drop_counts": drop_counts
                    }
                    logger.info(f"Applied data filters: {drop_counts}")
                else:
                    results['experimental_data'] = processed_experimental_data
            else:
                results['experimental_data'] = processed_experimental_data

            # Note: The 'results' object now contains processed and potentially filtered experimental data for the agents.
            # The UI display (render_results_section) should use this processed/filtered data as well for consistency.
            # We update the session state results to reflect the processed data used in analysis.
            st.session_state.analysis_results = results


            # --- Step 2: Chemical Context Agent ---
            update_progress(0.82, "🆔 Confirming Chemical Identity...")
            original_context = context if context else "General chemical hazard assessment"
            chemical_data_for_context = results.get('chemical_data', {})
            
            # Use await directly as we are in an async function
            confirmed_identity_str = await analyze_chemical_context(chemical_data_for_context, original_context, current_llm_config)
            
            # Prepend identity to context for other agents
            analysis_context = f"{confirmed_identity_str}\n\nUser Goal: {original_context}"
            
            # Add scope/filter details to context if available (for transparency in LLM analysis)
            if scope_config:
                analysis_context += f"\n\nAnalysis Scope Configuration:\n{json.dumps(scope_config, indent=2)}"
            if 'filtering_info' in inputs:
                 analysis_context += f"\n\nData Filtering Note:\n{inputs['filtering_info']['note']}"

            st.session_state.specialist_outputs_dict = {"Chemical_Context": confirmed_identity_str} # Store context output

            # --- Step 3: Parallel Specialist Agent Analysis ---
            update_progress(0.85, "🧠 Running core specialist agents...")

            # Prepare data slices for agents
            properties_data = results.get('chemical_data', {}).get('properties', {})
            profiling_data = results.get('profiling', {})
            metabolism_data = results.get('metabolism', {})
            qsar_processed = results.get('qsar_models', {}).get('processed', {})

            # **Handle experimental data truncation for LLM agents**
            
            # Use the processed/filtered experimental data (already processed in Step 1.5)
            # NEW: sort newest-first (then by endpoint) so any truncation keeps the most recent studies
            experimental_data_list_unsorted = results.get('experimental_data', [])
            experimental_data_list = sorted(
                experimental_data_list_unsorted,
                key=lambda r: ((r or {}).get('Publication_Year') or -1, str((r or {}).get('Endpoint') or '')),
                reverse=True
            )

            # Apply truncation if needed for LLM processing
            truncation_active_for_llm = len(experimental_data_list) > MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA
            if truncation_active_for_llm:
                experimental_data_for_llm_processing = experimental_data_list[:MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA]
                experimental_data_for_llm_processing.append({
                    "note": f"Dataset truncated: showing first {MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA} of {len(experimental_data_list)} total experimental records."
                })
            else:
                experimental_data_for_llm_processing = experimental_data_list

            # Wrap experimental data list in a dict for consistent agent input type
            experimental_data_dict_for_llm = {"experimental_results": experimental_data_for_llm_processing}
            if truncation_active_for_llm:
                experimental_data_dict_for_llm["note_to_agent"] = \
                    f"The 'experimental_results' list provided has been truncated to the first {MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA} records (plus a system note about this truncation) to manage data volume. Please base your analysis on this subset."


            # Create tasks for specialist agents
            task_phys = asyncio.create_task(analyze_physical_properties(properties_data, analysis_context, current_llm_config))
            task_env  = asyncio.create_task(analyze_environmental_fate(properties_data, analysis_context, current_llm_config))
            task_prof = asyncio.create_task(analyze_profiling_reactivity(profiling_data, analysis_context, current_llm_config))
            task_exp  = asyncio.create_task(analyze_experimental_data(experimental_data_dict_for_llm, analysis_context, current_llm_config))
            task_meta = asyncio.create_task(analyze_metabolism(metabolism_data, analysis_context, current_llm_config))
            task_qsar = asyncio.create_task(analyze_qsar_predictions(qsar_processed, analysis_context, current_llm_config))

            # ✅ Tolerate errors so the pipeline can continue
            results_gather = await asyncio.gather(
                task_phys, task_env, task_prof, task_exp, task_meta, task_qsar,
                return_exceptions=True
            )

            labels = [
                "Physical_Properties",
                "Environmental_Fate",
                "Profiling_Reactivity",
                "Experimental_Data",
                "Metabolism",
                "QSAR_Predictions",
            ]
            core_specialist_outputs_list = []
            for lbl, res in zip(labels, results_gather):
                if isinstance(res, Exception):
                    logger.error(f"{lbl} agent failed: {res}", exc_info=True)
                    fallback = f"[{lbl} agent failed: {res}]"
                    core_specialist_outputs_list.append(fallback)
                    st.session_state.specialist_outputs_dict[lbl] = fallback
                else:
                    txt = str(res)
                    core_specialist_outputs_list.append(txt)
                    st.session_state.specialist_outputs_dict[lbl] = txt

            # --- Step 4: Read Across Agent ---
            update_progress(0.90, "🧬 Analyzing Read-Across Potential...")

            # Prepare results for read_across agent
            results_for_read_across_llm = results.copy()
            # Use the truncated list for the LLM input
            results_for_read_across_llm['experimental_data'] = experimental_data_for_llm_processing
            
            # Avoid placing None under this key (prevents NoneType.get crash in agent)
            results_for_read_across_llm['scope_config'] = scope_config or {}

            # Use await directly
            read_across_report = await analyze_read_across(
                results_for_read_across_llm,
                core_specialist_outputs_list,
                analysis_context,
                current_llm_config
            )
            st.session_state.specialist_outputs_dict["Read_Across"] = read_across_report


            # --- Step 5: Synthesize Final Report ---
            update_progress(0.95, "✍️ Synthesizing final report...")
            
            # Clear any previous LLM errors
            st.session_state["last_llm_error"] = None
            
            try:
                # Use await directly
                final_report_content = await synthesize_report(
                    chemical_identifier=identifier,
                    specialist_outputs=core_specialist_outputs_list,
                    read_across_report=read_across_report,
                    context=original_context,
                    llm_config=current_llm_config
                )
            except Exception as e:
                logger.error(f"Synthesis failed: {e}", exc_info=True)
                # Store error for UI display
                st.session_state["last_llm_error"] = str(e)
                # Fallback: still give the user a consolidated report so UI never shows "not available"
                final_report_content = (
                    "## ⚠️ Report Synthesis Fallback\n"
                    f"**Reason:** {e}\n\n"
                    "Below is a concatenation of specialist outputs and read‑across so you can proceed:\n\n"
                    "### Specialist Outputs\n" + "\n\n---\n\n".join(core_specialist_outputs_list) + "\n\n"
                    "### Read‑Across\n" + (read_across_report or "[No read‑across content]")
                )
            
            st.session_state.final_report = final_report_content

            # --- Step 6: Generate Comprehensive Log ---
            st.session_state.comprehensive_log = generate_comprehensive_log(
                inputs,
                current_llm_config, qsar_config,
                results, st.session_state.specialist_outputs_dict, final_report_content
            )

            update_progress(1.0, "✅ Analysis complete!")

        else:
            # Handle case where perform_chemical_analysis returned None
            st.error("Analysis could not start due to data retrieval issues (e.g., connection error or chemical not found). Check configuration and inputs.")
            if 'progress_bar' in st.session_state:
                try:
                        st.session_state.progress_bar.empty()
                except Exception:
                        pass
                if 'progress_bar' in st.session_state:
                    try:
                        del st.session_state.progress_bar
                    except KeyError:
                        pass


    except (QSARConnectionError, QSARTimeoutError, QSARResponseError) as qsar_err:
        st.error(f"🚫 QSAR API Error: {str(qsar_err)}")
        if 'progress_bar' in st.session_state:
            try:
                st.session_state.progress_bar.empty()
            except Exception:
                pass
            if 'progress_bar' in st.session_state:
                try:
                    del st.session_state.progress_bar
                except KeyError:
                    pass
    except Exception as e:
        st.error(f"❌ Analysis failed unexpectedly: {str(e)}")
        logger.error(f"Analysis failed unexpectedly: {e}")
        # Check if the error relates to LLM configuration (e.g., invalid API key)
        if "AuthenticationError" in str(e) or "API key" in str(e) or "invalid_api_key" in str(e):
                st.error("⚠️ LLM Authentication failed. Please check your API key and provider configuration in the sidebar.")
        if st.session_state.error: # Check if detailed error was set
            st.error(f"Detailed error: {st.session_state.error}")
        if 'progress_bar' in st.session_state:
            try:
                st.session_state.progress_bar.empty()
            except Exception:
                pass
            if 'progress_bar' in st.session_state:
                try:
                    del st.session_state.progress_bar
                except KeyError:
                    pass

# NEW: Synchronous wrapper for execute_analysis_async
def execute_analysis(identifier: str, search_type: str, context: str, simulator_guids: List[str], llm_config: Dict[str, Any], qsar_config: Dict[str, Any], scope_config: Dict[str, Any] = None):
    """
    Synchronous wrapper to run the asynchronous analysis pipeline.
    Handles asyncio event loop management safely within Streamlit.
    """
    try:
        # Attempt to get the running event loop (Streamlit manages one)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running, create and set a new one (e.g., first run)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function until completion
        loop.run_until_complete(execute_analysis_async(
            identifier, search_type, context, simulator_guids, llm_config, qsar_config, scope_config
        ))

    except Exception as e:
        # Catch any unexpected errors during the execution wrapper logic
        # Check if streamlit context is available before using st.error
        try:
            st.error(f"An error occurred during analysis execution wrapper: {e}")
        except st.errors.StreamlitAPIException:
            print(f"An error occurred during analysis execution wrapper (no Streamlit context): {e}")
        logger.error(f"Error in execute_analysis wrapper: {e}")


# UPDATED: render_standard_mode is synchronous
def render_standard_mode():
    """Render the standard O-QT assistant interface"""
    
    # Always render methodology and transparency in sidebar first
    render_methodology_and_transparency()
    
    # Render downloads if we have completed analysis data
    if (st.session_state.get('final_report') or
        st.session_state.get('specialist_outputs_dict') or
        st.session_state.get('comprehensive_log')):
        try:
            identifier_display = st.session_state.comprehensive_log['inputs']['identifier']
        except (KeyError, TypeError):
            identifier_display = st.session_state.input_identifier or "previous_analysis"
        # This function now includes the PDF download button
        render_specialist_downloads(identifier_display)

    # --- Header ---
    st.title("🧪 O-QT Assistant: Multi-Agent Chemical Analysis")
    st.markdown("Automated Hazard Assessment, Metabolism Simulation, and Read-Across Recommendations using the OECD QSAR Toolbox")

    # --- Main Area: Search Input (No longer using st.form for better "Select All" functionality) ---
    
    # Check connection status proactively if configuration is complete (Needed for simulators)
    if st.session_state.qsar_config['config_complete'] and st.session_state.connection_status is None:
         try:
             api_client = QSARToolboxAPI(base_url=st.session_state.qsar_config['api_url'], timeout=(5, 10))
             check_connection(api_client)
             # Rerun might be needed to update UI elements relying on connection status (like simulator list)
             # st.rerun() # Avoid rerun here if possible, let the UI update naturally
         except Exception:
             st.session_state.connection_status = False

    identifier, search_type, context = render_search_section()

    # --- UPDATED: Metabolism Simulator Selection ---
    st.subheader("3. Metabolism Simulation")

    if st.session_state.connection_status:
        simulators = st.session_state.get('available_simulators', [])
        if simulators:
            simulator_options = {
                sim.get("Caption", sim.get("Guid", f"Simulator {idx}")): sim.get("Guid")
                for idx, sim in enumerate(simulators)
                if isinstance(sim, dict) and sim.get("Guid")
            }
            labels = list(simulator_options.keys())
            if "metabolism_selected_labels" not in st.session_state:
                st.session_state.metabolism_selected_labels = labels.copy()

            with st.expander("Select simulators", expanded=False):
                col_all, col_clear = st.columns(2)
                if col_all.button("Select all simulators"):
                    st.session_state.metabolism_selected_labels = labels.copy()
                    st.rerun()
                if col_clear.button("Clear simulators"):
                    st.session_state.metabolism_selected_labels = []
                    st.rerun()

                selected_labels = st.multiselect(
                    "Metabolism simulators",
                    options=labels,
                    default=st.session_state.metabolism_selected_labels,
                )
                st.session_state.metabolism_selected_labels = selected_labels

            st.session_state.selected_simulator_guids = [
                simulator_options[label] for label in st.session_state.metabolism_selected_labels
            ]
        else:
            st.info("No metabolism simulators available from the API.")
            st.session_state.selected_simulator_guids = []
    else:
        st.warning("Connect to the QSAR Toolbox (see sidebar) to enable simulator selection.")
        st.session_state.selected_simulator_guids = []


    # --- Profiler Selection ---
    st.subheader("4. Profiler Selection")

    if st.session_state.connection_status:
        profs = st.session_state.get('available_profilers', [])
        if profs:
            prof_options = {
                (p.get('Caption') or p.get('Name') or p.get('Guid')): p.get('Guid')
                for p in profs if isinstance(p, dict) and p.get('Guid')
            }
            slow_profiler_set = {guid for guid in prof_options.values() if guid in SLOW_PROFILER_GUIDS}
            slow_labels = [label for label, guid in prof_options.items() if guid in slow_profiler_set]
            fast_labels = [label for label in prof_options.keys() if label not in slow_labels]

            if "prof_selected_labels" not in st.session_state:
                st.session_state.prof_selected_labels = fast_labels.copy()
            if "include_slow_profilers" not in st.session_state:
                st.session_state.include_slow_profilers = False

            with st.expander("Select profilers", expanded=False):
                col_fast, col_all, col_clear = st.columns(3)
                if col_fast.button("Select fast profilers"):
                    st.session_state.prof_selected_labels = fast_labels.copy()
                    st.session_state.include_slow_profilers = False
                    st.rerun()
                if col_all.button("Select all profilers"):
                    st.session_state.prof_selected_labels = list(prof_options.keys())
                    st.session_state.include_slow_profilers = True
                    st.rerun()
                if col_clear.button("Clear profilers"):
                    st.session_state.prof_selected_labels = []
                    st.session_state.include_slow_profilers = False
                    st.rerun()

                include_slow = st.checkbox(
                    "Include ECHA profilers (~20 s each)",
                    value=st.session_state.get('include_slow_profilers', False),
                    help="Adds the three ECHA profilers that typically require ~20 seconds per run."
                )
                if include_slow != st.session_state.get('include_slow_profilers', False):
                    st.session_state.include_slow_profilers = include_slow
                    if include_slow:
                        combined = st.session_state.prof_selected_labels + [label for label in slow_labels if label not in st.session_state.prof_selected_labels]
                        st.session_state.prof_selected_labels = combined
                    else:
                        st.session_state.prof_selected_labels = [label for label in st.session_state.prof_selected_labels if label not in slow_labels]
                    st.rerun()

                selected_labels = st.multiselect(
                    "Profilers",
                    options=list(prof_options.keys()),
                    default=st.session_state.prof_selected_labels,
                )
                st.session_state.prof_selected_labels = selected_labels

            st.session_state.selected_profiler_guids = [
                prof_options[label] for label in st.session_state.prof_selected_labels
            ]
            st.session_state.include_slow_profilers = any(label in slow_labels for label in st.session_state.prof_selected_labels)
        else:
            st.info("Profiler catalog not available from the API.")
            st.session_state.selected_profiler_guids = []
            st.session_state.include_slow_profilers = False
    else:
        st.warning("Connect to the QSAR Toolbox to customise profilers.")
        st.session_state.selected_profiler_guids = []
        st.session_state.include_slow_profilers = False

    # --- QSAR Model Selection ---
    st.subheader("5. QSAR Model Predictions")

    if st.session_state.connection_status:
        qsar_catalog = st.session_state.get('available_qsar_models', [])
        if qsar_catalog:
            option_map = {}
            for entry in qsar_catalog:
                guid = entry.get("Guid")
                if not guid:
                    continue
                label = format_qsar_model_label(entry)
                option_map[label] = guid

            labels = list(option_map.keys())
            recommended_guids = st.session_state.get('recommended_qsar_model_guids', [])
            recommended_labels = [label for label, guid in option_map.items() if guid in recommended_guids]
            if not recommended_labels:
                recommended_labels = labels.copy()

            if "qsar_selected_labels" not in st.session_state or not st.session_state.qsar_selected_labels:
                st.session_state.qsar_selected_labels = recommended_labels.copy()
            else:
                # Ensure stored selections remain valid against current options
                valid_saved = [label for label in st.session_state.qsar_selected_labels if label in labels]
                if not valid_saved and recommended_labels:
                    valid_saved = recommended_labels.copy()
                if st.session_state.qsar_selected_labels != valid_saved:
                    st.session_state.qsar_selected_labels = valid_saved

            with st.expander("Select QSAR models", expanded=False):
                col_rec, col_all, col_clear = st.columns(3)
                if col_rec.button("Recommended set"):
                    st.session_state.qsar_selected_labels = recommended_labels.copy()
                    st.rerun()
                if col_all.button("Select all models"):
                    st.session_state.qsar_selected_labels = labels.copy()
                    st.rerun()
                if col_clear.button("Clear models"):
                    st.session_state.qsar_selected_labels = []
                    st.rerun()

                selected_labels = st.multiselect(
                    "QSAR models",
                    options=labels,
                    default=st.session_state.qsar_selected_labels,
                )
                st.session_state.qsar_selected_labels = selected_labels

            if not st.session_state.qsar_selected_labels:
                st.info("No QSAR models selected. The QSAR step will be skipped for this run.")

            st.session_state.selected_qsar_model_guids = [
                option_map[label] for label in st.session_state.qsar_selected_labels
            ]
        else:
            st.info("QSAR model catalog not yet loaded. Use ‘Check Connection’ in the sidebar to refresh.")
            st.session_state.selected_qsar_model_guids = []
    else:
        st.warning("Connect to the QSAR Toolbox to choose specific QSAR models.")
        st.session_state.selected_qsar_model_guids = []

    # Analyze Button placed prominently after inputs
    # Analyze Button placed prominently after inputs
    analyze_button_clicked = st.button("🚀 Analyze Chemical", type="primary", width="stretch")

    # Store inputs in session state for persistence
    st.session_state.input_identifier = identifier
    st.session_state.input_search_type = search_type
    st.session_state.input_context = context

    # --- Analysis Workflow ---
    # Check configuration status before analyzing
    is_qsar_ready = st.session_state.qsar_config['config_complete']
    is_llm_ready = st.session_state.llm_config['config_complete']
    is_ready_to_analyze = is_llm_ready and is_qsar_ready

    if analyze_button_clicked:
        # Perform initial checks (connection check is done here if not already established)
        # This check is redundant if the proactive check above runs, but safe to keep.
        if is_qsar_ready and st.session_state.connection_status is None:
             try:
                 api_client = QSARToolboxAPI(base_url=st.session_state.qsar_config['api_url'], timeout=(5, 10))
                 check_connection(api_client)
                 # Rerun to update status indicators
                 st.rerun()
             except Exception:
                 st.session_state.connection_status = False

        if not identifier:
            st.error("Please enter a chemical name or SMILES notation.")
        elif not is_ready_to_analyze:
             st.error("Configuration incomplete. Please ensure QSAR Toolbox URL and LLM API Key are provided in the sidebar.")
        elif st.session_state.connection_status is False:
             st.error("Cannot connect to QSAR Toolbox API. Please check the configuration and ensure the Toolbox is running.")
        else:
            # Run the analysis using the synchronous wrapper
            # Pass a minimal scope so profiling picks are respected even in Standard mode
            execute_analysis(
                identifier=identifier,
                search_type=search_type,
                context=context,
                simulator_guids=st.session_state.selected_simulator_guids,
                llm_config=st.session_state.llm_config,
                qsar_config=st.session_state.qsar_config,
                scope_config={
                    "include_properties": True,
                    "include_experimental": True,
                    "include_profiling": True,
                    "selected_profiler_guids": st.session_state.get('selected_profiler_guids', []),
                    "include_slow_profilers": st.session_state.get('include_slow_profilers', False),
                    "include_qsar": st.session_state.get('include_qsar_models', True),
                    "selected_qsar_model_guids": st.session_state.get('selected_qsar_model_guids', []),
                }
            )
            # Force rerun to display results and downloads
            st.rerun()

    # Display results (if they exist, either from current run or previous session)
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        # Determine identifier for display
        try:
            # Use the identifier stored in the log for consistency
            identifier_display = st.session_state.comprehensive_log['inputs']['identifier']
        except (KeyError, TypeError):
            identifier_display = st.session_state.input_identifier or "Analysis"

        # render_results_section uses st.session_state.analysis_results, which now contains processed/filtered data
        render_results_section(st.session_state.analysis_results, identifier_display)
        render_reports_section(identifier_display) # Show potentially existing report
        render_download_section(st.session_state.analysis_results, identifier_display)


# UPDATED: main is synchronous
def main():
    logo_candidates = [
        "o-qt_logo.jpg",
        "o-qt_logo.png",
        "logo.jpg",
        "logo.png",
    ]
    sidebar_logo = next((path for path in logo_candidates if os.path.exists(path)), None)

    st.set_page_config(
        page_title="O-QT Assistant",
        page_icon=sidebar_logo or "🧪",
        layout="wide"
    )

    initialize_session_state()

    # Display the logo at the top of the sidebar
    if sidebar_logo:
        st.sidebar.image(sidebar_logo, use_column_width=True)

    # Mode selection positioned directly under the logo
    # Determine index based on whether a wizard session is active
    if "wiz" in st.session_state:
        default_index = 1
    else:
        default_index = 0
    modes = ["Simple Mode", "Guided Mode"]
    ui_mode = st.sidebar.radio("Mode", modes, index=default_index, key="ui_mode_selector")

    # --- Sidebar Configuration and Information ---
    render_configuration_ui()

    # Use a key for the radio button to ensure state persistence
    if ui_mode == "Guided Mode":
        run_guided_wizard(
            ping_qsar=_ping_qsar,
            estimate_llm_cost=_estimate_cost,
            on_run_pipeline=_on_run_pipeline,
            get_llm_models=_get_llm_models,
        )
        # If we are in wizard mode, we stop rendering the rest of the app.
        st.stop()
    else:
        # If switching back from wizard (ui_mode is Standard), clear the wizard state if it exists
        if "wiz" in st.session_state:
            del st.session_state["wiz"]
            # Force a rerun to cleanly reload the standard interface
            st.rerun()

        render_standard_mode()

# Run the synchronous main function
if __name__ == "__main__":
    # Initialize logging configuration at startup
    logging.basicConfig(level=logging.INFO)
    main()
