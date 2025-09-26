# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

import streamlit as st
import sys
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file (adjust path for src layout)
# This is kept for fallback/default values if users prefer environment variables
try:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))
except Exception:
    print("Could not load .env file (optional).")

# Use absolute imports based on the package structure
from oqt_assistant.utils.qsar_api import (
    QSARToolboxAPI, QSARConnectionError, QSARTimeoutError, QSARResponseError,
    SearchOptions
)
# Import new agent functions (Updated signatures)
from oqt_assistant.utils.llm_utils import (
    analyze_chemical_context,
    analyze_physical_properties,
    analyze_environmental_fate,
    analyze_profiling_reactivity,
    analyze_experimental_data,
    analyze_read_across,
    synthesize_report
)
from oqt_assistant.components.search import render_search_section
from oqt_assistant.components.results import render_results_section, render_download_section

# Define the maximum number of experimental records to send to LLM agents
MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA = 500

# Define LLM Models and Costs (Updated for transparency)
# Costs are per 1 Million tokens (Input/Output). 'id' is the actual model identifier for the API.
LLM_MODELS = {
    "OpenAI": {
        "gpt-4.1 (Recommended)": {"cost_input": 3.00, "cost_output": 12.00, "id": "gpt-4.1"},
        "gpt-4.1-mini": {"cost_input": 0.80, "cost_output": 3.20, "id": "gpt-4.1-mini"},
        "gpt-4.1-nano": {"cost_input": 0.20, "cost_output": 0.80, "id": "gpt-4.1-nano"},
    },
    "OpenRouter": {
        "OpenAI GPT-4.1": {"cost_input": 3.00, "cost_output": 12.00, "id": "openai/gpt-4.1"},
        "OpenAI GPT-4.1 Mini": {"cost_input": 0.80, "cost_output": 3.20, "id": "openai/gpt-4.1-mini"},
        "OpenAI GPT-4.1 Nano": {"cost_input": 0.20, "cost_output": 0.80, "id": "openai/gpt-4.1-nano"},
        "OpenAI GPT-4o": {"cost_input": 5.00, "cost_output": 15.00, "id": "openai/gpt-4o"},
        "DeepSeek V3 (Free) - High Performance": {"cost_input": 0.00, "cost_output": 0.00, "id": "deepseek/deepseek-chat:free"},
        "Llama 3.3 70B (Free) - High Performance": {"cost_input": 0.00, "cost_output": 0.00, "id": "meta-llama/llama-3.3-70b-instruct:free"},
        "Llama 4 Scout (Free) - Huge Context": {"cost_input": 0.00, "cost_output": 0.00, "id": "meta-llama/llama-4-scout:free"},
        "Qwen 3 235B (Free) - Multilingual": {"cost_input": 0.00, "cost_output": 0.00, "id": "qwen/qwen-3-235b:free"},
        "Gemma 3 27B (Free) - Efficient": {"cost_input": 0.00, "cost_output": 0.00, "id": "google/gemma-3-27b:free"},
        "DeepSeek V3.1 (Free)": {"cost_input": 0.00, "cost_output": 0.00, "id": "deepseek/deepseek-chat-v3.1:free"},
        "Meta Llama 3 8B (Free)": {"cost_input": 0.00, "cost_output": 0.00, "id": "meta-llama/llama-3-8b-instruct:free"},
        "Meta Llama 3.1 8B (Free)": {"cost_input": 0.00, "cost_output": 0.00, "id": "meta-llama/llama-3.1-8b-instruct:free"},
        "Mistral 7B Instruct (Free)": {"cost_input": 0.00, "cost_output": 0.00, "id": "mistralai/mistral-7b-instruct:free"},
        "Google Gemma 2 9B (Free)": {"cost_input": 0.00, "cost_output": 0.00, "id": "google/gemma-2-9b-it:free"},
        "Anthropic Claude 3.5 Sonnet": {"cost_input": 3.00, "cost_output": 15.00, "id": "anthropic/claude-3.5-sonnet"},
        "Google Gemini 1.5 Flash": {"cost_input": 0.35, "cost_output": 1.05, "id": "google/gemini-flash-1.5"},
        "Meta Llama 3 70B Instruct": {"cost_input": 0.59, "cost_output": 0.79, "id": "meta-llama/llama-3-70b-instruct"},
    }
}

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
        'input_details': {}
    }

    # --- Configuration State Initialization ---
    # Load defaults from environment variables if available, otherwise set to defaults
    if 'llm_config' not in st.session_state:
        # Default configuration
        default_provider = 'OpenAI'
        default_model_name = 'gpt-4.1 (Recommended)'
        
        # Attempt to load API key from environment variables
        # Prioritize OPENAI_API_KEY as a common default
        default_api_key = os.getenv('OPENAI_API_KEY', '')
        
        st.session_state.llm_config = {
            'provider': default_provider,
            'model_name': default_model_name, # Display name
            'api_key': default_api_key,
            'api_base': None, # Used for OpenRouter
            'config_complete': bool(default_api_key) # Complete if API key is pre-filled
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
        st.session_state.progress_bar = st.progress(0.0)
    # Check if progress bar element still exists before updating
    try:
        st.session_state.progress_bar.progress(value, text=f"Status: {description}")
    except Exception: # Handle cases where the element might have been removed
        st.session_state.progress_bar = st.progress(value,
                                                    text=f"Status: {description}")


def check_connection(api_client: QSARToolboxAPI) -> bool:
    """Check if QSAR Toolbox API is accessible"""
    try:
        api_client.get_version()
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

# --- Function to render specialist downloads and Comprehensive Log ---
def render_specialist_downloads(identifier: str):
    """Render download buttons for individual specialist agent reports and the comprehensive log."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Downloads")

    # Add Comprehensive Log Download (Addressing Transparency)
    if st.session_state.get('comprehensive_log'):
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
        specialist_names = [
            "Chemical_Context",
            "Physical_Properties",
            "Environmental_Fate",
            "Profiling_Reactivity",
            "Experimental_Data",
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


def generate_comprehensive_log(
    inputs: Dict[str, Any],
    llm_config: Dict[str, Any], qsar_config: Dict[str, Any],
    raw_qsar_data: Dict[str, Any], specialist_analyses: Dict[str, str],
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
            "tool_name": "O'QT Assistant",
            "version": "1.0.1" # Update as needed
        },
        "configuration": {
            "llm_configuration": masked_llm_config,
            "qsar_toolbox_configuration": qsar_config
        },
        "inputs": inputs,
        "data_retrieval": {
            "raw_qsar_toolbox_data": raw_qsar_data
        },
        "analysis": {
            "specialist_agent_outputs": specialist_analyses,
            "synthesized_report": synthesized_report
        }
    }
    return log


# Modify perform_chemical_analysis to use configured URL
def perform_chemical_analysis(identifier: str, search_type: str, context: str) -> Optional[Dict[str, Any]]:
    """Perform chemical data retrieval using QSAR Toolbox API (Synchronous)."""
    try:
        # Initialize API client using configuration from session state
        api_url = st.session_state.qsar_config.get('api_url')

        if not api_url:
            raise ValueError("QSAR Toolbox API URL is not configured.")

        api_client = QSARToolboxAPI(
            base_url=api_url,
            timeout=(10, 120),
            max_retries=15  # Use higher retry count for better reliability
        )

        # st.write(f"Attempting connection to API at: {api_url}")
        if not check_connection(api_client):
            # Error message is displayed by check_connection or the main loop status check
            return None

        # --- Sequential Data Fetching ---
        update_progress(0.1, "🔍 Searching for chemical...")
        try:
            if search_type == 'name':
                # Check if caching is used and clear it if necessary
                if hasattr(api_client.search_by_name, 'cache_clear'):
                     api_client.search_by_name.cache_clear()
                search_result = api_client.search_by_name(
                    identifier,
                    search_option=SearchOptions.EXACT_MATCH   # <- "0"
                )
            else:
                search_result = api_client.search_by_smiles(identifier)

            if not search_result:
                raise QSARResponseError(f"Chemical not found: {identifier}")
        except QSARTimeoutError:
             st.warning("Search request timed out, retrying...")
             if st.session_state.retry_count < st.session_state.max_retries:
                 st.session_state.retry_count += 1
                 # Recursive call might be problematic in async context, consider loop/retry pattern if needed
                 return perform_chemical_analysis(identifier, search_type, context)
             else:
                 raise QSARTimeoutError("Maximum retries exceeded during chemical search")

        # --- Legacy selector: simply take the first record the API returns ---
        if isinstance(search_result, list):
            if not search_result:
                raise QSARResponseError(f"Chemical not found: {identifier}")
            selected_chemical_data = search_result[0]       # ← same as your 2024 build
        else:
            selected_chemical_data = search_result          # single-dict response

        chemical_data = selected_chemical_data # Use the selected data

        chem_id = chemical_data.get('ChemId')
        if not chem_id:
             raise QSARResponseError("Could not retrieve ChemId from search result.")


        update_progress(0.3, "📊 Calculating chemical properties...")
        try:
            raw_props = api_client.apply_all_calculators(chem_id) or {}
            # Flatten list‑of‑records → {parameter: value} if needed, though qsar_api.py should handle this
            if isinstance(raw_props, list):
                properties = {
                    (rec.get("Parameter") or rec.get("Name", f"prop_{i}")).strip(): rec.get("Value")
                    for i, rec in enumerate(raw_props) if isinstance(rec, dict)
                }
            elif isinstance(raw_props, dict):
                 properties = raw_props
            else:
                 properties = {}
        except Exception as e:
            st.warning(f"Error retrieving or processing properties: {str(e)}")
            properties = {}

        update_progress(0.5, "🧪 Retrieving experimental data...")
        try:
            experimental_data = api_client.get_all_chemical_data(chem_id) or []
        except Exception as e:
            st.warning(f"Error retrieving experimental data: {str(e)}")
            experimental_data = []

        update_progress(0.7, "🔬 Retrieving profiling data...")
        try:
            profiling_data = api_client.get_chemical_profiling(chem_id) or {}
        except Exception as e:
            st.warning(f"Error retrieving profiling data: {str(e)}")
            profiling_data = {'status': 'Error', 'note': f'Error retrieving profiling data: {str(e)}'}

        update_progress(0.8, "✅ QSAR data retrieval complete!")

        # Format results dictionary (without old report generation)
        results = {
            'chemical_data': {
                'basic_info': chemical_data,
                'properties': properties
            },
            'experimental_data': experimental_data, # This is the FULL experimental_data for UI/Download
            'profiling': profiling_data,
            'context': context # Pass context along
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
               del st.session_state.progress_bar
        raise # Re-raise the exception to be caught in main

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
            st.sidebar.info(f"Cost (USD per 1M tokens):\nInput: ${cost_input:.2f} | Output: ${cost_output:.2f}")
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

    # Check configuration completeness
    st.session_state.llm_config['config_complete'] = bool(api_key and model_name)

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
    st.sidebar.header("About O'QT Assistant")

    with st.sidebar.expander("Licensing and Costs (Reviewer #1)"):
        st.markdown("""
        **O'QT Assistant is Open Source and Free.**

        This tool is licensed under the Apache 2.0 license. It is **not** a paid service.

        **Costs:** While the tool itself is free, using Large Language Models (LLMs) like GPT-4o incurs costs based on API usage. You must provide your own API key (OpenAI or OpenRouter), and you are responsible for the associated charges. We provide cost estimates in the configuration section. Free models are available via OpenRouter.
        """)

    with st.sidebar.expander("Methodology and Scope (Reviewer #1 & #3)"):
        st.markdown("""
        **Methodology:** O'QT employs a multi-agent LLM framework to automate the analysis of data retrieved directly from the OECD QSAR Toolbox API.

        **Scope and Rationale for Exclusions:**
        The current version focuses on core hazard assessment endpoints readily available via the QSAR Toolbox API. We have excluded:
        - **Toxicokinetics/ADME models:** These often require complex simulations and specific input parameters not fully exposed or easily automated via the current API, often leading to timeouts.
        - **Complex Mammalian Toxicity Endpoints (e.g., LD50/LC50):** Comprehensive assessment often requires integrating diverse data types and complex QSAR models which are better handled within the desktop application's dedicated workflows.

        **Data Extraction:** The agents extract available experimental data points retrieved from the Toolbox, including physicochemical parameters, toxicity endpoints, environmental fate parameters, etc., focusing specifically on records marked as "Measured value".
        """)

    with st.sidebar.expander("Transparency and Reliability (Reviewer #3)"):
        st.markdown("""
        **LLM Reliability:** We acknowledge the risk of LLMs generating incorrect information (hallucinations). We mitigate this by:
        1. **Strict Prompts:** Agents are instructed to ONLY use data provided by the QSAR Toolbox API.
        2. **Data Provenance:** The synthesized report strictly distinguishes between "Experimental (Toolbox)" and "QSAR Estimate (Toolbox)" data.
        3. **Separation of Data:** Raw data retrieved from the Toolbox is displayed separately from the LLM-generated analysis.

        **Profiler Selection:** The QSAR Toolbox includes numerous profilers. To ensure performance via the API, O'QT automatically selects a subset of fast-responding and broadly relevant profilers (e.g., DNA/Protein binding, functional groups). This selection is predetermined by the tool based on performance testing.

        **Comprehensive Log:** A complete JSON log of the analysis (configuration, inputs, raw data, and outputs) is available for download to ensure full transparency and reproducibility.
        """)

# Make main async
async def main():
    st.set_page_config(
        page_title="O'QT Assistant",
        page_icon="logo.png",
        layout="wide"
    )

    initialize_session_state()

    # Display the logo at the top of the sidebar
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)

    # --- Sidebar Configuration and Information ---
    render_configuration_ui()
    
    # Determine identifier for download filenames if results exist
    if st.session_state.analysis_results:
        try:
            identifier_display = st.session_state.comprehensive_log['inputs']['identifier']
        except (KeyError, TypeError):
            identifier_display = st.session_state.input_identifier or "previous_analysis"
        render_specialist_downloads(identifier_display)
        
    render_methodology_and_transparency()


    # --- Header ---
    st.title("🧪 O'QT: The Open QSAR Toolbox AI Assistant") # Updated Title
    st.markdown("Multi-Agent Chemical Analysis, Hazard Assessment and Read-Across Recommendations")

    # --- Main Area: Search Input (No longer using st.form for better "Select All" functionality) ---
    identifier, search_type, context = render_search_section()

    # Analyze Button placed prominently after inputs
    analyze_button_clicked = st.button("🚀 Analyze Chemical", type="primary", use_container_width=True)

    # Store inputs in session state for persistence
    st.session_state.input_identifier = identifier
    st.session_state.input_search_type = search_type
    st.session_state.input_context = context
    # Create inputs dict for logging purposes
    inputs = {
        "identifier": identifier,
        "search_type": search_type, 
        "context": context,
        "details": {}
    }

    # --- Analysis Workflow ---
    # Check configuration status before analyzing
    is_qsar_ready = st.session_state.qsar_config['config_complete']
    is_llm_ready = st.session_state.llm_config['config_complete']
    is_ready_to_analyze = is_llm_ready and is_qsar_ready

    if analyze_button_clicked:
        # Perform initial checks (connection check is done here if not already established)
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
            # Reset state for new analysis
            st.session_state.analysis_results = None
            st.session_state.final_report = None
            st.session_state.specialist_outputs_dict = None
            st.session_state.comprehensive_log = None # Reset log
            st.session_state.error = None
            st.session_state.retry_count = 0
            if 'exp_data_page' in st.session_state:
                st.session_state.exp_data_page = 1  # Reset to page 1 for new data
            # Create progress bar placeholder here
            st.session_state.progress_bar = st.progress(0.0, text="Status: Starting analysis...")

            # Get the configuration required for the agents (FIXED: Retrieve config here)
            current_llm_config = st.session_state.llm_config.copy()
            
            # Map the display model name to the actual model ID required by the API
            model_display_name = current_llm_config['model_name']
            provider = current_llm_config['provider']
            try:
                model_id = LLM_MODELS[provider][model_display_name]['id']
                current_llm_config['model_id'] = model_id
            except KeyError:
                st.error(f"Error: Could not find Model ID for {model_display_name} under {provider}.")
                return

            try:
                # --- Step 1: Sequential Data Fetching ---
                results = perform_chemical_analysis(identifier, search_type, context)

                if results:
                    st.session_state.analysis_results = results # Store raw results (contains full experimental data)

                    # --- Step 2: Chemical Context Agent ---
                    update_progress(0.82, "🆔 Confirming Chemical Identity...")
                    original_context = context if context else "General chemical hazard assessment"
                    chemical_data_for_context = results.get('chemical_data', {})
                    # FIXED: Pass llm_config
                    confirmed_identity_str = await analyze_chemical_context(chemical_data_for_context, original_context, current_llm_config)
                    # Prepend identity to context for other agents
                    analysis_context = f"{confirmed_identity_str}\n\nUser Goal: {original_context}"
                    st.session_state.specialist_outputs_dict = {"Chemical_Context": confirmed_identity_str} # Store context output

                    # --- Step 3: Parallel Specialist Agent Analysis ---
                    update_progress(0.85, "🧠 Running core specialist agents...")

                    # Prepare data slices for agents
                    properties_data = results.get('chemical_data', {}).get('properties', {}) # Already fetched
                    profiling_data = results.get('profiling', {})

                    # **NEW: Handle experimental data truncation for LLM agents**
                    original_experimental_data_list = results.get('experimental_data', [])
                    experimental_data_for_llm_processing = original_experimental_data_list
                    truncation_active_for_llm = False
                    truncation_note_for_llm = {}

                    if len(original_experimental_data_list) > MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA:
                        experimental_data_for_llm_processing = original_experimental_data_list[:MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA]
                        truncation_active_for_llm = True

                        truncation_note_content = (
                            f"Note: The original {len(original_experimental_data_list)} experimental data records "
                            f"have been automatically truncated to the first {MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA} records for this LLM-based analysis "
                            f"due to processing volume limits. The agent should summarize these initial records. "
                            f"The complete dataset remains available in the application's UI tables and for download."
                        )
                        # This dictionary structure should be compatible with typical experimental data items.
                        truncation_note_for_llm = {
                            "Endpoint": "System Information",
                            "Value": truncation_note_content,
                            "Unit": "LLM Processing Note",
                            "Reference": "QSAR Assistant System Notification",
                            "DataType": "SystemNote", # Adding a type hint
                            # Include other common fields if your experimental data items usually have them, e.g., with "N/A"
                            "TestGuid": "N/A",
                            "Reliability": "N/A"
                        }
                        # Prepend the note so the LLM encounters it first.
                        experimental_data_for_llm_processing = [truncation_note_for_llm] + experimental_data_for_llm_processing

                    # Wrap experimental data list in a dict for consistent agent input type
                    experimental_data_dict_for_llm = {"experimental_results": experimental_data_for_llm_processing}
                    if truncation_active_for_llm:
                        experimental_data_dict_for_llm["note_to_agent"] = \
                            f"The 'experimental_results' list provided has been truncated to the first {MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA} records (plus a system note about this truncation) to manage data volume. Please base your analysis on this subset."


                    # Create tasks for specialist agents, passing specific data slices
                    # FIXED: Pass llm_config to all agents
                    task_phys = asyncio.create_task(analyze_physical_properties(properties_data, analysis_context, current_llm_config))
                    # Environmental fate often uses properties like LogKow, Solubility, Vapor Pressure
                    task_env = asyncio.create_task(analyze_environmental_fate(properties_data, analysis_context, current_llm_config))
                    task_prof = asyncio.create_task(analyze_profiling_reactivity(profiling_data, analysis_context, current_llm_config))
                    # Pass the potentially truncated and noted data to analyze_experimental_data
                    task_exp = asyncio.create_task(analyze_experimental_data(experimental_data_dict_for_llm, analysis_context, current_llm_config))

                    # Run core tasks concurrently - outputs should now be strings
                    core_specialist_outputs_list: List[str] = await asyncio.gather(
                        task_phys,
                        task_env,
                        task_prof,
                        task_exp
                    )

                    # Store core specialist outputs
                    st.session_state.specialist_outputs_dict.update({
                        "Physical_Properties": str(core_specialist_outputs_list[0]),
                        "Environmental_Fate": str(core_specialist_outputs_list[1]),
                        "Profiling_Reactivity": str(core_specialist_outputs_list[2]),
                        "Experimental_Data": str(core_specialist_outputs_list[3])
                    })

                    # --- Step 4: Read Across Agent ---
                    update_progress(0.90, "🧬 Analyzing Read-Across Potential...")

                    # Prepare results for read_across agent, ensuring experimental data is truncated if needed for its LLM call
                    results_for_read_across_llm = results.copy() # Start with a copy of the full results
                    if truncation_active_for_llm:
                        # Replace 'experimental_data' in this copied dict with the truncated list + note
                        # `experimental_data_for_llm_processing` already contains this
                        results_for_read_across_llm['experimental_data'] = experimental_data_for_llm_processing

                    # Pass full results (with potentially truncated experimental_data for LLM), the core outputs, and the enhanced context
                    # FIXED: Pass llm_config
                    read_across_report = await analyze_read_across(
                        results_for_read_across_llm,
                        core_specialist_outputs_list,
                        analysis_context,
                        current_llm_config
                    )
                    st.session_state.specialist_outputs_dict["Read_Across"] = read_across_report # Store read-across output


                    # --- Step 5: Synthesize Final Report ---
                    update_progress(0.95, "✍️ Synthesizing final report...")
                    # Pass the actual identifier, the core specialist outputs, the read-across report, and the original context
                    # FIXED: Pass llm_config
                    final_report_content = await synthesize_report(
                        chemical_identifier=identifier,
                        specialist_outputs=core_specialist_outputs_list, # Only the core 4
                        read_across_report=read_across_report,
                        context=original_context, # Use the original user context for the synthesizer's goal
                        llm_config=current_llm_config
                    )
                    st.session_state.final_report = final_report_content

                    # --- Step 6: Generate Comprehensive Log ---
                    st.session_state.comprehensive_log = generate_comprehensive_log(
                        inputs,
                        current_llm_config, st.session_state.qsar_config,
                        results, st.session_state.specialist_outputs_dict, final_report_content
                    )

                    update_progress(1.0, "✅ Analysis complete!")

                else:
                    # Handle case where perform_chemical_analysis returned None (e.g., connection error handled there)
                    st.error("Analysis could not start due to data retrieval issues (e.g., connection error or chemical not found). Check configuration and inputs.")
                    if 'progress_bar' in st.session_state:
                        try:
                             st.session_state.progress_bar.empty()
                        except Exception:
                             pass
                        if 'progress_bar' in st.session_state:
                            del st.session_state.progress_bar


            except (QSARConnectionError, QSARTimeoutError, QSARResponseError) as qsar_err:
                st.error(f"🚫 QSAR API Error: {str(qsar_err)}")
                if 'progress_bar' in st.session_state:
                    try:
                        st.session_state.progress_bar.empty()
                    except Exception:
                        pass
                    if 'progress_bar' in st.session_state:
                        del st.session_state.progress_bar
            except Exception as e:
                st.error(f"❌ Analysis failed unexpectedly: {str(e)}")
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
                        del st.session_state.progress_bar

    # Display results (if they exist, either from current run or previous session)
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        # Determine identifier for display
        try:
            # Use the identifier stored in the log for consistency
            identifier_display = st.session_state.comprehensive_log['inputs']['identifier']
        except (KeyError, TypeError):
            identifier_display = st.session_state.input_identifier or "Analysis"

        # render_results_section uses st.session_state.analysis_results, which has FULL data
        render_results_section(st.session_state.analysis_results, identifier_display)
        render_reports_section(identifier_display) # Show potentially existing report
        render_download_section(st.session_state.analysis_results, identifier_display)

# Run the async main function
if __name__ == "__main__":
    # Ensure asyncio event loop compatibility with Streamlit if needed
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            # This can happen in some Streamlit environments; typically harmless on shutdown
            pass
        else:
            raise
