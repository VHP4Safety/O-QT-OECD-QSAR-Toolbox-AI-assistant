# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

import streamlit as st
import sys
import os
import asyncio
# import json # No longer needed for specialist output formatting
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
# from pydantic import BaseModel # No longer needed

# Load environment variables from .env file (adjust path for src layout)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Use absolute imports based on the package structure
from oqt_assistant.utils.qsar_api import (
    QSARToolboxAPI, QSARConnectionError, QSARTimeoutError, QSARResponseError,
    SearchOptions                       #  ← add this [cite: 7]
)
# Import new agent functions
from oqt_assistant.utils.llm_utils import (
    analyze_chemical_context, # Added
    analyze_physical_properties,
    analyze_environmental_fate,
    analyze_profiling_reactivity,
    analyze_experimental_data,
    analyze_read_across, # Added
    synthesize_report
)
from oqt_assistant.components.search import render_search_section
from oqt_assistant.components.results import render_results_section, render_download_section # Keep render_download_section for raw data

# Define the maximum number of experimental records to send to LLM agents
MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA = 500

def initialize_session_state():
    """Initialize or reset session state variables"""
    defaults = {
        'chemical_name': '', # [cite: 8]
        'smiles': '', # [cite: 8]
        'search_type': 'name', # [cite: 8]
        'context': '', # [cite: 8]
        'analysis_results': None, # Raw data from QSAR API # [cite: 8]
        'final_report': None, # Synthesized report from agents # [cite: 8]
        'specialist_outputs_dict': None, # To store individual agent outputs # [cite: 8]
        'error': None, # [cite: 8]
        'connection_status': None, # [cite: 8]
        'progress_value': 0.0, # [cite: 9]
        'progress_description': '', # [cite: 9]
        'retry_count': 0, # [cite: 9]
        'max_retries': 3, # [cite: 9]
        'download_clicked': False # Keep for raw data downloads # [cite: 9]
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Keep update_progress and check_connection as they are
def update_progress(value: float, description: str):
    """Update progress bar with value and description""" # [cite: 10]
    st.session_state.progress_value = value # [cite: 10]
    st.session_state.progress_description = description # [cite: 10]
    # Ensure progress bar exists or create it
    if 'progress_bar' not in st.session_state: # [cite: 10]
        st.session_state.progress_bar = st.progress(0.0) # [cite: 10]
    # Check if progress bar element still exists before updating
    try:
        st.session_state.progress_bar.progress(value, text=f"Status: {description}") # [cite: 10]
    except Exception: # Handle cases where the element might have been removed # [cite: 10]
        st.session_state.progress_bar = st.progress(value, 
                                                    text=f"Status: {description}") # [cite: 11]


def check_connection(api_client: QSARToolboxAPI) -> bool:
    """Check if QSAR Toolbox API is accessible"""
    try:
        api_client.get_version()
        st.session_state.connection_status = True
        return True
    except QSARConnectionError:
        st.session_state.connection_status = False
        st.error("⚠️ Unable to connect to QSAR Toolbox API. Please check if the QSAR Toolbox is running.") # [cite: 11, 12]
        return False
    except Exception as e:
        st.session_state.connection_status = False
        st.error(f"⚠️ Unexpected error checking connection: {str(e)}") # [cite: 12]
        return False

# Modify render_reports_section to display the single final report
def render_reports_section(identifier: str):
    """Render the final synthesized report section."""
    st.header("Synthesized Analysis Report")

    if 'final_report' not in st.session_state:
        st.session_state.final_report = None

    if st.session_state.final_report: # [cite: 13]
        st.markdown(st.session_state.final_report) # [cite: 13]
        filename = f"{identifier}_synthesized_report.txt" # [cite: 13]
        st.download_button( # [cite: 13]
            label="Download Synthesized Report", # [cite: 13]
            data=st.session_state.final_report, # [cite: 13]
            file_name=filename, # [cite: 13]
            mime="text/plain", # [cite: 13]
            key="synthesized_report_download" # [cite: 13]
        )
    else: # [cite: 14]
        st.info("Synthesized report is not available or is being generated.") # [cite: 14]

# --- New function to render specialist downloads ---
def render_specialist_downloads(identifier: str):
    """Render download buttons for individual specialist agent reports."""
    st.sidebar.markdown("---") # Separator in sidebar
    st.sidebar.subheader("Download Specialist Analyses")

    if 'specialist_outputs_dict' not in st.session_state or not st.session_state.specialist_outputs_dict:
        st.sidebar.info("Specialist analyses not available.")
        return

    specialist_names = [
        "Chemical_Context", # Added [cite: 14, 15]
        "Physical_Properties", # [cite: 15]
        "Environmental_Fate", # [cite: 15]
        "Profiling_Reactivity", # [cite: 15]
        "Experimental_Data", # [cite: 15]
        "Read_Across" # Added [cite: 15]
    ]

    # Use the names directly as keys
    for name in specialist_names:
        report_content = st.session_state.specialist_outputs_dict.get(name)
        if report_content:
            filename = f"{identifier}_specialist_{name}.txt" # [cite: 15]
            st.sidebar.download_button( # [cite: 16]
                label=f"Download {name.replace('_', ' ')} Analysis", # [cite: 16]
                data=report_content, # [cite: 16]
                file_name=filename, # [cite: 16]
                mime="text/plain", # [cite: 16]
                key=f"specialist_{name}_download" # [cite: 16]
            ) # [cite: 17]
        else:
            st.sidebar.text(f"{name.replace('_', ' ')}: Not available") # [cite: 17]


# Modify perform_chemical_analysis to ONLY fetch data, remove report generation
def perform_chemical_analysis(identifier: str, search_type: str, context: str) -> Optional[Dict[str, Any]]:
    """Perform chemical data retrieval using QSAR Toolbox API (Synchronous)."""
    # This function remains largely synchronous as the API calls might need to be sequential
    # and we want all data before starting parallel agent analysis.
    try: # [cite: 18]
        # Initialize API client
        api_url = os.getenv('QSAR_TOOLBOX_API_URL')
        if not api_url:
            raise ValueError("QSAR_TOOLBOX_API_URL environment variable not set")

        api_client = QSARToolboxAPI(
            base_url=api_url,
            timeout=(10, 120),
            max_retries=st.session_state.max_retries
        ) # [cite: 19]

        st.write(f"Connecting to API at: {api_url}") # [cite: 19]
        if not check_connection(api_client): # [cite: 19]
            return None

        # --- Sequential Data Fetching ---
        update_progress(0.1, "🔍 Searching for chemical...") # [cite: 19]
        try:
            if search_type == 'name':
                api_client.search_by_name.cache_clear() # Clear cache for this specific call # [cite: 20]
                search_result = api_client.search_by_name( # [cite: 20]
                    identifier, # [cite: 20]
                    search_option=SearchOptions.EXACT_MATCH   # <- "0" # [cite: 20]
                )
            else: # [cite: 21]
                search_result = api_client.search_by_smiles(identifier) # [cite: 21]

            if not search_result: # [cite: 21]
                raise QSARResponseError(f"Chemical not found: {identifier}") # [cite: 21]
        except QSARTimeoutError:
             st.warning("Search request timed out, retrying...") # [cite: 21]
             if st.session_state.retry_count < st.session_state.max_retries: # [cite: 21]
                 st.session_state.retry_count += 1 # [cite: 22]
                 # Recursive call might be problematic in async context, consider loop/retry pattern if needed
                 return perform_chemical_analysis(identifier, search_type, context) # [cite: 22]
             else: # [cite: 22]
                 raise QSARTimeoutError("Maximum retries exceeded during chemical search") # [cite: 22]

        # --- Legacy selector: simply take the first record the API returns --- # [cite: 23]
        if isinstance(search_result, list): # [cite: 23]
            if not search_result: # [cite: 23]
                raise QSARResponseError(f"Chemical not found: {identifier}") # [cite: 23]
            selected_chemical_data = search_result[0]       # ← same as your 2024 build # [cite: 23]
        else:
            selected_chemical_data = search_result          # single-dict response # [cite: 24]

        chemical_data = selected_chemical_data # Use the selected data # [cite: 24]
        
        chem_id = chemical_data.get('ChemId')
        if not chem_id: # [cite: 24]
             raise QSARResponseError("Could not retrieve ChemId from search result.") # [cite: 24]


        update_progress(0.3, "📊 Calculating chemical properties...") # [cite: 24]
        try: # [cite: 25]
            raw_props = api_client.apply_all_calculators(chem_id) or {} # [cite: 25]
            # Flatten list‑of‑records → {parameter: value}
            properties = { # [cite: 25]
                (rec.get("Parameter") or rec.get("Name", f"prop_{i}")).strip(): rec.get("Value") # [cite: 25]
                for i, rec in enumerate(raw_props) if isinstance(rec, dict) # [cite: 25]
            } if isinstance(raw_props, list) else raw_props # Handle if API returns dict directly # [cite: 25, 26]
        except Exception as e: # [cite: 26]
            st.warning(f"Error retrieving or processing properties: {str(e)}") # [cite: 26]
            properties = {} # [cite: 26]

        update_progress(0.5, "🧪 Retrieving experimental data...") # [cite: 26]
        try:
            experimental_data = api_client.get_all_chemical_data(chem_id) or [] # [cite: 26]
        except Exception as e: # [cite: 27]
            st.warning(f"Error retrieving experimental data: {str(e)}") # [cite: 27]
            experimental_data = [] # [cite: 27]

        update_progress(0.7, "🔬 Retrieving profiling data...") # [cite: 27]
        try:
            profiling_data = api_client.get_chemical_profiling(chem_id) or {} # [cite: 27]
        except Exception as e:
            st.warning(f"Error retrieving profiling data: {str(e)}") # [cite: 27, 28]
            profiling_data = {'status': 'Error', 'note': f'Error retrieving profiling data: {str(e)}'} # [cite: 28]

        update_progress(0.8, "✅ QSAR data retrieval complete!") # [cite: 28]

        # Format results dictionary (without old report generation)
        results = { # [cite: 28]
            'chemical_data': { # [cite: 28]
                'basic_info': chemical_data, # [cite: 28]
                'properties': properties # [cite: 29]
            },
            'experimental_data': experimental_data, # This is the FULL experimental_data for UI/Download # [cite: 29]
            'profiling': profiling_data, # [cite: 29]
            'context': context # Pass context along # [cite: 29]
        }
        return results

    except Exception as e:
        st.session_state.error = str(e) # [cite: 29]
        # Ensure progress bar is removed or reset on error # [cite: 30]
        if 'progress_bar' in st.session_state: # [cite: 30]
            try:
                st.session_state.progress_bar.empty() # [cite: 30]
            except Exception: # [cite: 30]
                pass # Ignore if already gone # [cite: 30]
            del st.session_state.progress_bar # [cite: 31]
        raise # Re-raise the exception to be caught in main # [cite: 31]

# Make main async
async def main():
    st.set_page_config(
        page_title="QSAR Toolbox Assistant",
        page_icon="logo.png",
        layout="wide"
    )

    initialize_session_state()

    st.title("🧪O'QT: The OECD QSAR Toolbox AI Assistant")
    st.markdown("Multi-Agent Chemical Analysis, Hazard Assessment and Read-Across recommendations")

    # Display the logo
    st.image("logo.png", use_container_width=True) # Use container width

    api_url = os.getenv('QSAR_TOOLBOX_API_URL', 'Not set')
    st.sidebar.info(f"API URL: {api_url}")

    if st.session_state.connection_status is True: # [cite: 31, 32]
        st.sidebar.success("✅ Connected to QSAR Toolbox") # [cite: 32]
    elif st.session_state.connection_status is False: # [cite: 32]
        st.sidebar.error("❌ Not connected to QSAR Toolbox") # [cite: 32]
    else: # [cite: 32]
         # Initial check or if status is None
         st.sidebar.warning("Checking QSAR Toolbox connection...") # [cite: 32]
         # Perform an initial check non-blockingly if possible, or just wait for analysis button
         pass # [cite: 32]


    identifier, search_type, context = render_search_section() # [cite: 33]
    analyze_button = st.sidebar.button("Analyze Chemical")

    # --- Analysis Workflow ---
    if analyze_button:
        if not identifier:
            st.error("Please enter a chemical name or SMILES notation")
        else:
            # Reset state for new analysis
            st.session_state.analysis_results = None # [cite: 33]
            st.session_state.final_report = None # [cite: 34]
            st.session_state.specialist_outputs_dict = None # Reset specialist outputs # [cite: 34]
            st.session_state.error = None # [cite: 34]
            st.session_state.retry_count = 0 # [cite: 34]
            if 'exp_data_page' in st.session_state: # [cite: 34]
                st.session_state.exp_data_page = 1  # Reset to page 1 for new data # [cite: 34]
            # Create progress bar placeholder here # [cite: 35]
            st.session_state.progress_bar = st.progress(0.0, text="Status: Starting analysis...") # [cite: 35]


            try:
                # --- Step 1: Sequential Data Fetching ---
                results = perform_chemical_analysis(identifier, search_type, context)

                if results:
                    st.session_state.analysis_results = results # Store raw results (contains full experimental data) # [cite: 36]

                    # --- Step 2: Chemical Context Agent ---
                    update_progress(0.82, "🆔 Confirming Chemical Identity...") # [cite: 36]
                    original_context = context if context else "General chemical hazard assessment" # [cite: 36, 37]
                    chemical_data_for_context = results.get('chemical_data', {}) # [cite: 37]
                    confirmed_identity_str = await analyze_chemical_context(chemical_data_for_context, original_context) # [cite: 37]
                    # Prepend identity to context for other agents
                    analysis_context = f"{confirmed_identity_str}\n\nUser Goal: {original_context}" # [cite: 37, 38]
                    st.session_state.specialist_outputs_dict = {"Chemical_Context": confirmed_identity_str} # Store context output # [cite: 38]

                    # --- Step 3: Parallel Specialist Agent Analysis ---
                    update_progress(0.85, "🧠 Running core specialist agents...") # [cite: 38]

                    # Prepare data slices for agents
                    properties_data = results.get('chemical_data', {}).get('properties', {}) # Already fetched # [cite: 39]
                    profiling_data = results.get('profiling', {}) # [cite: 39]
                    
                    # **NEW: Handle experimental data truncation for LLM agents**
                    original_experimental_data_list = results.get('experimental_data', []) # [cite: 39]
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
                    
                    # Wrap experimental data list in a dict for consistent agent input type [cite: 40]
                    experimental_data_dict_for_llm = {"experimental_results": experimental_data_for_llm_processing} # [cite: 40]
                    if truncation_active_for_llm:
                        experimental_data_dict_for_llm["note_to_agent"] = \
                            f"The 'experimental_results' list provided has been truncated to the first {MAX_RECORDS_FOR_LLM_EXPERIMENTAL_DATA} records (plus a system note about this truncation) to manage data volume. Please base your analysis on this subset."


                    # Create tasks for specialist agents, passing specific data slices
                    task_phys = asyncio.create_task(analyze_physical_properties(properties_data, analysis_context)) # [cite: 40]
                    # Environmental fate often uses properties like LogKow, Solubility, Vapor Pressure # [cite: 41]
                    task_env = asyncio.create_task(analyze_environmental_fate(properties_data, analysis_context)) # [cite: 41]
                    task_prof = asyncio.create_task(analyze_profiling_reactivity(profiling_data, analysis_context)) # [cite: 41]
                    # Pass the potentially truncated and noted data to analyze_experimental_data
                    task_exp = asyncio.create_task(analyze_experimental_data(experimental_data_dict_for_llm, analysis_context)) # [cite: 41]

                    # Run core tasks concurrently - outputs should now be strings # [cite: 42]
                    core_specialist_outputs_list: List[str] = await asyncio.gather( # [cite: 42]
                        task_phys, # [cite: 42]
                        task_env, # [cite: 42]
                        task_prof, # [cite: 43]
                        task_exp # [cite: 43]
                    )

                    # Store core specialist outputs
                    st.session_state.specialist_outputs_dict.update({ # [cite: 44]
                        "Physical_Properties": str(core_specialist_outputs_list[0]), # [cite: 44]
                        "Environmental_Fate": str(core_specialist_outputs_list[1]), # [cite: 44]
                        "Profiling_Reactivity": str(core_specialist_outputs_list[2]), # [cite: 44]
                        "Experimental_Data": str(core_specialist_outputs_list[3]) # [cite: 45]
                    })

                    # --- Step 4: Read Across Agent ---
                    update_progress(0.90, "🧬 Analyzing Read-Across Potential...") # [cite: 45]
                    
                    # Prepare results for read_across agent, ensuring experimental data is truncated if needed for its LLM call
                    results_for_read_across_llm = results.copy() # Start with a copy of the full results
                    if truncation_active_for_llm:
                        # Replace 'experimental_data' in this copied dict with the truncated list + note
                        # `experimental_data_for_llm_processing` already contains this
                        results_for_read_across_llm['experimental_data'] = experimental_data_for_llm_processing
                    
                    # Pass full results (with potentially truncated experimental_data for LLM), the core outputs, and the enhanced context # [cite: 45, 46]
                    read_across_report = await analyze_read_across(
                        results_for_read_across_llm, # This version has experimental_data truncated if it was too large
                        core_specialist_outputs_list, 
                        analysis_context
                    ) # [cite: 46]
                    st.session_state.specialist_outputs_dict["Read_Across"] = read_across_report # Store read-across output # [cite: 46]


                    # --- Step 5: Synthesize Final Report ---
                    update_progress(0.95, "✍️ Synthesizing final report...") # [cite: 47]
                    # Pass the actual identifier, the core specialist outputs, the read-across report, and the original context
                    final_report_content = await synthesize_report( # [cite: 47]
                        chemical_identifier=identifier, # Use the actual input identifier # [cite: 47]
                        specialist_outputs=core_specialist_outputs_list, # Only the core 4 # [cite: 48]
                        read_across_report=read_across_report, # [cite: 48]
                        context=original_context # Use the original user context for the synthesizer's goal # [cite: 48]
                    )
                    st.session_state.final_report = final_report_content # [cite: 49]

                    update_progress(1.0, "✅ Analysis complete!") # [cite: 49]

                    # --- Step 6: Display Results ---

                    # --- Step 4: Display Results --- (Commented out, seems like a typo in original)
                    identifier_display = identifier # Use the input identifier for display consistency # [cite: 50]
                    # render_results_section uses st.session_state.analysis_results, which has FULL data
                    render_results_section(st.session_state.analysis_results, identifier_display) # Display raw data tables # [cite: 50]
                    render_reports_section(identifier_display) # Display synthesized report # [cite: 50]
                    render_download_section(st.session_state.analysis_results, identifier_display) # Offer raw data download # [cite: 50, 51]
                    render_specialist_downloads(identifier_display) # Offer specialist downloads # [cite: 51]

                else:
                    # Handle case where perform_chemical_analysis returned None (e.g., connection error handled there)
                    if 'progress_bar' in st.session_state: # [cite: 51]
                        try: # [cite: 52]
                             st.session_state.progress_bar.empty() # [cite: 52]
                        except Exception: # [cite: 52]
                             pass # [cite: 53]
                        del st.session_state.progress_bar # [cite: 53]


            except (QSARConnectionError, QSARTimeoutError, QSARResponseError) as qsar_err:
                st.error(f"🚫 QSAR API Error: {str(qsar_err)}") # [cite: 53]
                if 'progress_bar' in st.session_state: # [cite: 53]
                    try: # [cite: 54]
                        st.session_state.progress_bar.empty() # [cite: 54]
                    except Exception: # [cite: 54]
                        pass # [cite: 54]
                    del st.session_state.progress_bar # [cite: 54]
            except Exception as e: # [cite: 55]
                st.error(f"❌ Analysis failed unexpectedly: {str(e)}") # [cite: 55]
                if st.session_state.error: # Check if detailed error was set # [cite: 55]
                    st.error(f"Detailed error: {st.session_state.error}") # [cite: 55]
                if 'progress_bar' in st.session_state: # [cite: 55]
                    try: # [cite: 56]
                        st.session_state.progress_bar.empty() # [cite: 56]
                    except Exception: # [cite: 56]
                        pass # [cite: 56]
                    del st.session_state.progress_bar # [cite: 57]

    # Display previous results if they exist and button wasn't clicked
    elif st.session_state.analysis_results is not None: # [cite: 57]
        identifier_display = (st.session_state.chemical_name # [cite: 57]
                          if st.session_state.search_type == 'name' # [cite: 57]
                          else st.session_state.smiles) # [cite: 57]
        if not identifier_display: # Fallback if name/smiles weren't stored properly # [cite: 58]
             identifier_display = "previous_analysis" # [cite: 58]

        # render_results_section uses st.session_state.analysis_results, which has FULL data
        render_results_section(st.session_state.analysis_results, identifier_display) # [cite: 58]
        render_reports_section(identifier_display) # Show potentially existing report # [cite: 58]
        render_download_section(st.session_state.analysis_results, identifier_display) # [cite: 58]
        render_specialist_downloads(identifier_display) # Show specialist downloads if available # [cite: 58]

# Run the async main function
if __name__ == "__main__":
    # Ensure asyncio event loop compatibility with Streamlit if needed
    # For newer Python/Streamlit versions, asyncio.run() might be sufficient # [cite: 58, 59]
    # Older versions might need: asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) on Windows
    asyncio.run(main()) # [cite: 59]
