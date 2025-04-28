import streamlit as st
import sys
import os
import asyncio
# import json # No longer needed for specialist output formatting
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
# from pydantic import BaseModel # No longer needed

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Use relative imports
from utils.qsar_api import QSARToolboxAPI, QSARConnectionError, QSARTimeoutError, QSARResponseError
# Import new agent functions
from utils.llm_utils import (
    analyze_physical_properties,
    analyze_environmental_fate,
    analyze_profiling_reactivity,
    analyze_experimental_data,
    synthesize_report
)
# Removed commented out import for old functions
from components.search import render_search_section
from components.results import render_results_section, render_download_section # Keep render_download_section for raw data

def initialize_session_state():
    """Initialize or reset session state variables"""
    defaults = {
        'chemical_name': '',
        'smiles': '',
        'search_type': 'name',
        'context': '',
        'analysis_results': None, # Raw data from QSAR API
        'final_report': None, # Synthesized report from agents
        'specialist_outputs_dict': None, # To store individual agent outputs
        'error': None,
        'connection_status': None,
        'progress_value': 0.0,
        'progress_description': '',
        'retry_count': 0,
        'max_retries': 3,
        'download_clicked': False # Keep for raw data downloads
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Keep update_progress and check_connection as they are
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
        st.session_state.progress_bar = st.progress(value, text=f"Status: {description}")


def check_connection(api_client: QSARToolboxAPI) -> bool:
    """Check if QSAR Toolbox API is accessible"""
    try:
        api_client.get_version()
        st.session_state.connection_status = True
        return True
    except QSARConnectionError:
        st.session_state.connection_status = False
        st.error("⚠️ Unable to connect to QSAR Toolbox API. Please check if the QSAR Toolbox is running.")
        return False
    except Exception as e:
        st.session_state.connection_status = False
        st.error(f"⚠️ Unexpected error checking connection: {str(e)}")
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
            label="Download Synthesized Report",
            data=st.session_state.final_report,
            file_name=filename,
            mime="text/plain",
            key="synthesized_report_download"
        )
    else:
        st.info("Synthesized report is not available or is being generated.")

# --- New function to render specialist downloads ---
def render_specialist_downloads(identifier: str):
    """Render download buttons for individual specialist agent reports."""
    st.sidebar.markdown("---") # Separator in sidebar
    st.sidebar.subheader("Download Specialist Analyses")

    if 'specialist_outputs_dict' not in st.session_state or not st.session_state.specialist_outputs_dict:
        st.sidebar.info("Specialist analyses not available.")
        return

    specialist_names = [
        "Physical_Properties",
        "Environmental_Fate",
        "Profiling_Reactivity",
        "Experimental_Data"
    ]

    for i, name in enumerate(specialist_names):
        report_content = st.session_state.specialist_outputs_dict.get(name)
        if report_content:
            filename = f"{identifier}_specialist_{name}.txt"
            st.sidebar.download_button(
                label=f"Download {name.replace('_', ' ')} Analysis",
                data=report_content,
                file_name=filename,
                mime="text/plain",
                key=f"specialist_{name}_download"
            )
        else:
            st.sidebar.text(f"{name.replace('_', ' ')}: Not available")


# Modify perform_chemical_analysis to ONLY fetch data, remove report generation
def perform_chemical_analysis(identifier: str, search_type: str, context: str) -> Optional[Dict[str, Any]]:
    """Perform chemical data retrieval using QSAR Toolbox API (Synchronous)."""
    # This function remains largely synchronous as the API calls might need to be sequential
    # and we want all data before starting parallel agent analysis.
    try:
        # Initialize API client
        api_url = os.getenv('QSAR_TOOLBOX_API_URL')
        if not api_url:
            raise ValueError("QSAR_TOOLBOX_API_URL environment variable not set")

        api_client = QSARToolboxAPI(
            base_url=api_url,
            timeout=(10, 120),
            max_retries=st.session_state.max_retries
        )

        st.write(f"Connecting to API at: {api_url}")
        if not check_connection(api_client):
            return None

        # --- Sequential Data Fetching ---
        update_progress(0.1, "🔍 Searching for chemical...")
        try:
            if search_type == 'name':
                search_result = api_client.search_by_name(identifier)
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

        if isinstance(search_result, list):
            if not search_result:
                raise QSARResponseError(f"No results found for: {identifier}")
            chemical_data = search_result[0]
        else:
            chemical_data = search_result
        chem_id = chemical_data.get('ChemId')
        if not chem_id:
             raise QSARResponseError("Could not retrieve ChemId from search result.")


        update_progress(0.3, "📊 Calculating chemical properties...")
        try:
            raw_props = api_client.apply_all_calculators(chem_id) or {}
            # Flatten list‑of‑records → {parameter: value}
            properties = {
                (rec.get("Parameter") or rec.get("Name", f"prop_{i}")).strip(): rec.get("Value")
                for i, rec in enumerate(raw_props) if isinstance(rec, dict)
            } if isinstance(raw_props, list) else raw_props # Handle if API returns dict directly
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
            'experimental_data': experimental_data,
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
            del st.session_state.progress_bar
        raise # Re-raise the exception to be caught in main

# Make main async
async def main():
    st.set_page_config(
        page_title="QSAR Toolbox Assistant",
        page_icon="🧪",
        layout="wide"
    )

    initialize_session_state()

    st.title("🧪 Chemical Hazard Assistant (Agentic)")
    st.markdown("Chemical Analysis and Hazard Assessment System - Multi-Agent Analysis")

    api_url = os.getenv('QSAR_TOOLBOX_API_URL', 'Not set')
    st.sidebar.info(f"API URL: {api_url}")

    if st.session_state.connection_status is True:
        st.sidebar.success("✅ Connected to QSAR Toolbox")
    elif st.session_state.connection_status is False:
        st.sidebar.error("❌ Not connected to QSAR Toolbox")
    else:
         # Initial check or if status is None
         st.sidebar.warning("Checking QSAR Toolbox connection...")
         # Perform an initial check non-blockingly if possible, or just wait for analysis button
         pass


    identifier, search_type, context = render_search_section()
    analyze_button = st.sidebar.button("Analyze Chemical")

    # --- Analysis Workflow ---
    if analyze_button:
        if not identifier:
            st.error("Please enter a chemical name or SMILES notation")
        else:
            # Reset state for new analysis
            st.session_state.analysis_results = None
            st.session_state.final_report = None
            st.session_state.specialist_outputs_dict = None # Reset specialist outputs
            st.session_state.error = None
            st.session_state.retry_count = 0
            # Create progress bar placeholder here
            st.session_state.progress_bar = st.progress(0.0, text="Status: Starting analysis...")


            try:
                # --- Step 1: Sequential Data Fetching ---
                results = perform_chemical_analysis(identifier, search_type, context)

                if results:
                    st.session_state.analysis_results = results # Store raw results

                    # --- Step 2: Parallel Specialist Agent Analysis ---
                    update_progress(0.85, "🧠 Running specialist agents...")
                    analysis_context = context if context else "General chemical hazard assessment"

                    # Prepare data slices for agents
                    properties_data = results.get('chemical_data', {}).get('properties', {})
                    profiling_data = results.get('profiling', {})
                    experimental_data_list = results.get('experimental_data', [])
                    # Wrap experimental data list in a dict for consistent agent input type
                    experimental_data_dict = {"experimental_results": experimental_data_list}


                    # Create tasks for specialist agents, passing specific data slices
                    task_phys = asyncio.create_task(analyze_physical_properties(properties_data, analysis_context))
                    # Environmental fate often uses properties like LogKow, Solubility, Vapor Pressure
                    task_env = asyncio.create_task(analyze_environmental_fate(properties_data, analysis_context))
                    task_prof = asyncio.create_task(analyze_profiling_reactivity(profiling_data, analysis_context))
                    task_exp = asyncio.create_task(analyze_experimental_data(experimental_data_dict, analysis_context))

                    # Run tasks concurrently - outputs should now be strings
                    specialist_outputs_list: List[str] = await asyncio.gather(
                        task_phys,
                        task_env,
                        task_prof,
                        task_exp
                    )

                    # Store individual string outputs in a dictionary for download
                    # (Handle potential non-string error returns just in case)
                    st.session_state.specialist_outputs_dict = {
                        "Physical_Properties": str(specialist_outputs_list[0]),
                        "Environmental_Fate": str(specialist_outputs_list[1]),
                        "Profiling_Reactivity": str(specialist_outputs_list[2]),
                        "Experimental_Data": str(specialist_outputs_list[3])
                    }


                    # --- Step 3: Synthesize Report ---
                    # synthesize_report now expects the list of strings directly
                    update_progress(0.95, "✍️ Synthesizing final report...")
                    final_report_content = await synthesize_report(specialist_outputs_list, analysis_context)
                    st.session_state.final_report = final_report_content

                    update_progress(1.0, "✅ Analysis complete!")

                    # --- Step 4: Display Results ---
                    identifier_display = identifier # Use the input identifier for display consistency
                    render_results_section(results, identifier_display) # Display raw data tables
                    render_reports_section(identifier_display) # Display synthesized report
                    render_download_section(results, identifier_display) # Offer raw data download
                    render_specialist_downloads(identifier_display) # Offer specialist downloads

                else:
                    # Handle case where perform_chemical_analysis returned None (e.g., connection error handled there)
                    if 'progress_bar' in st.session_state:
                         try:
                             st.session_state.progress_bar.empty()
                         except Exception:
                             pass
                         del st.session_state.progress_bar


            except (QSARConnectionError, QSARTimeoutError, QSARResponseError) as qsar_err:
                st.error(f"🚫 QSAR API Error: {str(qsar_err)}")
                if 'progress_bar' in st.session_state:
                    try:
                        st.session_state.progress_bar.empty()
                    except Exception:
                        pass
                    del st.session_state.progress_bar
            except Exception as e:
                st.error(f"❌ Analysis failed unexpectedly: {str(e)}")
                if st.session_state.error: # Check if detailed error was set
                    st.error(f"Detailed error: {st.session_state.error}")
                if 'progress_bar' in st.session_state:
                    try:
                        st.session_state.progress_bar.empty()
                    except Exception:
                        pass
                    del st.session_state.progress_bar

    # Display previous results if they exist and button wasn't clicked
    elif st.session_state.analysis_results is not None:
        identifier_display = (st.session_state.chemical_name
                          if st.session_state.search_type == 'name'
                          else st.session_state.smiles)
        if not identifier_display: # Fallback if name/smiles weren't stored properly
             identifier_display = "previous_analysis"

        render_results_section(st.session_state.analysis_results, identifier_display)
        render_reports_section(identifier_display) # Show potentially existing report
        render_download_section(st.session_state.analysis_results, identifier_display)
        render_specialist_downloads(identifier_display) # Show specialist downloads if available

# Run the async main function
if __name__ == "__main__":
    # Ensure asyncio event loop compatibility with Streamlit if needed
    # For newer Python/Streamlit versions, asyncio.run() might be sufficient
    # Older versions might need: asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) on Windows
    asyncio.run(main())
