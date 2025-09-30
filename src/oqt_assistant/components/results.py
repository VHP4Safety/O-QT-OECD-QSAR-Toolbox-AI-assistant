# oqt_assistant/components/results.py
# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import json
import io
import logging

logger = logging.getLogger(__name__)

# Assuming safe_json is available if needed for downloads
try:
    from oqt_assistant.utils.data_formatter import safe_json
except ImportError:
    # Fallback if import fails
    def safe_json(data):
        return json.dumps(data, indent=2, default=str)

def render_results_section(results: Dict[str, Any], identifier: str):
    """Render the results section with tabs for different data types."""
    st.header(f"Analysis Results for: {identifier}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧪 Chemical Data",
        "📊 Properties (Calculated)",
        "📈 Experimental Data (Measured)",
        "🔬 Profiling & Reactivity",
        "🧬 Metabolism (Simulated)"
    ])

    # --- Tab 1: Chemical Data ---
    with tab1:
        _render_chemical_info(results.get('chemical_data', {}))

    # --- Tab 2: Properties (Calculated) ---
    with tab2:
        _render_properties(results.get('chemical_data', {}).get('properties', {}))

    # --- Tab 3: Experimental Data (Measured) ---
    with tab3:
        # Data is already processed (metadata parsed, year extracted) in app.py
        _render_experimental_data_interactive(results.get('experimental_data', []))

    # --- Tab 4: Profiling & Reactivity ---
    with tab4:
        _render_profiling_data(results.get('profiling', {}))

    # --- Tab 5: Metabolism (Simulated) ---
    with tab5:
        _render_metabolism_data(results.get('metabolism', {}))


def _render_chemical_info(chemical_data: Dict[str, Any]):
    """Render basic chemical information."""
    st.subheader("Chemical Identification")
    basic_info = chemical_data.get('basic_info', {})
    if basic_info:
        # Display key identifiers prominently
        st.markdown(f"**Name:** {basic_info.get('Name', 'N/A')}")
        st.markdown(f"**CAS:** {basic_info.get('Cas', 'N/A')}")
        st.markdown(f"**SMILES:** `{basic_info.get('Smiles', 'N/A')}`")
        st.markdown(f"**IUPAC Name:** {basic_info.get('IUPACName', 'N/A')}")
        st.markdown(f"**ChemID (Toolbox Internal):** {basic_info.get('ChemId', 'N/A')}")
        
        with st.expander("View Full Identification JSON"):
            st.json(basic_info)
    else:
        st.info("Chemical identification data not available.")

def _render_properties(properties: Dict[str, Any] | List[Dict[str, Any]]):
    """Render calculated properties."""
    st.subheader("Physicochemical Properties (Calculated/QSAR)")
    if not properties:
        st.info("No calculated properties available.")
        return

    try:
        # Shape 1: dict of {name -> {value, unit, type, family, calculator_name}}
        if isinstance(properties, dict) and all(isinstance(v, dict) for v in properties.values()):
            rows = []
            for param, info in properties.items():
                rows.append({
                    "Parameter": param,
                    "Value": info.get("value"),
                    "Unit": info.get("unit"),
                    "Type": info.get("type"),
                    "Calculator": info.get("calculator_name"),
                    "Family": info.get("family"),
                })
            df_props = pd.DataFrame(rows)
            # Stable sort by Parameter for readability
            df_props.sort_values(by=["Parameter"], inplace=True)
        # Shape 2: plain dict {key: value}
        elif isinstance(properties, dict):
            df_props = pd.DataFrame(list(properties.items()), columns=['Parameter', 'Value'])
        # Shape 3: list of dicts (fallback)
        elif isinstance(properties, list):
            df_props = pd.DataFrame(properties)
            # Try to normalize a common pattern
            possible_cols = ["Parameter", "Value", "Unit", "Type", "Calculator", "Family"]
            existing = [c for c in possible_cols if c in df_props.columns]
            if existing:
                df_props = df_props[existing]
        else:
            raise ValueError(f"Unexpected properties data format: {type(properties)}")
        
        st.dataframe(df_props, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not format properties data: {e}")
        st.json(properties)

# UPDATED: Enhanced experimental data rendering using interactive dataframe
def _render_experimental_data_interactive(experimental_data: List[Dict[str, Any]]):
    """Render experimental data in an interactive table with sorting and selection."""
    
    # Filter out system notes (like truncation messages) from the main data display
    actual_data_records = [r for r in experimental_data if isinstance(r, dict) and r.get('DataType') != 'SystemNote']

    if not actual_data_records:
        st.info("No experimental data retrieved or available after filtering.")
        return

    st.subheader(f"Experimental Data Records ({len(actual_data_records)})")

    # --- Data Preparation ---
    try:
        df = pd.DataFrame(actual_data_records)

        # Define desired columns and their order
        # 'Publication_Year' is added by data_formatter.py
        columns_to_display_map = {
            'Publication_Year': 'Year',
            'Endpoint': 'Endpoint',
            'Value': 'Value',
            'Unit': 'Unit',
            'DataType': 'Type',
            'Reliability': 'Reliability',
            'Reference': 'Reference',
            # We keep Parsed_Metadata internally for the detail view, but hide it in the main table
            'Parsed_Metadata': 'Parsed_Metadata' 
        }
        
        # Filter columns that actually exist in the DataFrame
        existing_columns = [col for col in columns_to_display_map.keys() if col in df.columns]
        
        df_display = df[existing_columns].copy()
        
        # Rename columns for display
        df_display.rename(columns=columns_to_display_map, inplace=True)

        # Handle the 'Year' column for proper sorting
        if 'Year' in df_display.columns:
            # Convert 'Year' to nullable integer type (Pandas >= 1.0.0)
            try:
                # Use Int64 (capital I) to handle potential None/NaN values gracefully
                df_display['Year'] = df_display['Year'].astype('Int64')
            except TypeError:
                # Fallback if conversion fails (e.g., non-numeric strings slipped through)
                df_display['Year'] = pd.to_numeric(df_display['Year'], errors='coerce')

            # NEW: Default Sorting (by Year descending, then Endpoint)
            try:
                # Sort by Year (descending) and Endpoint (ascending). Missing years (NaN/NaT) are placed last.
                df_display.sort_values(by=['Year', 'Endpoint'], ascending=[False, True], inplace=True, na_position='last')
            except Exception as e:
                st.warning(f"Could not apply default sorting: {e}")
                logger.warning(f"Sorting failed: {e}. DataFrame dtypes: {df_display.dtypes}")

    except Exception as e:
        st.error(f"Error preparing data table: {e}")
        st.json(experimental_data) # Fallback to raw JSON
        return

    # --- Display Table (Interactive) ---
    st.info("Data sorted by Year (Newest first). Click on a row to view detailed study metadata below.")

    # Define column configurations (Hiding Parsed_Metadata)
    column_config = {}
    # Identify the actual name used for Parsed_Metadata in df_display after renaming
    metadata_display_name = columns_to_display_map.get('Parsed_Metadata', 'Parsed_Metadata')

    if metadata_display_name in df_display.columns:
        # Hide this column from the main table view
        column_config[metadata_display_name] = None 

    # Use st.dataframe with on_select for direct row interaction
    selection = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun", # Rerun the app when a row is selected
        selection_mode="single-row",
        column_config=column_config
    )

    # --- Metadata Details View ---
    st.subheader("Study Metadata Details")
    
    selected_rows = selection.get("selection", {}).get("rows")
    
    if selected_rows:
        # Get the positional index of the selected row in the displayed (sorted) data
        selected_positional_index = selected_rows[0]
        
        try:
            # Retrieve the metadata directly from df_display using the positional index.
            # We must use iloc because the index might be non-sequential after sorting.
            selected_row_data = df_display.iloc[selected_positional_index]

            if metadata_display_name in df_display.columns:
                metadata = selected_row_data.get(metadata_display_name)
                
                # Check if metadata is a valid dictionary (Pandas might convert empty dicts to NaN)
                if metadata and isinstance(metadata, dict) and metadata:
                    # Display metadata using st.json for a clean, collapsible view of the structured data
                    st.json(metadata)
                else:
                    st.warning("No structured metadata available for the selected record (parsing might have failed).")
            else:
                    st.error("Metadata column not found in the data structure.")

        except IndexError:
            st.error("Selected index is out of bounds. Please try selecting again.")
        except Exception as e:
            st.error(f"Error retrieving metadata details: {e}")

    else:
        st.info("Select a record above to see details.")


def _render_profiling_data(profiling_data: Dict[str, Any]):
    """Render profiling data (aware of the new structure from qsar_api.get_chemical_profiling)."""
    st.subheader("Profiling and Reactivity Alerts")

    if not profiling_data:
        st.info("No profiling data retrieved.")
        return

    # High-level status
    status = profiling_data.get("status", "Unknown")
    note = profiling_data.get("note", "")
    if status == "Success":
        st.success(f"{status}: {note}")
    elif status in ("Partial success", "Limited"):
        st.warning(f"{status}: {note}")
    elif status == "Error":
        st.error(f"{status}: {profiling_data.get('error', 'No details provided')}")
    else:
        st.info(note or f"Status: {status}")

    # Available profilers (metadata)
    available = profiling_data.get("available_profilers", [])
    if available:
        with st.expander("Available profilers"):
            try:
                df_av = pd.DataFrame(available)[["name", "type", "guid", "status"]]
            except Exception:
                df_av = pd.DataFrame(available)
            st.dataframe(df_av, use_container_width=True, hide_index=True)

    # Actual results (per profiler)
    results = profiling_data.get("results", {})
    if not results:
        return

    st.markdown("---")
    st.markdown("#### Profiler Results")
    for profiler_name, info in results.items():
        payload = info.get("result")
        profiler_type = info.get("type", "Profiler")
        guid = info.get("guid", "")
        with st.expander(f"{profiler_name}  •  {profiler_type}  •  GUID: {guid}"):
            if isinstance(payload, list) and payload:
                try:
                    df = pd.DataFrame(payload)
                    # Select common columns if present
                    preferred = [c for c in ["Alert", "Category", "Explanation", "SubstanceCategory", "Endpoint"] if c in df.columns]
                    st.dataframe(df[preferred] if preferred else df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not display table for {profiler_name}: {e}")
                    st.json(payload)
            else:
                st.json(payload or {"note": "No alerts returned"})

def _render_metabolism_data(metabolism_data: Dict[str, Any]):
    """Render metabolism simulation results."""
    st.subheader("Metabolism Simulation Results")

    if not metabolism_data:
        st.info("Metabolism data not available.")
        return

    # Display overall status
    status = metabolism_data.get("status", "Unknown")
    note = metabolism_data.get("note", "N/A")

    if status == "Skipped":
        st.info(f"Status: {status}. {note}")
    elif status == "Success":
        st.success(f"Status: {status}. {note}")
    elif status == "Partial Success":
        st.warning(f"Status: {status}. {note}")
    else:
        st.error(f"Status: {status}. {note}")

    # Display individual simulation results
    simulations = metabolism_data.get("simulations", {})
    
    if simulations:
        st.markdown("---")
        st.markdown("#### Details per Simulator")

        for sim_guid, sim_result in simulations.items():
            sim_name = sim_result.get("simulator_name", f"GUID: {sim_guid}")
            sim_status = sim_result.get("status", "Unknown")
            metabolites = sim_result.get("metabolites", [])

            # Use expander for each simulator
            with st.expander(f"{sim_name} - Status: {sim_status} ({len(metabolites)} metabolites analyzed)"):
                st.markdown(f"**Details:** {sim_result.get('note', 'N/A')}")
                
                if metabolites:
                    try:
                        metabolites_df = pd.DataFrame(metabolites)
                        desired_cols = ['Smiles', 'Generation', 'Probability', 'Pathway', 'Name', 'Cas']
                        available_cols = [col for col in desired_cols if col in metabolites_df.columns]
                        
                        if available_cols:
                                st.dataframe(metabolites_df[available_cols], use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(metabolites_df, use_container_width=True, hide_index=True)
                    
                    except Exception as e:
                        st.warning(f"Could not display metabolites for {sim_name} in table format: {e}")
                        st.json(metabolites)

def render_download_section(results: Dict[str, Any], identifier: str):
    """Render section for downloading the raw QSAR Toolbox data."""
    st.header("Download Data")
    
    # Use the comprehensive log if available as it contains the processed data
    if 'comprehensive_log' in st.session_state and st.session_state.comprehensive_log:
        data_to_download = st.session_state.comprehensive_log.get('data_retrieval', {}).get('processed_qsar_toolbox_data', results)
        filename_suffix = "_processed_data.json"
        label = "Download Processed QSAR Data (JSON)"
    else:
        # Fallback to the raw results if log is not yet generated
        data_to_download = results
        filename_suffix = "_raw_data.json"
        label = "Download Raw QSAR Data (JSON)"

    try:
        # Safely convert to JSON string for download
        json_data = safe_json(data_to_download)
        
        st.download_button(
            label=label,
            data=json_data,
            file_name=f"{identifier}{filename_suffix}",
            mime="application/json",
            help="Download the data retrieved from QSAR Toolbox used for this analysis (includes metadata parsing and filters)."
        )
    except Exception as e:
        st.error(f"Error preparing data for download: {e}")

    # Download Experimental Data (CSV)
    exp_data = results.get('experimental_data', [])
    if exp_data:
        # Filter out SystemNotes before CSV download
        download_data = [r for r in exp_data if isinstance(r, dict) and r.get("DataType") != "SystemNote"]
        if download_data:
            try:
                df_exp = pd.DataFrame(download_data)
                # Convert DataFrame to CSV in memory
                csv_buffer = io.StringIO()
                # Drop internal processing columns from the CSV export
                df_exp.drop(columns=['Parsed_Metadata', 'Metadata'], errors='ignore').to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Experimental Data (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"{identifier}_experimental_data.csv",
                    mime="text/csv",
                    help="Download the processed experimental data table as a CSV file."
                )
            except Exception as e:
                st.error(f"Could not generate CSV download: {e}")
