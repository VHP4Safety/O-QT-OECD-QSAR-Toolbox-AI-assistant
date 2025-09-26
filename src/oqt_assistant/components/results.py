# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

import streamlit as st
import pandas as pd
import math
import json
import logging # Added logging
from typing import Dict, Any, List
from ..utils.data_formatter import clean_response_data

logger = logging.getLogger(__name__) # Initialize logger

def display_profilers_list(available_profilers):
    # ... (This function remains the same)
    """Helper method to display a list of profilers"""
    profilers_list = []
    
    if isinstance(available_profilers, list):
        # If it's already a list (standardized format), use it directly
        for profiler in available_profilers:
            profilers_list.append({
                "Profiler Name": profiler.get('name', 'Unknown'),
                "Type": profiler.get('type', 'Unknown'),
                "Status": profiler.get('status', 'Unknown'),
                # Description might not be in the standardized list, add a default if needed
                "Description": profiler.get('description', 'A chemical profiler that categorizes chemicals based on structural features')
            })
    elif isinstance(available_profilers, dict):
        # If it's a dict (legacy format), convert to list
        for key, profiler in available_profilers.items():
            profilers_list.append({
                "Profiler Name": profiler.get('name', key),
                "Type": profiler.get('type', 'Unknown'),
                "Status": profiler.get('status', 'Unknown'),
                "Description": profiler.get('description', 'A chemical profiler that categorizes chemicals based on structural features')
            })
    
    if profilers_list:
        profilers_df = pd.DataFrame(profilers_list)
        st.dataframe(profilers_df, use_container_width=True)
        return profilers_list
    else:
        st.info("No profiler information available")
        return []


# UPDATED FUNCTION (Case-insensitive column exclusion)
def render_results_section(results: Dict[str, Any], identifier_display: str):
    """Render the analysis results in an interactive dashboard"""
    st.header(f"Analysis Results for {identifier_display}")
    
    # Clean and format the data (this now includes structured metadata parsing)
    cleaned_results = clean_response_data(results)
    
    # Context-specific analysis
    if results.get('context'):
        with st.expander("View Analysis Context"):
            st.info(f"{results['context']}")
    
    # Tabs for different result categories
    tabs = st.tabs([
        "Chemical Overview",
        "Properties",
        "Experimental Data",
        "Profiling",
        "Metabolism"
    ])
    
    with tabs[0]:
        st.subheader("Chemical Information")
        if cleaned_results["chemical_data"]:
            chemical_info = pd.DataFrame([cleaned_results["chemical_data"]]).transpose()
            chemical_info.columns = ["Value"]
            st.dataframe(chemical_info, use_container_width=True)
        else:
            st.info("No chemical information available")
    
    with tabs[1]:
        st.subheader("Chemical Properties")
        if cleaned_results["properties"]:
            # Convert properties to DataFrame
            properties_list = []
            for name, data in cleaned_results["properties"].items():
                 # Ensure data is a dictionary before accessing keys
                if isinstance(data, dict):
                    properties_list.append({
                        "Property": name,
                        "Value": data.get("value"),
                        "Unit": data.get("unit"),
                        "Type": data.get("type"),
                        "Category": data.get("family")
                    })
                else:
                     # Handle cases where the value is not a dict (e.g., direct value from older API versions)
                     properties_list.append({
                        "Property": name,
                        "Value": data,
                        "Unit": "N/A",
                        "Type": "Unknown",
                        "Category": "Unknown"
                    })
            props_df = pd.DataFrame(properties_list)
            st.dataframe(props_df, use_container_width=True)
        else:
            st.info("No property data available")

    # UPDATED TAB 2: Experimental Data with interactive metadata viewer
    with tabs[2]:
        st.subheader("Experimental Data and Metadata")

        with st.expander("About the Data Extracted (Reviewer #1)"):
            st.markdown("""
            This table displays the raw experimental data retrieved from the OECD QSAR Toolbox. 
            **Select a row in the table below to view detailed study metadata.**

            **Data Schema:**
            - **Endpoint, Value, Unit, DataType, Reliability, Reference/TestGuid:** Standard fields.
            - **Metadata:** Detailed study conditions (e.g., Author, Year, Test Organism) are parsed and available upon row selection.

            **AI Analysis Focus:** The AI agents utilize this metadata for context. Data labeled "Measured value." is strictly reported as **"Experimental (Toolbox)"** in the synthesized report.
            """)

        # Data is already processed by clean_response_data
        if cleaned_results["experimental_data"]:
            exp_df = pd.DataFrame(cleaned_results["experimental_data"])
            
            # Prepare DataFrame for display
            if not exp_df.empty:
                # Define columns to display in the main table.
                # IMPROVEMENT: Case-insensitive exclusion to handle 'Metadata', 'metadata', etc.
                # This ensures that even if the backend processing somehow missed a variant, the frontend filters it.
                columns_to_exclude = {'parsed_metadata', 'metadata', 'processing_error'}
                columns_to_display = [col for col in exp_df.columns if col.lower() not in columns_to_exclude]
                
                # Check if we have columns left to display
                if not columns_to_display:
                    # If all columns were excluded (e.g. only error messages), show relevant info
                    st.warning("No standard data columns available to display. Showing error overview.")
                    # Filter to show only error columns if they exist
                    error_cols = [col for col in exp_df.columns if col.lower() in {'processing_error', 'raw_value'}]
                    if error_cols:
                         display_df = exp_df[error_cols].copy()
                    else:
                         display_df = pd.DataFrame() # Empty dataframe
                else:
                    display_df = exp_df[columns_to_display].copy()
                
                # Fix PyArrow serialization issues
                if not display_df.empty:
                    if 'Value' in display_df.columns:
                        display_df['Value'] = display_df['Value'].astype(str)
                    
                    for col in display_df.columns:
                        if display_df[col].dtype == 'object':
                            # Handle potential lists within cells if any remain
                            display_df[col] = display_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

                # --- Interactive Table Display (using st.dataframe with selection) ---

                try:
                    st.info(f"Showing {len(display_df)} records. Select a row to view details.")
                    
                    # Using the modern approach for selection feedback (Streamlit >= 1.35 recommended)
                    event = st.dataframe(
                        display_df, 
                        use_container_width=True,
                        on_select="rerun", # Rerun the script when a row is selected
                        selection_mode="single-row"
                    )
                    
                    selected_rows_indices = event.selection.get("rows", [])

                except Exception as e:
                    # Fallback for older Streamlit versions or if selection fails
                    st.warning(f"Interactive table selection not available or failed. Displaying static table. If you wish to use interactive features, please upgrade Streamlit (>= 1.35). Error: {e}")
                    st.dataframe(display_df, use_container_width=True)
                    selected_rows_indices = []

                # --- Metadata Viewer ---
                st.markdown("---")
                st.subheader("Study Metadata Details")
                
                if selected_rows_indices:
                    # Get the index of the selected row
                    selected_index = selected_rows_indices[0]
                    
                    # Retrieve the corresponding record from the original (non-displayed) DataFrame
                    # We use the index derived from the event on the display_df to access the full data in exp_df
                    try:
                        selected_record = exp_df.iloc[selected_index]
                        
                        # Extract the parsed metadata
                        metadata = selected_record.get('Parsed_Metadata', {})
                        
                        if metadata:
                            # Display metadata cleanly using a DataFrame
                            try:
                                meta_detail_df = pd.DataFrame(list(metadata.items()), columns=['Field', 'Value'])
                                st.dataframe(meta_detail_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Error displaying metadata details: {e}")
                                st.json(metadata) # Fallback to JSON if DataFrame fails
                        else:
                            st.info("No structured metadata available for the selected record.")

                    except IndexError:
                        # Handle case where index might be out of bounds (e.g. if data changed rapidly)
                        st.error("Error retrieving selected record. Please try selecting again.")

                else:
                    st.info("Select a row from the table above to view detailed study metadata.")

            else:
                st.info("No experimental data records found.")
        else:
            st.info("No experimental data available for this chemical")
    
    # Tabs 3 and 4 (Profiling and Metabolism) remain the same.
    with tabs[3]:
        st.subheader("Chemical Profiling")
        
        # Handle the profiling data structure from our updated API
        if isinstance(cleaned_results["profiling"], dict):
            # Show profiling status
            status = cleaned_results["profiling"].get("status", "Unknown")
            note = cleaned_results["profiling"].get("note", "")
            
            # Different display based on status
            if status == "Success" or status == "Partial success":
                st.success(f"Status: {status}")
                if note:
                    st.info(note)
            elif status == "Limited":
                st.warning(f"Status: {status}")
                if note:
                    st.info(note)
            elif status == "Error":
                st.error(f"Status: {status}")
                if note:
                    st.warning(note)
            else:
                 if note:
                    st.info(note)
            
            # First check if we have actual profiling results
            if 'results' in cleaned_results["profiling"] and cleaned_results["profiling"]["results"]:
                # Create subtabs for profiling results and available profilers
                profiling_tabs = st.tabs(["Profiling Results", "Available Profilers"])
                
                with profiling_tabs[0]:
                    st.subheader("Profiling Results")
                    
                    results = cleaned_results["profiling"]["results"]
                    
                    # Handle different result structures
                    if isinstance(results, list):
                        # Handle results from profiling/all/{chemId} endpoint (if used)
                        # ... (List handling logic omitted for brevity as the API uses the dict approach)
                        st.warning("List format detected (unexpected).")
                        st.json(results)
                    
                    elif isinstance(results, dict):
                        # Handle results from individual profilers (e.g., the fast profilers approach)

                        # Convert to a standardized list format for easier display
                        results_list = []
                        for profiler_name, profiler_data in results.items():
                            result_data = profiler_data.get('result', [])
                            profiler_type = profiler_data.get('type', 'Unknown')

                            if isinstance(result_data, list) and result_data:
                                for category in result_data:
                                    # Handle if category is a dict (sometimes API returns complex objects)
                                    if isinstance(category, dict):
                                         category_name = category.get("Name") or category.get("Category", str(category))
                                    else:
                                         category_name = str(category)

                                    results_list.append({
                                        "Profiler": profiler_name,
                                        "Type": profiler_type,
                                        "Category": category_name
                                    })
                            else:
                                # No categories found or unexpected format
                                results_list.append({
                                    "Profiler": profiler_name,
                                    "Type": profiler_type,
                                    "Category": "No categories found or unexpected format"
                                })
                        
                        if results_list:
                            # Show as dataframe
                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.warning("No categories found in profiling results")

                    else:
                        # Unknown result structure
                        st.json(results)
                
                with profiling_tabs[1]:
                    st.subheader("Available Profilers")
                    available_profilers = cleaned_results["profiling"].get("available_profilers", [])
                    display_profilers_list(available_profilers)
            
            # If we only have available profilers (and no results)
            elif 'available_profilers' in cleaned_results["profiling"]:
                st.subheader("Available Profilers")
                available_profilers = cleaned_results["profiling"]["available_profilers"]
                display_profilers_list(available_profilers)

                # Add explanation of what profilers do
                st.markdown("""
                ### What are chemical profilers?
                
                Chemical profilers are tools that analyze a chemical's structure to identify specific features or structural alerts 
                that may be associated with particular toxicological effects or mechanisms of action. They help categorize chemicals 
                and identify potential concerns based on structural similarity to known problematic chemicals.
                """)

            # Handle older structures if necessary (fallback)
            elif 'results' in cleaned_results["profiling"] or 'profilers' in cleaned_results["profiling"]:
                # ... (Legacy handling logic remains as a fallback if needed)
                st.warning("Legacy profiling data structure detected.")
                # st.json(cleaned_results["profiling"])
        else:
            st.info("No profiling data available")

    # UPDATED TAB: Metabolism (Handles Multi-simulator display and robust data handling)
    with tabs[4]: 
        st.subheader("Metabolism Simulation Results")
        metabolism_data = cleaned_results.get("metabolism", {})

        if metabolism_data:
            overall_status = metabolism_data.get("status", "Unknown")
            overall_note = metabolism_data.get("note", "")
            simulations = metabolism_data.get("simulations", {})

            # Display overall status
            if overall_status == "Success":
                st.success(f"Overall Status: {overall_status}. {overall_note}")
            elif overall_status == "Skipped":
                st.info(f"Overall Status: {overall_status}. {overall_note}")
            elif overall_status == "Partial Success":
                st.warning(f"Overall Status: {overall_status}. {overall_note}")
            elif overall_status == "Failed" or overall_status == "Error":
                st.error(f"Overall Status: {overall_status}. {overall_note}")
            
            # Display results for each simulator
            if simulations:
                st.markdown("---")
                st.subheader("Detailed Results per Simulator")
                
                # Use expanders for each simulator's results
                for guid, simulation_result in simulations.items():
                    sim_name = simulation_result.get("simulator_name", f"GUID: {guid}")
                    sim_status = simulation_result.get("status", "Unknown")
                    sim_note = simulation_result.get("note", "")
                    metabolites = simulation_result.get("metabolites", [])

                    # Determine the expander title color/icon based on status
                    if sim_status == "Success":
                        title = f"✅ {sim_name}"
                    elif sim_status == "Failed" or sim_status == "Error":
                        title = f"❌ {sim_name}"
                    else:
                        title = f"ℹ️ {sim_name}"

                    with st.expander(title):
                        st.info(f"Status: {sim_status}. Note: {sim_note}")
                        
                        if metabolites:
                            try:
                                # Convert to DataFrame for display
                                # Use the robust conversion logic here as well (mirrors create_downloadable_data)
                                processed_metabolites = []
                                for metabolite in metabolites:
                                    if isinstance(metabolite, dict):
                                        processed_metabolites.append(metabolite)
                                    elif isinstance(metabolite, str):
                                        # Handle string format (e.g., SMILES) if API standardization missed it
                                        processed_metabolites.append({"SMILES": metabolite, "Note": "Simple structure returned by API"})
                                    else:
                                        processed_metabolites.append({"Note": f"Unexpected data type: {type(metabolite)}", "RawValue": str(metabolite)})

                                meta_df = pd.DataFrame(processed_metabolites)

                                # Ensure consistent data types for PyArrow
                                if not meta_df.empty:
                                    for col in meta_df.columns:
                                        if meta_df[col].dtype == 'object':
                                            # Convert lists (like Names) to strings
                                            meta_df[col] = meta_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

                                st.dataframe(meta_df, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error displaying metabolism data for {sim_name}: {e}")
                                st.json(metabolites) # Fallback to JSON display
        else:
            st.info("No metabolism data available.")


# create_downloadable_data and render_download_section remain the same.
def create_downloadable_data(results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Create downloadable DataFrames from results"""
    cleaned_results = clean_response_data(results)
    downloads = {}
    
    # Properties (Remains the same)
    if cleaned_results["properties"]:
        properties_list = []
        for name, data in cleaned_results["properties"].items():
            if isinstance(data, dict):
                properties_list.append({
                    "Property": name,
                    "Value": data.get("value"),
                    "Unit": data.get("unit"),
                    "Type": data.get("type"),
                    "Category": data.get("family")
                })
            else:
                 properties_list.append({
                        "Property": name,
                        "Value": data,
                        "Unit": "N/A",
                        "Type": "Unknown",
                        "Category": "Unknown"
                    })
        downloads['properties'] = pd.DataFrame(properties_list)
    
    # Experimental Data (Updated to flatten the structured metadata for CSV)
    if cleaned_results["experimental_data"]:
        # We need to flatten the 'Parsed_Metadata' dictionary back into columns for the CSV export
        try:
            exp_data_list = cleaned_results["experimental_data"]
            
            # Use pandas json_normalize to flatten the 'Parsed_Metadata' column
            
            # Ensure 'Parsed_Metadata' exists in all records before normalization
            for record in exp_data_list:
                # Handle potential non-dict records that might have slipped through (e.g. error records)
                if not isinstance(record, dict):
                    continue
                if 'Parsed_Metadata' not in record or record['Parsed_Metadata'] is None:
                    record['Parsed_Metadata'] = {}

            # Identify keys other than 'Parsed_Metadata' to use as 'meta' (i.e., columns to keep)
            if exp_data_list:
                # Get all unique keys across all records, excluding Parsed_Metadata and potential error fields
                all_keys = set().union(*(d.keys() for d in exp_data_list if isinstance(d, dict)))
                # Case-insensitive exclusion for robustness
                keys_to_exclude = {'parsed_metadata', 'metadata', 'processing_error'}
                meta_keys = [key for key in all_keys if key.lower() not in keys_to_exclude]
            else:
                meta_keys = []

            # Normalize the data
            # This will flatten 'Parsed_Metadata' while keeping the 'meta_keys' alongside
            df_exp = pd.json_normalize(
                exp_data_list,
                errors='ignore',
                meta=meta_keys if meta_keys else None,
                record_path=None,
                meta_prefix=None,
            )
            
            # When record_path is None, normalization typically puts metadata under 'Parsed_Metadata.XXX'
            
            if any(col.startswith('Parsed_Metadata.') for col in df_exp.columns):
                 # Rename columns: Parsed_Metadata.Key -> Meta_Key
                 rename_dict = {col: f"Meta_{col.split('.', 1)[1]}" for col in df_exp.columns if col.startswith('Parsed_Metadata.')}
                 df_exp = df_exp.rename(columns=rename_dict)
            
            # Drop the original Parsed_Metadata column if it still exists
            if 'Parsed_Metadata' in df_exp.columns:
                    df_exp = df_exp.drop(columns=['Parsed_Metadata'])
            
            # Also drop 'Metadata' or 'Processing_Error' if they exist (case-insensitive cleanup)
            cols_to_drop = [col for col in df_exp.columns if col.lower() in {'metadata', 'processing_error'}]
            if cols_to_drop:
                df_exp = df_exp.drop(columns=cols_to_drop)

            downloads['experimental'] = df_exp

        except Exception as e:
            # Fallback if normalization fails
            logger.error(f"Error flattening experimental data for download: {e}. Falling back to basic DataFrame.")
            df_exp = pd.DataFrame(cleaned_results["experimental_data"])
            # Serialize the dictionary column for CSV if normalization failed
            if 'Parsed_Metadata' in df_exp.columns:
                df_exp['Parsed_Metadata_JSON'] = df_exp['Parsed_Metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
                df_exp = df_exp.drop(columns=['Parsed_Metadata'])
            downloads['experimental'] = df_exp

    
    # Profiling Data (Remains the same)
    if cleaned_results["profiling"] and isinstance(cleaned_results["profiling"], dict):
        results_list = []
        profiling_results = cleaned_results["profiling"].get("results")

        if profiling_results:
            if isinstance(profiling_results, dict):
                # Handle individual profilers (fast profilers approach)
                for profiler_name, profiler_data in profiling_results.items():
                    result_data = profiler_data.get('result', [])
                    profiler_type = profiler_data.get('type', 'Unknown')

                    if isinstance(result_data, list) and result_data:
                        for category in result_data:
                            if isinstance(category, dict):
                                 category_name = category.get("Name") or category.get("Category", str(category))
                            else:
                                 category_name = str(category)

                            results_list.append({
                                "Profiler": profiler_name,
                                "Type": profiler_type,
                                "Category": category_name
                            })
                    else:
                        results_list.append({
                            "Profiler": profiler_name,
                            "Type": profiler_type,
                            "Category": "No categories found"
                        })
            # ... (List handling omitted for brevity)

        if results_list:
            downloads['profiling_results'] = pd.DataFrame(results_list)

    # Metabolism Data (Remains the same, handling AttributeError fix from previous iteration)
    if cleaned_results.get("metabolism") and cleaned_results["metabolism"].get("simulations"):
        simulations = cleaned_results["metabolism"]["simulations"]
        all_metabolites_list = []

        for guid, simulation_result in simulations.items():
            metabolites = simulation_result.get("metabolites", [])
            simulator_name = simulation_result.get("simulator_name", f"GUID: {guid}")
            
            if metabolites:
                for metabolite in metabolites:
                    # FIX: Robustly handle metabolite data type
                    if isinstance(metabolite, dict):
                        # If it's a dictionary, we can safely copy it
                        metabolite_record = metabolite.copy()
                    elif isinstance(metabolite, str):
                        # If the API returned a string (e.g., just SMILES), create a dict structure
                        metabolite_record = {"SMILES": metabolite, "Note": "Simple structure returned by API"}
                    else:
                        # Handle other unexpected types gracefully
                        metabolite_record = {"Note": f"Unexpected data type: {type(metabolite)}", "RawValue": str(metabolite)}
                        try:
                            # Log a warning in the Streamlit UI if possible, otherwise print to console
                            st.warning(f"Warning: Unexpected data type found in metabolites list (expected dict, got {type(metabolite)}).")
                        except Exception:
                            print(f"Warning: Unexpected data type found in metabolites list (expected dict, got {type(metabolite)}).")

                    # Add simulator information to each metabolite record
                    metabolite_record['SimulatorName'] = simulator_name
                    metabolite_record['SimulatorGUID'] = guid
                    all_metabolites_list.append(metabolite_record)

        if all_metabolites_list:
            df_meta = pd.DataFrame(all_metabolites_list)
             # Clean up list columns (like Names) for CSV export
            for col in df_meta.columns:
                 # Check if the column contains any lists safely
                try:
                    if df_meta[col].apply(lambda x: isinstance(x, list)).any():
                        df_meta[col] = df_meta[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                except Exception as e:
                     # Handle potential errors during apply (e.g. mixed types that are hard to process)
                    print(f"Error processing column {col} for CSV export: {e}")
                    df_meta[col] = df_meta[col].astype(str)

            
            # Reorder columns to put Simulator info first
            cols = df_meta.columns.tolist()
            if 'SimulatorName' in cols and 'SimulatorGUID' in cols:
                # Use safe removal and insertion
                try:
                    if 'SimulatorGUID' in cols:
                        cols.insert(0, cols.pop(cols.index('SimulatorGUID')))
                    if 'SimulatorName' in cols:
                        cols.insert(0, cols.pop(cols.index('SimulatorName')))
                    df_meta = df_meta[cols]
                except ValueError:
                    # Handle case where pop/index fails unexpectedly
                    pass


            downloads['metabolism'] = df_meta

    return downloads

def render_download_section(results: Dict[str, Any], identifier: str):
    """Render the download section for raw data and reports"""
    st.header("Download Raw Data (CSV)")
    
    # Get downloadable data
    downloads = create_downloadable_data(results)
    
    # Raw Data Downloads (Updated to 4 columns)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Properties Data CSV
        if 'properties' in downloads and not downloads['properties'].empty:
            csv = downloads['properties'].to_csv(index=False)
            st.download_button(
                label="Download Properties",
                data=csv,
                file_name=f"{identifier}_properties.csv",
                mime="text/csv"
            )
        else:
            st.info("No properties data")
    
    with col2:
         # Experimental Data CSV
        if 'experimental' in downloads and not downloads['experimental'].empty:
            # Ensure Value column is string for consistent CSV export
            df_exp = downloads['experimental'].copy()
            
            # Ensure all object columns (including potentially mixed types) are string for CSV export consistency
            for col in df_exp.columns:
                if df_exp[col].dtype == 'object':
                    # Use astype(str) for robust conversion, handling None/NaN gracefully
                    df_exp[col] = df_exp[col].astype(str)

            csv = df_exp.to_csv(index=False)
            st.download_button(
                label="Download Experimental Data",
                data=csv,
                file_name=f"{identifier}_experimental_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No experimental data")

    with col3:
        # Profiling Data CSV
        if 'profiling_results' in downloads and not downloads['profiling_results'].empty:
            csv = downloads['profiling_results'].to_csv(index=False)
            st.download_button(
                label="Download Profiling Results",
                data=csv,
                file_name=f"{identifier}_profiling_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No profiling data")

    with col4:
        # NEW: Metabolism Data CSV
        if 'metabolism' in downloads and not downloads['metabolism'].empty:
            csv = downloads['metabolism'].to_csv(index=False)
            st.download_button(
                label="Download Metabolites",
                data=csv,
                file_name=f"{identifier}_metabolites.csv",
                mime="text/csv"
            )
        else:
            st.info("No metabolism data")