# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

import streamlit as st
import pandas as pd
import math
import json
from typing import Dict, Any, List
from ..utils.data_formatter import clean_response_data

def display_profilers_list(available_profilers):
    """Helper method to display a list of profilers"""
    profilers_list = []
    
    if isinstance(available_profilers, list):
        # If it's already a list, use it directly
        for profiler in available_profilers:
            profilers_list.append({
                "Profiler Name": profiler.get('name', 'Unknown'),
                "Type": profiler.get('type', 'Unknown'),
                "Status": profiler.get('status', 'Unknown'),
                "Description": profiler.get('description', 'A chemical profiler that categorizes chemicals based on structural features')
            })
    elif isinstance(available_profilers, dict):
        # If it's a dict, convert to list
        for key, profiler in available_profilers.items():
            profilers_list.append({
                "Profiler Name": profiler.get('name', 'Unknown'),
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

def render_results_section(results: Dict[str, Any], identifier_display: str):
    """Render the analysis results in an interactive dashboard"""
    st.header(f"Analysis Results for {identifier_display}")
    
    # Clean and format the data
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
        "Profiling"
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
    
    with tabs[2]:
        st.subheader("Experimental Data")

        # Add explanation of data extraction (Addressing Reviewer #1)
        with st.expander("About the Data Extracted (Reviewer #1)"):
            st.markdown("""
            This table displays the raw experimental data retrieved from the OECD QSAR Toolbox for the target chemical.

            **Data Schema:**
            - **Endpoint:** The specific property or effect being measured (e.g., Water Solubility, BCF, Ames test).
            - **Value:** The numerical result or qualitative outcome.
            - **Unit:** The unit of measurement (if applicable).
            - **DataType:** Indicates the source of the data. **"Measured value."** indicates experimental data. Other values (e.g., estimates) may also be present.
            - **Reliability:** Indicates the quality or reliability of the study (e.g., Klimisch score).
            - **Reference/TestGuid:** Information about the source study or the testing guideline followed (e.g., OECD TG).

            **AI Analysis Focus:** The AI agents are programmed to prioritize data where `DataType` is "Measured value." In the synthesized report, this data is strictly labeled as **"Experimental (Toolbox)"** to ensure clear provenance.
            """)

        if cleaned_results["experimental_data"]:
            exp_df = pd.DataFrame(cleaned_results["experimental_data"])
            
            # Fix PyArrow serialization issue by ensuring consistent data types
            if not exp_df.empty:
                if 'Value' in exp_df.columns:
                    # Convert all values in Value column to strings to avoid mixed type issues
                    exp_df['Value'] = exp_df['Value'].astype(str)
                
                # Convert any remaining numeric columns that might have mixed types
                for col in exp_df.columns:
                    if exp_df[col].dtype == 'object':
                        try:
                            # Try to convert to numeric, if fails keep as string
                            exp_df[col] = pd.to_numeric(exp_df[col], errors='ignore')
                            # If still object type, convert to string to ensure consistency
                            if exp_df[col].dtype == 'object':
                                exp_df[col] = exp_df[col].astype(str)
                        except Exception:
                            # Fallback: convert to string
                            exp_df[col] = exp_df[col].astype(str)
            
            # --- Pagination Logic ---
            items_per_page = 15  # Number of rows per page
            total_items = len(exp_df)

            if total_items > items_per_page:
                total_pages = math.ceil(total_items / items_per_page)

                # Use session state to remember the current page
                if 'exp_data_page' not in st.session_state:
                    st.session_state.exp_data_page = 1

                # Ensure page number stays within bounds if data changes
                if total_pages > 0:
                    st.session_state.exp_data_page = min(st.session_state.exp_data_page, total_pages)
                    st.session_state.exp_data_page = max(st.session_state.exp_data_page, 1)
                else:
                    st.session_state.exp_data_page = 1


                # Display pagination controls: Previous button, Page indicator, Next button
                col1, col2, col3 = st.columns([2, 1, 2])  # Adjust column ratios as needed

                with col1:
                    # Disable button if on the first page
                    if st.button("⬅️ Previous", key="exp_prev_page", disabled=(st.session_state.exp_data_page <= 1)):
                        st.session_state.exp_data_page -= 1
                        st.rerun()

                with col2:
                    # Display current page and total pages
                    st.write(f"Page {st.session_state.exp_data_page} of {total_pages}")

                with col3:
                    # Disable button if on the last page
                    if st.button("Next ➡️", key="exp_next_page", disabled=(st.session_state.exp_data_page >= total_pages)):
                        st.session_state.exp_data_page += 1
                        st.rerun()

                # Calculate slice indices for the current page
                start_idx = (st.session_state.exp_data_page - 1) * items_per_page
                end_idx = start_idx + items_per_page

                # Display the sliced DataFrame for the current page
                st.dataframe(exp_df.iloc[start_idx:end_idx], use_container_width=True)
            else:
                # If data fits on one page, display the whole DataFrame
                st.dataframe(exp_df, use_container_width=True)
            # --- End Pagination Logic ---
        else:
            st.info("No experimental data available for this chemical")
    
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
                        results_list = []
                        for profiler_result in results:
                            if isinstance(profiler_result, dict):
                                profiler_name = profiler_result.get("ProfilerName", "Unknown")
                                profiler_type = profiler_result.get("ProfilerType", "Unknown")
                                categories = profiler_result.get("Categories", [])
                                
                                if categories:
                                    for category in categories:
                                        results_list.append({
                                            "Profiler": profiler_name,
                                            "Type": profiler_type,
                                            "Category": category
                                        })
                                else:
                                    # No categories found
                                    results_list.append({
                                        "Profiler": profiler_name,
                                        "Type": profiler_type,
                                        "Category": "No categories found"
                                    })
                        
                        if results_list:
                            # Show as dataframe
                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.warning("No categories found in profiling results")
                    
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

def create_downloadable_data(results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Create downloadable DataFrames from results"""
    cleaned_results = clean_response_data(results)
    downloads = {}
    
    # Properties
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
    
    # Experimental Data
    if cleaned_results["experimental_data"]:
        downloads['experimental'] = pd.DataFrame(cleaned_results["experimental_data"])
    
    # Profiling Data (Updated to handle the standardized list format)
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

            elif isinstance(profiling_results, list):
                 # Handle results from profiling/all/{chemId} endpoint
                 for profiler_result in profiling_results:
                     if isinstance(profiler_result, dict):
                         profiler_name = profiler_result.get("ProfilerName", "Unknown")
                         profiler_type = profiler_result.get("ProfilerType", "Unknown")
                         categories = profiler_result.get("Categories", [])

                         if categories:
                             for category in categories:
                                 results_list.append({
                                     "Profiler": profiler_name,
                                     "Type": profiler_type,
                                     "Category": category
                                 })
                         else:
                             results_list.append({
                                 "Profiler": profiler_name,
                                 "Type": profiler_type,
                                 "Category": "No categories found"
                             })

        if results_list:
            downloads['profiling_results'] = pd.DataFrame(results_list)
    
    return downloads

def render_download_section(results: Dict[str, Any], identifier: str):
    """Render the download section for raw data and reports"""
    st.header("Download Raw Data (CSV)")
    
    # Clean the data
    # cleaned_results = clean_response_data(results) # Not needed here if only using downloads dict
    
    # Get downloadable data
    downloads = create_downloadable_data(results)
    
    # Raw Data Downloads
    col1, col2, col3 = st.columns(3)
    
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
            if 'Value' in df_exp.columns:
                df_exp['Value'] = df_exp['Value'].astype(str)

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