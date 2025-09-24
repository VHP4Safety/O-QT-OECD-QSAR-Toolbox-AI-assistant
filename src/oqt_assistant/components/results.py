# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

import streamlit as st
import pandas as pd
import math
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
        st.info(f"Analysis Context: {results['context']}")
    
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
                properties_list.append({
                    "Property": name,
                    "Value": data["value"],
                    "Unit": data["unit"],
                    "Type": data["type"],
                    "Category": data["family"]
                })
            props_df = pd.DataFrame(properties_list)
            st.dataframe(props_df, use_container_width=True)
        else:
            st.info("No property data available")
    
    with tabs[2]:
        st.subheader("Experimental Data")
        if cleaned_results["experimental_data"]:
            exp_df = pd.DataFrame(cleaned_results["experimental_data"])
            
            # Fix PyArrow serialization issue by ensuring consistent data types
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
                st.session_state.exp_data_page = min(st.session_state.exp_data_page, total_pages)
                st.session_state.exp_data_page = max(st.session_state.exp_data_page, 1)

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
            else:
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
                        # Handle results from profiling/all/{chemId} endpoint
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
                        # Handle results from individual profilers
                        # Create a tab for each profiler
                        profiler_names = list(results.keys())
                        
                        if profiler_names:
                            profiler_result_tabs = st.tabs(profiler_names)
                            
                            for i, name in enumerate(profiler_names):
                                with profiler_result_tabs[i]:
                                    profiler_result = results[name]
                                    
                                    # Show profiler info
                                    st.markdown(f"**Type:** {profiler_result.get('type', 'Unknown')}")
                                    
                                    # Display the result
                                    result_data = profiler_result.get('result', [])
                                    
                                    if isinstance(result_data, list):
                                        if result_data:
                                            st.subheader("Categories")
                                            for category in result_data:
                                                st.markdown(f"- {category}")
                                        else:
                                            st.info("No categories found for this profiler")
                                    else:
                                        st.json(result_data)
                    else:
                        # Unknown result structure
                        st.json(results)
                
                with profiling_tabs[1]:
                    st.subheader("Available Profilers")
                    available_profilers = cleaned_results["profiling"].get("available_profilers", [])
                    display_profilers_list(available_profilers)
            
            # If we only have available profilers
            elif 'available_profilers' in cleaned_results["profiling"]:
                st.subheader("Available Profilers")
                available_profilers = cleaned_results["profiling"]["available_profilers"]
                profilers_list = display_profilers_list(available_profilers)
                
                # Check if we have results from our chemical profiling attempts
                if 'results' in cleaned_results["profiling"]:
                    st.subheader("Profiling Results")
                    
                    # Process the results
                    if isinstance(cleaned_results["profiling"]["results"], dict):
                        # Create tabs for each profiler with results
                        profiler_names = list(cleaned_results["profiling"]["results"].keys())
                        
                        if profiler_names:
                            # Show success message
                            st.success(f"Successfully profiled the chemical with {len(profiler_names)} profilers")
                            
                            # Create profiler tabs
                            profiler_tabs = st.tabs(profiler_names)
                            
                            # Display results for each profiler
                            for i, profiler_name in enumerate(profiler_names):
                                with profiler_tabs[i]:
                                    profiler_data = cleaned_results["profiling"]["results"][profiler_name]
                                    
                                    # Show profiler type and GUID
                                    st.markdown(f"**Type:** {profiler_data.get('type', 'Unknown')}")
                                    st.markdown(f"**GUID:** {profiler_data.get('guid', 'Unknown')}")
                                    
                                    # Process the results
                                    result = profiler_data.get('result', [])
                                    
                                    if isinstance(result, list):
                                        # Display as a list of categories
                                        st.subheader("Categories")
                                        for category in result:
                                            st.markdown(f"- {category}")
                                    elif isinstance(result, dict):
                                        # Display as a table
                                        st.subheader("Categories")
                                        result_df = pd.DataFrame([result])
                                        st.dataframe(result_df, use_container_width=True)
                                    else:
                                        # Show raw result
                                        st.json(result)
                        else:
                            st.warning("No profiling results available")
                    elif isinstance(cleaned_results["profiling"]["results"], list):
                        # Display results from profiling/all/{chemId} endpoint
                        results_list = []
                        
                        for profiler_result in cleaned_results["profiling"]["results"]:
                            if isinstance(profiler_result, dict):
                                profile_name = profiler_result.get("ProfilerName", "Unknown")
                                profile_type = profiler_result.get("ProfilerType", "Unknown")
                                categories = profiler_result.get("Categories", [])
                                
                                if categories:
                                    for category in categories:
                                        results_list.append({
                                            "Profiler": profile_name,
                                            "Type": profile_type,
                                            "Category": category
                                        })
                        
                        if results_list:
                            # Show as dataframe
                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.warning("No categories found in profiling results")
                    else:
                        st.json(cleaned_results["profiling"]["results"])
                
                # Add explanation of what profilers do
                st.markdown("""
                ### What are chemical profilers?
                
                Chemical profilers are tools that analyze a chemical's structure to identify specific features or structural alerts 
                that may be associated with particular toxicological effects or mechanisms of action. They help categorize chemicals 
                and identify potential concerns based on structural similarity to known problematic chemicals.
                """)
                
                # Show the list of available profilers if we don't have results for all of them
                if st.session_state.get("show_all_profilers", False) or 'status' in cleaned_results["profiling"] and cleaned_results["profiling"]["status"] != "Success":
                    st.subheader("All Available Profilers")
                    st.write("These profilers are available in the QSAR Toolbox but may not have been applied to this chemical:")
                    if profilers_list:
                        profilers_df = pd.DataFrame(profilers_list)
                        st.dataframe(profilers_df, use_container_width=True)
                
                    # Add a button to toggle showing all profilers
                    if st.button("Hide Available Profilers"):
                        st.session_state["show_all_profilers"] = False
                        st.experimental_rerun()
                else:
                    # Add a button to toggle showing all profilers
                    if st.button("Show All Available Profilers"):
                        st.session_state["show_all_profilers"] = True
                        st.experimental_rerun()
            
            # Handle our older structure with results/profilers split
            elif 'results' in cleaned_results["profiling"] or 'profilers' in cleaned_results["profiling"]:
                profiling_tabs = st.tabs(["Profiling Results", "Available Profilers"])
                
                # Tab for chemical-specific profiling results
                with profiling_tabs[0]:
                    if cleaned_results["profiling"].get("results"):
                        # Convert chemical-specific profiling results to DataFrame
                        results_list = []
                        for category, profile_data in cleaned_results["profiling"]["results"].items():
                            if isinstance(profile_data, dict):
                                for result in profile_data.values():
                                    if isinstance(result, dict):
                                        results_list.append({
                                            "Category": category,
                                            "Name": result.get("Name", "Unknown"),
                                            "Value": result.get("Value", "N/A"),
                                            "Description": result.get("Description", ""),
                                            "Applicability": result.get("Applicability", "Unknown")
                                        })
                        
                        if results_list:
                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.info("No specific profiling results available for this chemical")
                    else:
                        st.info("No specific profiling results available for this chemical")
                
                # Tab for available profilers
                with profiling_tabs[1]:
                    if cleaned_results["profiling"].get("profilers"):
                        # Convert profilers to DataFrame
                        profilers_list = []
                        for name, data in cleaned_results["profiling"]["profilers"].items():
                            profilers_list.append({
                                "Profiler": name,
                                "Type": data.get("type", "Unknown"),
                                "ID": data.get("id", "Unknown"),
                                "Description": data.get("description", "")
                            })
                        
                        if profilers_list:
                            profilers_df = pd.DataFrame(profilers_list)
                            st.dataframe(profilers_df, use_container_width=True)
                        else:
                            st.info("No profiler information available")
                    else:
                        st.info("No profiler information available")
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
            properties_list.append({
                "Property": name,
                "Value": data["value"],
                "Unit": data["unit"],
                "Type": data["type"],
                "Category": data["family"]
            })
        downloads['properties'] = pd.DataFrame(properties_list)
    
    # Experimental Data
    if cleaned_results["experimental_data"]:
        downloads['experimental'] = pd.DataFrame(cleaned_results["experimental_data"])
    
    # Profiling Data
    if cleaned_results["profiling"]:
        # Handle the new nested structure
        if isinstance(cleaned_results["profiling"], dict) and 'results' in cleaned_results["profiling"]:
            # Chemical-specific profiling results
            if cleaned_results["profiling"].get("results"):
                results_list = []
                for category, profile_data in cleaned_results["profiling"]["results"].items():
                    if isinstance(profile_data, dict):
                        for result in profile_data.values():
                            if isinstance(result, dict):
                                results_list.append({
                                    "Category": category,
                                    "Name": result.get("Name", "Unknown"),
                                    "Value": result.get("Value", "N/A"),
                                    "Description": result.get("Description", ""),
                                    "Applicability": result.get("Applicability", "Unknown")
                                })
                
                if results_list:
                    downloads['profiling_results'] = pd.DataFrame(results_list)
            
            # Available profilers
            if cleaned_results["profiling"].get("profilers"):
                profilers_list = []
                for name, data in cleaned_results["profiling"]["profilers"].items():
                    profilers_list.append({
                        "Profiler": name,
                        "Type": data.get("type", "Unknown"),
                        "ID": data.get("id", "Unknown")
                    })
                
                if profilers_list:
                    downloads['profiling'] = pd.DataFrame(profilers_list)
        else:
            # Fallback for old data structure
            profilers_list = []
            for name, data in cleaned_results["profiling"].items():
                if isinstance(data, dict):
                    profilers_list.append({
                        "Profiler": name,
                        "Type": data.get("type", "Unknown"),
                        "ID": data.get("id", "Unknown")
                    })
            if profilers_list:
                downloads['profiling'] = pd.DataFrame(profilers_list)
    
    return downloads

def render_download_section(results: Dict[str, Any], identifier: str):
    """Render the download section for raw data and reports"""
    st.header("Download Results")
    
    # Clean the data
    cleaned_results = clean_response_data(results)
    
    # Get downloadable data
    downloads = create_downloadable_data(results)
    
    # Raw Data Downloads
    st.subheader("Raw Data Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete JSON download
        json_str = pd.json_normalize(cleaned_results).to_json(orient='records')
        st.download_button(
            label="Download Complete Results (JSON)",
            data=json_str,
            file_name=f"{identifier}_complete_analysis.json",
            mime="application/json"
        )
    
    with col2:
        # Properties Data CSV
        if 'properties' in downloads:
            csv = downloads['properties'].to_csv(index=False)
            st.download_button(
                label="Download Properties (CSV)",
                data=csv,
                file_name=f"{identifier}_properties.csv",
                mime="text/csv"
            )
        else:
            st.info("No properties data")
    
    with col3:
        # Profiling Data CSV
        if 'profiling_results' in downloads:
            csv = downloads['profiling_results'].to_csv(index=False)
            st.download_button(
                label="Download Profiling Results (CSV)",
                data=csv,
                file_name=f"{identifier}_profiling_results.csv",
                mime="text/csv"
            )
        elif 'profiling' in downloads:
            csv = downloads['profiling'].to_csv(index=False)
            st.download_button(
                label="Download Profiling Data (CSV)",
                data=csv,
                file_name=f"{identifier}_profiling_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No profiling data")
