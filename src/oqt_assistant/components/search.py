# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

import streamlit as st
from typing import Tuple, List

# New imports
from oqt_assistant.utils.qsar_api import QSARToolboxAPI, SearchOptions  # already in project
from oqt_assistant.utils.structure_3d import render_smiles_3d # NEW

# Define predefined regulatory endpoints for guided input
REGULATORY_ENDPOINTS = [
    "Ecotoxicity (Aquatic/Terrestrial)",
    "Environmental Fate (Persistence/Bioaccumulation)",
    "Genotoxicity/Mutagenicity",
    "Carcinogenicity",
    "Reproductive/Developmental Toxicity",
    "Neurotoxicity",
    "Endocrine Disruption",
    "Skin/Eye Irritation/Corrosion",
    "Sensitization (Skin/Respiratory)"
]

def initialize_endpoint_state():
    """Initialize session state for endpoint checkboxes."""
    if 'endpoints' not in st.session_state:
        st.session_state.endpoints = {endpoint: False for endpoint in REGULATORY_ENDPOINTS}
    if 'select_all_endpoints' not in st.session_state:
        st.session_state.select_all_endpoints = False

def handle_select_all():
    """Handles the 'Select All' checkbox logic."""
    # This function is called when the "Select All" checkbox state changes
    select_all = st.session_state.select_all_endpoints
    for endpoint in REGULATORY_ENDPOINTS:
        # Update the state for the individual checkboxes and the dictionary
        st.session_state[f"check_{endpoint}"] = select_all
        st.session_state.endpoints[endpoint] = select_all

def handle_individual_checkbox(endpoint):
    """Handles individual checkbox logic to update the dictionary."""
    # This function is called when an individual checkbox state changes
    is_checked = st.session_state[f"check_{endpoint}"]
    st.session_state.endpoints[endpoint] = is_checked
    
    # If an item is unchecked, uncheck "Select All"
    if not is_checked:
        st.session_state.select_all_endpoints = False


def render_search_section() -> Tuple[str, str, str]:
    """Render the chemical search section of the UI (now in the main area)
    
    Returns:
        Tuple containing (identifier, search_type, context)
    """
    initialize_endpoint_state()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Chemical Identification")
        # Search type selection
        search_type = st.radio(
            "Search By",
            options=['name', 'smiles'],
            format_func=lambda x: "Chemical Name" if x == 'name' else "SMILES Notation",
            help="Select how you want to identify the chemical."
        )
        
        # Dynamic input field based on search type
        if search_type == 'name':
            identifier = st.text_input(
                "Chemical Name (Exact Match)",
                placeholder="e.g., 1,1-diethoxyheptane",
                help="Enter the exact name of the chemical. The system currently uses exact matching via the QSAR Toolbox API. Case study chemical '1,1-diethoxyheptane' is provided as an example."
            )
        else:
            identifier = st.text_input(
                "SMILES Notation",
                placeholder="e.g., CCCCCCC(OCC)OCC",
                help="Enter the SMILES notation of the chemical."
            )

        with st.expander("3D Preview (optional)"):
            if st.button("Preview structure"):
                try:
                    # If user typed a SMILES, render it here; otherwise guide them to run the search.
                    if search_type == 'smiles' and identifier:
                        render_smiles_3d(identifier)
                    else:
                        st.info("Enter a SMILES above to preview here, or run the analysis and open the 3D preview under ‘Chemical Data’.")
                except Exception as e:
                    st.warning(f"3D preview unavailable: {e}")

    with col2:
        st.subheader("2. Analysis Context")
        st.markdown("Select relevant endpoints and add any context that should steer the analysis.")

        # --- Guided Context (Checkboxes) ---
        expander_title = "Select Regulatory Endpoints (Optional)"
        # Calculate selected count dynamically based on the current state
        selected_count = sum(st.session_state.endpoints.values())
        
        if selected_count > 0:
            expander_title += f" ({selected_count} selected)"

        with st.expander(expander_title, expanded=False):
            # Use the on_change callback for "Select All"
            st.checkbox("Select All Endpoints", key="select_all_endpoints", on_change=handle_select_all)
            
            for endpoint in REGULATORY_ENDPOINTS:
                # Initialize the checkbox state from the session state if it exists, otherwise default to False
                initial_value = st.session_state.endpoints.get(endpoint, False)
                # Use dynamic keys and on_change callback for individual checkboxes
                st.checkbox(endpoint, key=f"check_{endpoint}", value=initial_value, on_change=handle_individual_checkbox, args=(endpoint,))

        # --- Read-Across Potential ---
        read_across_help = """
        **Read-Across Potential (Definition):** The likelihood that the toxicity profile of the target chemical can be accurately predicted by using existing data from structurally or mechanistically similar chemicals (analogues).
        Checking this box directs the AI agents to prioritize identifying data gaps and developing a detailed strategy for finding suitable analogues.
        """
        prioritize_read_across = st.checkbox(
            "Prioritize Read-Across Strategy Development",
            help=read_across_help
        )

        # --- Custom Context (Free Text) ---
        custom_context = st.text_area(
            "Custom Context/Concerns (Optional)",
            placeholder="e.g., Specific concern about neurotoxicity related to Parkinson's disease, or focus on environmental impact in agricultural settings.",
            help="Describe any specific interests, concerns, or regulatory frameworks not covered by the checkboxes."
        )
        
        # --- Combine Context ---
        context_parts = []
        # Recalculate selected endpoints based on the final state after user interaction
        selected_endpoints = [e for e, selected in st.session_state.endpoints.items() if selected]

        if selected_endpoints:
            context_parts.append(f"Focus the analysis on the following regulatory endpoints: {', '.join(selected_endpoints)}.")

        if prioritize_read_across:
            context_parts.append("Critically evaluate the data gaps and develop a detailed read-across strategy.")

        if custom_context:
            context_parts.append(f"Address these specific user concerns: {custom_context}")

        final_context = " ".join(context_parts)

        if not final_context:
            final_context = "General chemical hazard assessment and identification of key properties."

        # Display the final context that will be used for the analysis
        with st.expander("View Final Analysis Context", expanded=False):
            st.info(final_context)

    
    return identifier, search_type, final_context
