# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

import streamlit as st
from typing import Tuple

def render_search_section() -> Tuple[str, str, str]:
    """Render the chemical search section of the UI
    
    Returns:
        Tuple containing (identifier, search_type, context)
    """
    st.sidebar.header("Chemical Analysis Input")
    
    # Search type selection
    search_type = st.sidebar.radio(
        "Search Type",
        options=['name', 'smiles'],
        format_func=lambda x: "Chemical Name" if x == 'name' else "SMILES",
        help="Select how you want to search for the chemical"
    )
    
    # Dynamic input field based on search type
    if search_type == 'name':
        identifier = st.sidebar.text_input(
            "Chemical Name",
            help="Enter the name of the chemical you want to analyze (e.g., chlorpyrifos)"
        )
    else:
        identifier = st.sidebar.text_input(
            "SMILES",
            help="Enter the SMILES notation of the chemical (e.g., CC(=O)OC1=CC=CC=C1C(=O)O for Aspirin)"
        )
    
    context = st.sidebar.text_area(
        "Analysis Context",
        help="Describe your specific interest or concern (e.g., neurotoxicity, environmental impact)"
    )
    
    return identifier, search_type, context
