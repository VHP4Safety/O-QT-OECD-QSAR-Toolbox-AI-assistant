# oqt_assistant/components/results.py
# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

# NEW: Import 3D rendering utility
from oqt_assistant.utils.structure_3d import render_smiles_3d

def _render_static_depiction(smiles: str, width: int = 520, height: int = 360):
    """Render static 2D molecular depiction using RDKit."""
    import streamlit as st
    import io
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            img = Draw.MolToImage(mol, size=(width, height))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="2D depiction (fallback)")
            # Store PNG for PDF use (picked up by pdf generator later if you wire it)
            st.session_state['oqt_static_depiction_png'] = buf.getvalue()
    except Exception:
        pass

def render_results_section(results: Dict[str, Any], identifier: str):
    """Render the results section with tabs for different data types."""
    st.header(f"Analysis Results for: {identifier}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧪 Chemical Data",
        "📊 Properties (Calculated)",
        "📈 Experimental Data (Measured)",
        "🔬 Profiling & Reactivity",
        "🧬 Metabolism (Simulated)",
        "🔮 QSAR Models"
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

    # --- Tab 6: QSAR Models ---
    with tab6:
        _render_qsar_predictions(results.get('qsar_models', {}))


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
        
        # NEW: Display IUCLID IDs if available (QPRF §2.3)
        iuclid_data = basic_info.get('iuclid', {})
        if iuclid_data and iuclid_data.get('entity_ids'):
            st.markdown("---")
            st.markdown("**🔗 IUCLID Regulatory Identifiers (QPRF §2.3)**")
            for entity in iuclid_data['entity_ids']:
                entity_id = entity.get('entity_id', 'N/A')
                entity_name = entity.get('name', 'N/A')
                echa_url = iuclid_data.get('echa_url', '')
                
                if echa_url:
                    st.markdown(f"- **IUCLID ID:** [{entity_id}]({echa_url}) - {entity_name}")
                else:
                    st.markdown(f"- **IUCLID ID:** {entity_id} - {entity_name}")
        
        # NEW: Display canonical SMILES and connectivity if available (QPRF §5.1)
        if basic_info.get('canonical_smiles'):
            with st.expander("📋 Normalized Structure (QPRF §5.1)"):
                st.markdown(f"**Canonical SMILES:** `{basic_info.get('canonical_smiles')}`")
                if basic_info.get('connectivity'):
                    st.markdown(f"**Connectivity:** `{basic_info.get('connectivity')}`")
                if basic_info.get('stereochemistry_note'):
                    st.info(basic_info.get('stereochemistry_note'))
        
        # --- New: 3D viewer with graceful fallback ---
        with st.expander("3D Preview (optional)"):
            smiles = basic_info.get('Smiles') or ""
            if smiles:
                try:
                    render_smiles_3d(smiles)
                except Exception as e:
                    st.warning(f"3D preview unavailable: {e}")
            else:
                st.info("SMILES not available for this record.")
        
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
        
        st.dataframe(df_props, width="stretch", hide_index=True)
    except Exception as e:
        st.warning(f"Could not format properties data: {e}")
        st.json(properties)

# UPDATED: Enhanced experimental data rendering with metadata columns and filters
def _render_experimental_data_interactive(experimental_data: List[Dict[str, Any]]):
    """Render experimental data in an interactive table with metadata columns, filters, and selection."""
    
    # Filter out system notes (like truncation messages) from the main data display
    actual_data_records = [r for r in experimental_data if isinstance(r, dict) and r.get('DataType') != 'SystemNote']

    if not actual_data_records:
        st.info("No experimental data retrieved or available after filtering.")
        return

    st.subheader(f"Experimental Data Records ({len(actual_data_records)})")

    # --- Data Preparation ---
    try:
        df = pd.DataFrame(actual_data_records)

        # Build a friendlier endpoint column
        def _endpoint_from_row(r):
            v = r.get('Endpoint')
            if v in (None, "", "n/a", "N/A"):
                v = r.get('Family') or None
            if not v:
                pm = r.get('Parsed_Metadata') or {}
                if isinstance(pm, dict):
                    v = pm.get('Effect') or pm.get('Measurement') or pm.get('Endpoint') or None
            return v or "—"

        df['Endpoint_Display'] = [ _endpoint_from_row(rec) for rec in actual_data_records ]

        # Columns to show (Parsed_Metadata is hidden but available for the details panel)
        columns_to_display_map = {
            "Publication_Year": "Year",
            "Endpoint_Display": "Endpoint",   # ⬅ new
            "Value": "Value",
            "Unit": "Unit",
            "DataType": "Type",
            "Reliability": "Reliability",
            "Reference": "Reference",
            "Parsed_Metadata": "Parsed_Metadata",
        }

        existing = [c for c in columns_to_display_map.keys() if c in df.columns]
        df_display = df[existing].copy()

        # Safe (non‑inplace) rename and null‑to‑blank sanitization for the UI
        df_display = df_display.rename(columns=columns_to_display_map, errors="ignore")
        
        # Handle the 'Year' column for proper sorting
        if 'Year' in df_display.columns:
            try:
                # Normalize empties before conversion
                df_display['Year'] = df_display['Year'].replace({"": None, "N/A": None, "n/a": None})
                df_display['Year'] = pd.to_numeric(df_display['Year'], errors='coerce').astype('Int64')
            except (TypeError, ValueError) as e:
                logger.warning(f"Year column conversion failed: {e}")
                df_display['Year'] = pd.to_numeric(df_display['Year'], errors='coerce')
            try:
                df_display.sort_values(by=['Year', 'Endpoint'], ascending=[False, True],
                                       inplace=True, na_position='last')
            except Exception as e:
                st.warning(f"Could not apply default sorting: {e}")
                logger.warning(f"Sorting failed: {e}. DataFrame dtypes: {df_display.dtypes}")

        # Fill NaN values with empty strings for all columns EXCEPT 'Year' (which uses Int64 and needs to keep NaN)
        for col in df_display.columns:
            if col != 'Year':
                df_display[col] = df_display[col].fillna("")

    except Exception as e:
        st.error(f"Error preparing data table: {e}")
        st.json(experimental_data)
        return

    # --- Display Table (Interactive) ---
    st.info("Data sorted by Year (newest first). Click a row to view detailed study metadata below.")

    column_config = {}
    if "Parsed_Metadata" in df_display.columns:
        column_config["Parsed_Metadata"] = None  # hide from the grid; used for details

    selection = st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config
    )

    st.subheader("Study Metadata Details")
    try:
        # Streamlit returns a dict‑like selection object in recent versions. Be defensive:
        sel_rows = getattr(selection, "get", lambda *_: {})("selection", {}).get("rows")
        if sel_rows:
            pos = sel_rows[0]
            row = df_display.iloc[pos]
            meta = row.get("Parsed_Metadata", {})
            if isinstance(meta, dict) and meta:
                st.json(meta)
            else:
                st.warning("No structured metadata available for the selected record.")
        else:
            st.info("Select a record above to see details.")
    except Exception as e:
        st.warning(f"Could not display metadata details: {e}")


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
            st.dataframe(df_av, width="stretch", hide_index=True)

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
                    st.dataframe(df[preferred] if preferred else df, width="stretch", hide_index=True)
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
                                st.dataframe(metabolites_df[available_cols], width="stretch", hide_index=True)
                        else:
                            st.dataframe(metabolites_df, width="stretch", hide_index=True)
                    
                    except Exception as e:
                        st.warning(f"Could not display metabolites for {sim_name} in table format: {e}")
                        st.json(metabolites)

def _render_qsar_predictions(qsar_data: Dict[str, Any]):
    """Render QSAR model predictions grouped by applicability domain."""
    st.subheader("QSAR Model Predictions (Applicability Domain)")

    if not isinstance(qsar_data, dict) or not qsar_data:
        st.info("QSAR predictions were not available.")
        return

    processed = qsar_data.get("processed", {}) if isinstance(qsar_data, dict) else {}
    if not processed:
        st.info("QSAR predictions were not available.")
        return

    summary = processed.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Models", summary.get("total", 0))
    col2.metric("In Domain", summary.get("in_domain", 0))
    col3.metric("Not Applicable", summary.get("not_applicable", 0))
    col4.metric("Out of Domain", summary.get("out_of_domain", 0))

    in_domain_records = processed.get("in_domain", [])
    if in_domain_records:
        st.markdown("#### In-Domain Predictions")
        try:
            rows = []
            for rec in in_domain_records:
                rows.append({
                    "Model": rec.get("caption"),
                    "Category": rec.get("top_category") or rec.get("requested_position"),
                    "Value": rec.get("value"),
                    "Unit": rec.get("unit"),
                    "Runtime (s)": rec.get("runtime_seconds"),
                    "Donator": rec.get("donator"),
                })
            df_in_domain = pd.DataFrame(rows)
            df_in_domain.sort_values(by=["Runtime (s)"], inplace=True)
            st.dataframe(df_in_domain, width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Could not format in-domain predictions: {e}")
            st.json(in_domain_records)
    else:
        st.info("No QSAR models reported the chemical within their applicability domain.")

    not_applicable = processed.get("not_applicable", [])
    if not_applicable:
        with st.expander(f"Not Applicable ({len(not_applicable)})"):
            st.json(not_applicable)

    out_of_domain = processed.get("out_of_domain", [])
    if out_of_domain:
        with st.expander(f"Out of Domain ({len(out_of_domain)})"):
            st.json(out_of_domain)

    errors = processed.get("errors", [])
    if errors:
        with st.expander(f"Errors ({len(errors)})"):
            st.json(errors)

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
