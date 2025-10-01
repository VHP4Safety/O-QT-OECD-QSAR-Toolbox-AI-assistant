# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

def _smiles_to_molblock(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

def render_smiles_3d(smiles: str, height: int = 420, width: int = 700):
    try:
        import py3Dmol
    except ImportError:
        st.warning("py3Dmol not available. Install it with: pip install py3Dmol")
        return
    
    mb = _smiles_to_molblock(smiles)
    
    # Create the viewer
    if width == 0:
        width = 700  # Default width if 0 is passed
    
    view = py3Dmol.view(width=width, height=height)
    
    if mb:
        view.addModel(mb, 'mol')
    else:
        # Fallback: let 3Dmol try to depict directly from SMILES
        view.addModel(smiles, 'smi')
    
    view.setStyle({'stick': {}})
    view.zoomTo()
    
    # Generate HTML and embed in Streamlit
    try:
        html = view._make_html()
    except AttributeError:
        html = view._repr_html_()
    
    # Use scrolling=True to ensure the iframe renders properly
    components.html(html, height=height, scrolling=True)
