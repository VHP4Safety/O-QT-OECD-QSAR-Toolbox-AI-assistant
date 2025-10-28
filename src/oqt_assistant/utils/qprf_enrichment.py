# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
QPRF/RAAF Data Enrichment Module

This module enriches chemical data with QPRF v2.0 and RAAF-compliant metadata
from the OECD QSAR Toolbox WebAPI v6.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class QPRFEnricher:
    """Enriches chemical data with QPRF/RAAF metadata"""
    
    def __init__(self, api_client):
        """Initialize with QSAR Toolbox API client"""
        self.api = api_client
        self._software_info_cache = None
    
    def get_software_info(self) -> Dict[str, Any]:
        """Get software version information (QPRF §3.2)
        
        Cached to avoid repeated API calls
        """
        if self._software_info_cache is None:
            toolbox_version = self.api.get_toolbox_version()
            webapi_version = self.api.get_webapi_version()
            
            self._software_info_cache = {
                "toolbox_version": toolbox_version.get("Version", "Unknown"),
                "toolbox_name": "OECD QSAR Toolbox",
                "webapi_version": webapi_version.get("Version", "Unknown"),
                "webapi_name": "QSAR Toolbox WebAPI",
                "timestamp": datetime.now().isoformat()
            }
        
        return self._software_info_cache
    
    def enrich_substance_identity(self, chemical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich substance identity with IUCLID IDs and normalized structures
        
        Maps to QPRF §2.1-2.6 (Substance identity)
        """
        enriched = chemical_data.copy()
        
        # Get CAS, EC, name from chemical data
        cas = chemical_data.get("Cas") or chemical_data.get("CAS")
        ec = chemical_data.get("ECNumber") or chemical_data.get("EC")
        name = chemical_data.get("Name") or chemical_data.get("name")
        smiles = chemical_data.get("Smiles") or chemical_data.get("SMILES")
        
        # Initialize IUCLID data structure
        iuclid_data = {
            "entity_ids": [],
            "echa_url": None,
            "source_type": None
        }
        
        # NEW: Try multiple IUCLID lookup strategies with robust fallbacks
        # Try CAS (sanitize: API expects digits only, no hyphens in path)
        if cas:
            cas_digits = None
            try:
                # Strip non-digits and convert to sanitized string for path param
                cas_digits = re.sub(r"\D+", "", str(cas or ""))
                if cas_digits:
                    iuclid_result = self.api.search_iuclid_by_cas(int(cas_digits))
                    if iuclid_result and isinstance(iuclid_result, dict):
                        iuclid_data = self._extract_iuclid_data(iuclid_result)
                    elif not isinstance(iuclid_result, dict):
                        logger.warning(f"IUCLID CAS response not dict; skipping enrichment for CAS {cas}")
            except Exception as e:
                logger.warning(f"IUCLID CAS lookup failed for {cas} → {cas_digits}: {e}")
        
        # Try EC number if CAS failed
        if not iuclid_data["entity_ids"] and ec:
            try:
                iuclid_result = self.api.search_iuclid_by_ec(ec)
                if iuclid_result and isinstance(iuclid_result, dict):
                    iuclid_data = self._extract_iuclid_data(iuclid_result)
            except Exception as e:
                logger.warning(f"Could not retrieve IUCLID data by EC: {e}")
        
        # Try name if both CAS and EC failed
        if not iuclid_data["entity_ids"] and name:
            try:
                iuclid_result = self.api.search_iuclid_by_name(name)
                if iuclid_result and isinstance(iuclid_result, dict):
                    iuclid_data = self._extract_iuclid_data(iuclid_result)
            except Exception as e:
                logger.warning(f"Could not retrieve IUCLID data by name: {e}")
        
        # Normalize SMILES
        if smiles:
            try:
                canonical_smiles = self.api.canonicalize_smiles(smiles)
                connectivity = self.api.get_connectivity(smiles)
                
                enriched["canonical_smiles"] = canonical_smiles
                enriched["connectivity"] = connectivity
                enriched["stereochemistry_note"] = "Stereochemistry preserved in canonical SMILES" if canonical_smiles != smiles else "No stereochemistry"
            except Exception as e:
                logger.warning(f"Could not normalize SMILES: {e}")
        
        # Add IUCLID data
        enriched["iuclid"] = iuclid_data
        
        return enriched
    
    def _extract_iuclid_data(self, iuclid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract IUCLID data from API response (with robust type checking)"""
        # NEW: Handle both dict and list responses
        if isinstance(iuclid_result, list):
            iuclid_result = iuclid_result[0] if iuclid_result else {}
        
        if not isinstance(iuclid_result, dict):
            logger.warning(f"IUCLID result is not a dict: {type(iuclid_result)}")
            return {"entity_ids": [], "echa_url": None, "source_type": None}
        
        chemical = iuclid_result.get("Chemical", {})
        if not isinstance(chemical, dict):
            logger.warning(f"Chemical data in IUCLID result is not a dict: {type(chemical)}")
            return {"entity_ids": [], "echa_url": None, "source_type": None}
        
        iuclid_ids = chemical.get("IUCLIDIds", [])
        
        entity_ids = []
        for iuclid_id in iuclid_ids:
            entity_ids.append({
                "entity_id": iuclid_id.get("IUCLIDEntityId"),
                "name": iuclid_id.get("IUCLIDName"),
                "type": iuclid_id.get("Type"),
                "source_id": iuclid_id.get("SourceId")
            })
        
        return {
            "entity_ids": entity_ids,
            "echa_url": iuclid_result.get("ECHAEntityUrl"),
            "section_url": iuclid_result.get("ECHASectionUrl"),
            "source_type": iuclid_result.get("SourceType")
        }
    
    def enrich_calculator_results(self, calculator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich calculator results with model metadata
        
        Maps to QPRF §3.1 (Model), §4.1-4.2 (Predicted property), §5.3 (Settings)
        """
        enriched = {}
        
        for calc_id, calc_data in calculator_results.items():
            if not isinstance(calc_data, dict):
                enriched[calc_id] = calc_data
                continue
            
            # Get calculator GUID if available
            calc_guid = calc_data.get("CalculatorGuid") or calc_data.get("Guid")
            
            # Try to get detailed metadata
            metadata = {}
            if calc_guid:
                try:
                    about_info = self.api.get_about_object(calc_guid)
                    calc_info = self.api.get_calculator_info(calc_guid)
                    
                    metadata = {
                        "name": about_info.get("Name") or calc_data.get("CalculatorName", "Unknown"),
                        "description": about_info.get("Description", ""),
                        "donator": about_info.get("Donator", ""),
                        "authors": about_info.get("Authors", ""),
                        "url": about_info.get("Url", ""),
                        "disclaimer": about_info.get("Disclaimer", ""),
                        "additional_info": about_info.get("AdditionalInfo", ""),
                        "calculator_info": calc_info
                    }
                except Exception as e:
                    logger.warning(f"Could not retrieve calculator metadata for {calc_guid}: {e}")
            
            # Merge with existing data
            enriched_calc = calc_data.copy()
            enriched_calc["qprf_metadata"] = metadata
            enriched[calc_id] = enriched_calc
        
        return enriched
    
    def enrich_profiling_results(self, profiling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich profiling results with literature and metadata
        
        Maps to QPRF §7.3 (Mechanistic information, literature)
        """
        if not profiling_results or "results" not in profiling_results:
            return profiling_results
        
        enriched_results = {}
        
        for profiler_name, profiler_data in profiling_results.get("results", {}).items():
            profiler_guid = profiler_data.get("guid")
            
            if not profiler_guid:
                enriched_results[profiler_name] = profiler_data
                continue
            
            # Get profiler metadata
            try:
                about_info = self.api.get_about_object(profiler_guid)
                profiler_info = self.api.get_profiler_info(profiler_guid)
                
                # Get literature for triggered categories
                literature = {}
                result = profiler_data.get("result", [])
                if isinstance(result, list):
                    for category in result[:5]:  # Limit to first 5 categories to avoid too many API calls
                        if isinstance(category, str):
                            try:
                                lit_refs = self.api.get_profiler_literature(profiler_guid, category)
                                if lit_refs:
                                    literature[category] = lit_refs
                            except Exception as e:
                                logger.debug(f"Could not get literature for category {category}: {e}")
                
                # Enrich profiler data
                enriched_profiler = profiler_data.copy()
                enriched_profiler["qprf_metadata"] = {
                    "name": about_info.get("Name", profiler_name),
                    "description": about_info.get("Description", ""),
                    "donator": about_info.get("Donator", ""),
                    "authors": about_info.get("Authors", ""),
                    "url": about_info.get("Url", ""),
                    "profiler_info": profiler_info,
                    "literature": literature
                }
                enriched_results[profiler_name] = enriched_profiler
                
            except Exception as e:
                logger.warning(f"Could not enrich profiler {profiler_name}: {e}")
                enriched_results[profiler_name] = profiler_data
        
        # Update results
        enriched = profiling_results.copy()
        enriched["results"] = enriched_results
        
        return enriched
    
    def enrich_metabolism_results(self, metabolism_results: List[Dict[str, Any]], 
                                  simulator_guids: List[str]) -> Dict[str, Any]:
        """Enrich metabolism results with simulator metadata
        
        Maps to QPRF §7.3.e (Metabolic considerations)
        """
        enriched = {
            "simulators": [],
            "metabolites": metabolism_results
        }
        
        # Get metadata for each simulator used
        for sim_guid in simulator_guids:
            try:
                about_info = self.api.get_about_object(sim_guid)
                sim_info = self.api.get_simulator_info(sim_guid)
                
                enriched["simulators"].append({
                    "guid": sim_guid,
                    "name": about_info.get("Name", "Unknown"),
                    "description": about_info.get("Description", ""),
                    "donator": about_info.get("Donator", ""),
                    "authors": about_info.get("Authors", ""),
                    "url": about_info.get("Url", ""),
                    "simulator_info": sim_info
                })
            except Exception as e:
                logger.warning(f"Could not get simulator metadata for {sim_guid}: {e}")
        
        return enriched
    
    def get_endpoint_metadata(self, position: str = None, endpoint: str = None) -> Dict[str, Any]:
        """Get endpoint taxonomy and database provenance
        
        Maps to QPRF §4.1 (Endpoint), provenance documentation
        """
        metadata = {
            "endpoint_tree": None,
            "endpoint_info": None,
            "units": [],
            "databases": []
        }
        
        # Get endpoint tree (cached)
        try:
            metadata["endpoint_tree"] = self.api.get_endpoint_tree()
        except Exception as e:
            logger.warning(f"Could not get endpoint tree: {e}")
        
        # Get specific endpoint info if provided
        if position:
            try:
                metadata["endpoint_info"] = self.api.get_endpoint_info(position)
            except Exception as e:
                logger.warning(f"Could not get endpoint info: {e}")
        
        # Get units and databases if endpoint specified
        if position and endpoint:
            try:
                metadata["units"] = self.api.get_endpoint_units(position, endpoint)
            except Exception as e:
                logger.warning(f"Could not get endpoint units: {e}")
            
            try:
                metadata["databases"] = self.api.get_endpoint_databases(position, endpoint)
            except Exception as e:
                logger.warning(f"Could not get endpoint databases: {e}")
        
        return metadata
    
    def create_qprf_report_data(self, chemical_data: Dict[str, Any], 
                                calculator_results: Dict[str, Any] = None,
                                profiling_results: Dict[str, Any] = None,
                                metabolism_results: List[Dict[str, Any]] = None,
                                simulator_guids: List[str] = None) -> Dict[str, Any]:
        """Create a complete QPRF-ready data structure
        
        This consolidates all enriched data into a format suitable for QPRF reporting
        """
        # Get software info
        software_info = self.get_software_info()
        
        # Enrich substance identity
        enriched_substance = self.enrich_substance_identity(chemical_data)
        
        # Enrich calculator results if provided
        enriched_calculators = None
        if calculator_results:
            enriched_calculators = self.enrich_calculator_results(calculator_results)
        
        # Enrich profiling results if provided
        enriched_profiling = None
        if profiling_results:
            enriched_profiling = self.enrich_profiling_results(profiling_results)
        
        # Enrich metabolism results if provided
        enriched_metabolism = None
        if metabolism_results and simulator_guids:
            enriched_metabolism = self.enrich_metabolism_results(metabolism_results, simulator_guids)
        
        # Build QPRF data structure
        qprf_data = {
            "report_metadata": {
                "date": datetime.now().isoformat(),
                "software": software_info,
                "format_version": "QPRF v2.0"
            },
            "substance_identity": enriched_substance,
            "calculators": enriched_calculators,
            "profiling": enriched_profiling,
            "metabolism": enriched_metabolism
        }
        
        return qprf_data
