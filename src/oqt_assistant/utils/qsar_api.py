# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Streamlined QSAR Toolbox API client
"""
import os
import requests
from typing import Optional, Dict, Any, List, Union
import urllib.parse
import json
import logging
import time
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QSARConnectionError(Exception):
    """Raised when there are connection issues with the QSAR Toolbox API"""
    pass

class QSARTimeoutError(Exception):
    """Raised when API requests timeout"""
    pass

class QSARResponseError(Exception):
    """Raised when API response is invalid"""
    pass

class SearchOptions:
    EXACT_MATCH = "0"
    STARTS_WITH = "1"
    CONTAINS = "2"

class QSARToolboxAPI:
    def __init__(self, base_url: str, timeout: Union[int, tuple] = (5, 30), max_retries: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.0,  # Increased backoff factor
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False  # Don't raise immediately, let us handle retries
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"Making {method} request to: {url} with params: {params}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if data else None,
                timeout=self.timeout
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response content: {response.text[:200]}...")  # Log first 200 chars
            
            if not response.ok:
                raise QSARResponseError(f"API returned status {response.status_code}: {response.text}")
            
            if not response.content:
                return None
            
            return response.json()
            
        except requests.exceptions.Timeout as e:
            raise QSARTimeoutError(f"Request timed out: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            raise QSARConnectionError(f"Failed to connect to QSAR Toolbox API: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise QSARResponseError(f"Request failed: {str(e)}")

    def _make_request_with_robust_retry(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None, max_retries: int = None) -> Any:
        """Make request with robust retry mechanism specifically for 500 errors"""
        if max_retries is None:
            max_retries = self.max_retries
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"Making {method} request to: {url} with robust retry (max_retries={max_retries})")
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} for {url}")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data if data else None,
                    timeout=self.timeout
                )
                
                logger.debug(f"Response status: {response.status_code}")
                
                # If we get a successful response, process it
                if response.ok:
                    if not response.content:
                        return None
                    
                    result = response.json()
                    logger.info(f"Successfully retrieved data after {attempt + 1} attempts")
                    
                    # Handle the specific case for calculator results
                    if 'calculation/all/' in endpoint:
                        if isinstance(result, list):
                            # Convert list of calculator results to a dictionary
                            return {
                                str(i): calc_result
                                for i, calc_result in enumerate(result)
                            } if result else {}
                        return result or {}
                    
                    return result
                
                # Handle 500 errors with exponential backoff
                elif response.status_code == 500:
                    if attempt < max_retries:
                        wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                        logger.warning(f"Got 500 error on attempt {attempt + 1}, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise QSARResponseError(f"API returned status 500 after {max_retries + 1} attempts: {response.text}")
                
                # Handle other HTTP errors
                else:
                    raise QSARResponseError(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout as e:
                last_exception = QSARTimeoutError(f"Request timed out on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Timeout on attempt {attempt + 1}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise last_exception
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = QSARConnectionError(f"Connection failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Connection error on attempt {attempt + 1}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise last_exception
                    
            except requests.exceptions.RequestException as e:
                last_exception = QSARResponseError(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Request error on attempt {attempt + 1}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise last_exception
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise QSARResponseError(f"All {max_retries + 1} attempts failed for unknown reasons")

    def get_version(self) -> Dict[str, Any]:
        """Get API version information"""
        return self._make_request('about/toolbox/version')

    @lru_cache(maxsize=64)
    def search_by_name(self, name: str, search_option: str = SearchOptions.CONTAINS) -> List[Dict[str, Any]]:
        """Search chemical by name with a shorter timeout. Defaults to CONTAINS for robustness."""
        encoded_name = urllib.parse.quote(name)

        # Use shorter timeout for search to prevent hanging
        original_timeout = self.timeout
        search_timeout = (5, 15)  # 5s connect, 15s read timeout
        
        try:
            self.timeout = search_timeout
            result = self._make_request(f'search/name/{encoded_name}/{search_option}/false')
            if isinstance(result, dict):
                return [result]
            return result or []
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    @lru_cache(maxsize=64)
    def search_by_smiles(self, smiles: str) -> List[Dict[str, Any]]:
        """Search chemical by SMILES with a shorter timeout"""
        # Use shorter timeout for search to prevent hanging
        original_timeout = self.timeout
        search_timeout = (5, 15)  # 5s connect, 15s read timeout
        
        try:
            self.timeout = search_timeout
            result = self._make_request('search/smiles/true/false', params={'smiles': smiles})
            if isinstance(result, dict):
                return [result]
            return result or []
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    # UPDATED: Added include_metadata parameter
    @lru_cache(maxsize=32)
    def get_all_chemical_data(self, chem_id: str, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Get all experimental data for a chemical, attempting to include metadata."""
        # Use shorter timeout for data fetching to prevent hanging
        original_timeout = self.timeout
        data_timeout = (5, 60)  # Increased read timeout to 60s as metadata might increase payload size
        
        try:
            self.timeout = data_timeout
            
            # Attempt to request metadata if supported by the API endpoint (common pattern)
            params = {}
            if include_metadata:
                # Based on pyQSARToolbox hints, this parameter might be supported
                params['includeMetadata'] = 'true'
                logger.info(f"Requesting experimental data for {chem_id} with metadata.")

            result = self._make_request(f'data/all/{chem_id}', params=params)
            
            # Ensure result is always a list
            if isinstance(result, dict):
                return [result]
            return result or []
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    @lru_cache(maxsize=32)
    def apply_all_calculators(self, chem_id: str) -> Dict[str, Any]:
        """
        Apply all calculators to a chemical.

        Change: **Fail-fast profile** to prevent the UI from hanging at
        “📊 Calculating chemical properties…”. We run with a reduced timeout budget
        and far fewer retries, and return `{}` on persistent failure so the
        rest of the pipeline continues.
        """
        original_timeout = self.timeout
        # Conservative per-attempt timeout for this heavy endpoint
        self.timeout = (8, 60)  # 8s connect, 60s read

        try:
            try:
                # Limit retries to avoid minute-long stalls
                result = self._make_request_with_robust_retry(f'calculation/all/{chem_id}', max_retries=2)
                # Normalize list -> index dict for consistency
                if isinstance(result, list):
                    result = {str(i): v for i, v in enumerate(result)}
                result = result or {}

                # NEW: Attach per-calculator metadata envelope (non-breaking, additive)
                for k, rec in list(result.items()):
                    if not isinstance(rec, dict):
                        continue
                    calc_name = rec.get("CalculatorName") or rec.get("Parameter") or "Unknown"
                    family = (rec.get("Calculation") or {}).get("Family", "") or rec.get("Parameter") or ""
                    # Light-touch metadata (safe defaults)
                    rec["ModelInfo"] = {
                        "model_name": calc_name,
                        "model_version": rec.get("Version") or "N/A",
                        "software": "OECD QSAR Toolbox v6.x",
                        "developer": "N/A",
                        "endpoint": family or "N/A",
                        "algorithm_summary": "N/A",
                        "training_set_summary": None,
                        "validation_summary": None,
                        "applicability_domain_method": "N/A",
                        "applicability_domain_decision_for_query": "N/A",
                        "parameters": {},
                        "references": []
                    }
                    result[k] = rec
                return result
            except (QSARTimeoutError, QSARResponseError, QSARConnectionError) as e:
                logger.warning(f"apply_all_calculators failed fast for {chem_id}: {e}")
                return {}  # fail fast; let UI proceed
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    # --- Metabolism (UPDATED) ---

    def get_simulators(self) -> List[Dict[str, Any]]:
        """Get all available metabolism simulators (GET /api/v6/metabolism)"""
        try:
            # According to the provided API spec, the endpoint is just 'metabolism'
            result = self._make_request('metabolism')
            if not isinstance(result, list):
                result = [result] if result else []

            # Filter out the "No metabolism" entry if it exists. 
            # In the multi-select UI, an empty selection means "skip", so we don't need this option.
            filtered_result = [s for s in result if s.get("Guid") != "00000000-0000-0000-0000-000000000000"]
            
            return filtered_result

        except Exception as e:
            logger.error(f"Error getting simulators: {str(e)}. Returning fallback list.")
            # Fallback list if the API call fails (e.g., QSAR Toolbox version differences)
            # Exclude "No metabolism" from the fallback list as well.
            return [
                {"Guid": "7F130D2D-EB7F-4765-A731-FBD5B7D43D6C", "Caption": "Autoxidation simulator"},
                {"Guid": "6f90b44e-cd34-4b10-be36-e9f2b1d9f5f0", "Caption": "Rat liver S9 metabolism"},
                {"Guid": "e1853bff-d81c-480b-b3b6-00913c051d4b", "Caption": "In vivo Rat metabolism"},
                {"Guid": "83142d29-7520-4298-9158-b1271a974336", "Caption": "Microbial metabolism"}
            ]

    @lru_cache(maxsize=32)
    def apply_simulator(self, simulator_guid: str, chem_id: str) -> List[Dict[str, Any]]:
        """Applies a simulator to a chemical, returning metabolites (GET /api/v6/metabolism/{simulatorGuid}/{chemId})"""
        
        # Safety check, although UI should prevent this GUID from being passed here.
        if simulator_guid == "00000000-0000-0000-0000-000000000000":
            logger.warning("Attempted to run 'No metabolism' simulator via apply_simulator. Skipping.")
            return []

        # Metabolism simulation can be slow, use a longer timeout and robust retry
        original_timeout = self.timeout
        # Increase timeout significantly for metabolism: 5 min read timeout
        metabolism_timeout = (10, 300) 

        try:
            self.timeout = metabolism_timeout
            logger.info(f"Applying metabolism simulator {simulator_guid} to chemical {chem_id} with extended timeout.")
            # Using robust retry as metabolism can sometimes cause internal server errors (500)
            result = self._make_request_with_robust_retry(
                f'metabolism/{simulator_guid}/{chem_id}', 
                max_retries=3 # Fewer retries due to long timeout and high cost of re-running
            )
            
            if isinstance(result, list):
                return result
            elif result:
                return [result]
            return []
        except (QSARTimeoutError, QSARResponseError) as e:
            logger.error(f"Metabolism simulation failed for simulator {simulator_guid} on chemical {chem_id}: {str(e)}")
            # Re-raise the exception so the caller (app.py) can handle it and inform the user
            # We wrap it in QSARResponseError for consistent handling in the app layer
            raise QSARResponseError(f"Metabolism simulation failed or timed out: {str(e)}")
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    # --- Profiling (Updated) ---

    def get_profilers(self) -> List[Dict[str, Any]]:
        """Get all available profilers (GET /api/v6/profiling)"""
        try:
            result = self._make_request('profiling')
            if isinstance(result, list):
                return result
            elif result:
                return [result]
            return []
        except Exception as e:
            logger.error(f"Error getting profilers: {str(e)}")
            return []
    
        
    @lru_cache(maxsize=16)
    def get_chemical_profiling(
        self,
        chem_id: str,
        simulator_guid: str = "00000000-0000-0000-0000-000000000000",
        selected_profiler_guids: tuple | None = None
    ) -> Dict[str, Any]:
        """Get profiling results for a specific chemical.
        If `selected_profiler_guids` (GUID tuple) is provided, run exactly those.
        Otherwise, use an optimized default subset (fast profilers)."""
        try:
            # Get available profilers first
            available_profilers_list = self.get_profilers()
            logger.info(f"Got {len(available_profilers_list)} available profilers")
            
            # Prepare profiler information for display
            profiler_display_list = []
            profilers_by_name = {}
            profilers_by_guid = {}
            for profiler in available_profilers_list:
                if isinstance(profiler, dict):
                    caption = profiler.get('Caption', 'Unknown')
                    guid = profiler.get('Guid', '')
                    profiler_type = profiler.get('Type', 'Unknown')
                    
                    if caption and guid:
                        profilers_by_name[caption] = profiler
                        profilers_by_guid[guid] = profiler
                        profiler_display_list.append({
                            'name': caption,
                            'type': profiler_type,
                            'guid': guid,
                            'status': 'Available'
                        })

            
            # List of PROVEN fast profilers (optimized balance of speed vs coverage)
            # This optimization addresses the practical limitations of the QSAR Toolbox API responsiveness.
            fast_profilers_names = [
                "DNA binding by OASIS",
                "Protein binding by OASIS",
                "Protein binding alerts for Chromosomal aberration by OASIS",
                "DART scheme",
                "Estrogen Receptor Binding",
                "Chemical elements",
                "Substance type",
                "Acute aquatic toxicity classification by Verhaar",
                "Aquatic toxicity classification by ECOSAR",
                "Organic functional groups",
                "Organic functional groups (nested)",
                "Organic functional groups (US EPA)",
                "Organic functional groups, Norbert Haider (checkmol)"
            ]
            
            successful_results = {}
            # Set a reasonable timeout for individual profiling requests
            original_timeout = self.timeout
            profiling_timeout = (5, 15)  # 5s connect, 15s read
            
            try:
                # Set the timeout once
                self.timeout = profiling_timeout
                
                # Determine which profilers to run
                run_pairs: list[tuple[str, str]] = []
                if selected_profiler_guids:
                    # Run exactly what the user selected (GUIDs)
                    for g in selected_profiler_guids:
                        prof = profilers_by_guid.get(g)
                        if prof:
                            run_pairs.append( (prof.get('Guid'), prof.get('Caption') or prof.get('Name') or g) )
                else:
                    # Fallback to optimized default by name
                    for name in fast_profilers_names:
                        if name in profilers_by_name:
                            guid = profilers_by_name[name].get('Guid')
                            if guid:
                                run_pairs.append( (guid, name) )

                for guid, label in run_pairs:
                    logger.info(f"Attempting to profile with: {label}")
                    try:
                        result = self._make_request(f'profiling/{guid}/{chem_id}/{simulator_guid}')
                        if result:
                            logger.info(f"Successfully profiled with {label}")
                            p = profilers_by_guid.get(guid, {})
                            successful_results[label] = {
                                'result': result,
                                'type': p.get('Type', 'Profiler'),
                                'guid': guid
                            }
                    except Exception as e:
                        logger.warning(f"Error with profiler {label} (GUID: {guid}): {str(e)}")
            finally:
                # Restore original timeout
                self.timeout = original_timeout
            
            # Return results
            if successful_results:
                target_count = len(run_pairs)
                status = 'Success' if len(successful_results) == target_count else 'Partial success'
                note = f"Successfully profiled with {len(successful_results)}/{target_count} selected profilers."
                return {
                    'results': successful_results,
                    'available_profilers': profiler_display_list,
                    'status': status,
                    'note': note
                }
            else:
                return {
                    'available_profilers': profiler_display_list,
                    'status': 'Limited',
                    'note': 'No profiler results returned within the time limit. Consider running a smaller subset.'
                }
            
        except Exception as e:
            logger.error(f"Error during chemical profiling orchestration: {str(e)}")
            return {
                'status': 'Error',
                'error': str(e),
                'note': 'Unable to retrieve profiler information or execute profiling.'
            }
    
    def get_relevant_profilers(self, endpoint_position: str = "Endpoints") -> List[Dict]:
        """Get profilers relevant to a specific endpoint position"""
        try:
            params = {
                'position': endpoint_position
            }
            
            return self._make_request('profiling/relevancies', params=params) or []
        except Exception as e:
            logger.error(f"Error getting relevant profilers: {str(e)}")
            return []
            
    def get_categorization(self, chem_id: str) -> Dict[str, Any]:
        """Get categorization data for a chemical which might include profiling"""
        try:
            result = self._make_request(f'categorization/{chem_id}')
            if not result:
                logger.warning(f"No categorization data for chemical {chem_id}")
                return {}
            return result or {}
        except Exception as e:
            logger.error(f"Error getting categorization data: {str(e)}")
            return {}
