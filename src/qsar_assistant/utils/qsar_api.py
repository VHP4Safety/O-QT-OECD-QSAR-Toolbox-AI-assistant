"""
Streamlined QSAR Toolbox API client
"""
import os
import requests
from typing import Optional, Dict, Any, List, Union
import urllib.parse
import json
import logging
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
    def __init__(self, base_url: str, timeout: Union[int, tuple] = (5, 30), max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
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
        logger.debug(f"Making {method} request to: {url}")
        
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

    @lru_cache(maxsize=32)
    def get_all_chemical_data(self, chem_id: str) -> List[Dict[str, Any]]:
        """Get all experimental data for a chemical with a shorter timeout"""
        # Use shorter timeout for data fetching to prevent hanging
        original_timeout = self.timeout
        data_timeout = (5, 30)  # 5s connect, 30s read timeout for potentially larger datasets
        
        try:
            self.timeout = data_timeout
            result = self._make_request(f'data/all/{chem_id}')
            # Ensure result is always a list
            if isinstance(result, dict):
                return [result]
            return result or []
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    @lru_cache(maxsize=32)
    def apply_all_calculators(self, chem_id: str) -> Dict[str, Any]:
        """Apply all calculators to a chemical with a shorter timeout"""
        # Use shorter timeout for calculator operations to prevent hanging
        original_timeout = self.timeout
        calc_timeout = (5, 30)  # 5s connect, 30s read timeout
        
        try:
            self.timeout = calc_timeout
            result = self._make_request(f'calculation/all/{chem_id}')
            if isinstance(result, list):
                # Convert list of calculator results to a dictionary
                return {
                    str(i): calc_result
                    for i, calc_result in enumerate(result)
                } if result else {}
            return result or {}
        finally:
            # Restore original timeout
            self.timeout = original_timeout

    def get_profilers(self) -> Dict[str, Any]:
        """Get all available profilers"""
        try:
            result = self._make_request('profiling')
            if isinstance(result, list):
                # Convert list of profilers to a dictionary with better naming
                profilers_dict = {}
                for i, profiler in enumerate(result):
                    # Use Caption as key if available, otherwise use a numbered key
                    if isinstance(profiler, dict) and 'Caption' in profiler:
                        key = profiler['Caption']
                    else:
                        key = f"profiler_{i}"
                    profilers_dict[key] = profiler
                return profilers_dict if result else {}
            return result or {}
        except Exception as e:
            logger.error(f"Error getting profilers: {str(e)}")
            return {}
    
    def get_simulators(self) -> Dict[str, Any]:
        """Get all available metabolism simulators"""
        try:
            # Attempt to get a list of all available simulators
            result = self._make_request('metabolism/simulators')
            if isinstance(result, list):
                # Convert list of simulators to a dictionary
                simulators_dict = {}
                for i, simulator in enumerate(result):
                    # Use Caption as key if available, otherwise use a numbered key
                    if isinstance(simulator, dict) and 'Caption' in simulator:
                        key = simulator['Caption']
                    else:
                        key = f"simulator_{i}"
                    simulators_dict[key] = simulator
                return simulators_dict if result else {}
            return result or {}
        except Exception as e:
            logger.error(f"Error getting simulators: {str(e)}")
            # Return a default set of well-known simulator GUIDs
            return {
                "No metabolism": {"Guid": "00000000-0000-0000-0000-000000000000", "Caption": "No metabolism"},
                "Autoxidation simulator": {"Guid": "7F130D2D-EB7F-4765-A731-FBD5B7D43D6C", "Caption": "Autoxidation simulator"},
                "Rat liver S9 metabolism": {"Guid": "6f90b44e-cd34-4b10-be36-e9f2b1d9f5f0", "Caption": "Rat liver S9 metabolism"}
            }
        
    @lru_cache(maxsize=16)
    def get_chemical_profiling(self, chem_id: str) -> Dict[str, Any]:
        """Get profiling results for a specific chemical using all available profilers"""
        try:
            # Get available profilers first
            profilers = self.get_profilers()
            logger.info(f"Got {len(profilers)} profilers")
            
            # Get available simulators
            simulators = self.get_simulators()
            logger.info(f"Got {len(simulators)} simulators")
            
            # Use the "No metabolism" simulator by default
            default_simulator_guid = "00000000-0000-0000-0000-000000000000"
            
            # First try the /all endpoint with a shorter timeout
            short_timeout = (5, 15)  # Much shorter timeout for this problematic endpoint
            original_timeout = self.timeout
            all_results = None
            
            try:
                # Temporarily set a shorter timeout for this specific call
                self.timeout = short_timeout
                logger.info(f"Attempting to use the profiling/all/{chem_id} endpoint with shorter timeout {short_timeout}")
                
                try:
                    # This endpoint should profile the chemical with all available profilers
                    all_results = self._make_request(f'profiling/all/{chem_id}')
                    if all_results:
                        logger.info(f"Successfully got profiling results for all profilers")
                except Exception as e:
                    logger.warning(f"Could not profile with all profilers at once: {str(e)}")
            finally:
                # Restore original timeout for subsequent calls
                self.timeout = original_timeout
            
            # Extract profiler information for display
            profiler_list = []
            successful_results = {}
            
            # Create a list of profilers for display
            for key, profiler in profilers.items():
                if isinstance(profiler, dict):
                    caption = profiler.get('Caption', '')
                    guid = profiler.get('Guid', '')
                    profiler_type = profiler.get('Type', 'Unknown')
                    
                    if caption and guid:
                        # Add to display list
                        profiler_list.append({
                            'name': caption,
                            'type': profiler_type,
                            'guid': guid,
                            'status': 'Available',
                            'description': 'This profiler can be used to categorize chemicals based on structural features'
                        })
            
            # If we got results from the /all endpoint, return them
            if all_results:
                return {
                    'results': all_results,
                    'available_profilers': profiler_list,
                    'status': 'Success',
                    'note': 'Successfully profiled the chemical with all profilers'
                }
            
            # If that fails, try individual profilers with the right parameters
            # First, try all profilers using their GUIDs
            logger.info(f"Trying to profile with individual profilers")
            
            # Try these known simulator GUIDs in order
            simulator_guids = [
                "00000000-0000-0000-0000-000000000000",  # No metabolism (default)
                "7F130D2D-EB7F-4765-A731-FBD5B7D43D6C",  # Autoxidation simulator
                "6f90b44e-cd34-4b10-be36-e9f2b1d9f5f0"   # Rat liver S9 metabolism
            ]
            
            # Attempt to profile with all available profilers
            attempts = 0
            max_attempts = 80  # Set high to try all profilers
            
            # List of PROVEN fast profilers (based on actual testing)
            fast_profilers = [
                # These profilers respond in under 5 seconds consistently
                "DNA binding by OASIS",
                "Protein binding by OASIS",
                "Protein binding alerts for Chromosomal aberration by OASIS",
                "DART scheme",
                "Estrogen Receptor Binding",
                "Chemical elements",
                "Substance type",
                "Acute aquatic toxicity classification by Verhaar",
                "Aquatic toxicity classification by ECOSAR",
                "Organic functional groups"
            ]
            
            # Simple approach: try only proven fast profilers with a reasonable timeout
            original_timeout = self.timeout
            reasonable_timeout = (5, 15)  # 5s connect, 15s read
            
            try:
                # Set the timeout once
                self.timeout = reasonable_timeout
                
                # Try only fast profilers
                fast_profiler_guids = {}
                
                # Build dictionary of fast profilers
                for profiler_name in fast_profilers:
                    for key, profiler in profilers.items():
                        if isinstance(profiler, dict) and profiler.get('Caption', ''):
                            caption = profiler.get('Caption', '')
                            if profiler_name.lower() in caption.lower():
                                fast_profiler_guids[caption] = profiler.get('Guid', '')
                
                # Only try the fast profilers
                for name, guid in fast_profiler_guids.items():
                    if guid == '':
                        continue
                        
                    attempts += 1
                    logger.info(f"Attempting to profile with FAST profiler: {name}")
                    
                    try:
                        # Make the profiling request with profiler and simulator GUIDs
                        result = self._make_request(f'profiling/{guid}/{chem_id}/{default_simulator_guid}')
                        if result:
                            logger.info(f"Successfully profiled with {name}")
                            successful_results[name] = {
                                'result': result,
                                'type': 'Profiler',
                                'guid': guid
                            }
                    except Exception as e:
                        logger.warning(f"Error with profiler {name}: {str(e)}")
            finally:
                # Restore original timeout
                self.timeout = original_timeout
            
            # Try the relevant profilers endpoint if we didn't get enough results
            if len(successful_results) < 5:
                try:
                    relevant_profilers = self._make_request('profiling/relevancies', 
                                                           params={'position': 'Endpoints'})
                    
                    if isinstance(relevant_profilers, list) and len(relevant_profilers) > 0:
                        logger.info(f"Got {len(relevant_profilers)} relevant profilers")
                        
                        # Try each relevant profiler
                        for relevant in relevant_profilers:
                            if attempts >= max_attempts:
                                break
                                
                            if isinstance(relevant, dict):
                                rel_guid = relevant.get('Guid', '')
                                rel_name = relevant.get('Caption', f"Relevant Profiler {attempts}")
                                
                                if rel_guid and rel_name not in successful_results:
                                    attempts += 1
                                    
                                    # Try with default simulator
                                    try:
                                        result = self._make_request(f'profiling/{rel_guid}/{chem_id}/{default_simulator_guid}')
                                        if result:
                                            logger.info(f"Successfully profiled with relevant profiler {rel_name}")
                                            successful_results[rel_name] = {
                                                'result': result,
                                                'type': 'Relevant Profiler',
                                                'guid': rel_guid
                                            }
                                    except Exception as e:
                                        logger.warning(f"Failed to profile with relevant profiler {rel_name}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to get relevant profilers: {str(e)}")
            
            # Return results if we got any
            if successful_results:
                logger.info(f"Successfully profiled with {len(successful_results)} profilers")
                return {
                    'results': successful_results,
                    'available_profilers': profiler_list,
                    'status': 'Partial success',
                    'note': f'Successfully profiled with {len(successful_results)} out of {len(profilers)} profilers'
                }
            
            # If we couldn't get any profiling results, return just the profiler list
            logger.info(f"Could not get any profiling results, returning only profiler list")
            return {
                'available_profilers': profiler_list,
                'status': 'Limited',
                'note': 'Could not retrieve profiling results with the current API configuration. The complete profiling results are available in the QSAR Toolbox desktop application.'
            }
            
        except Exception as e:
            logger.error(f"Error getting profiling results: {str(e)}")
            return {
                'status': 'Error',
                'error': str(e),
                'note': 'Unable to retrieve profiler information'
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
