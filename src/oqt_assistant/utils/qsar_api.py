# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""
Streamlined QSAR Toolbox API client
"""
from __future__ import annotations

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

DEFAULT_EXPERIMENTAL_RECORD_LIMIT = int(os.getenv("OQT_EXPERIMENTAL_RECORD_LIMIT", "500"))
KEY_STUDY_MARKERS = ("key study", "key-study", "keystudy")

FAST_PROFILER_GUIDS = (
    "f05e3ca0-bba7-4a31-89a0-ab345fad7ef7",  # Uncouplers (MITOTOX)
    "44619e93-36a4-412f-9575-c21a4928fa57",  # Biodegradation primary (Biowin 4)
    "ac82ed8d-a156-42d4-83e2-ac420b748304",  # Structure similarity
    "359655bf-5bdb-43f6-96b7-6162bffe9ea1",  # Skin permeability
    "e5349d01-44b1-44bd-9213-2d194d9549fa",  # Eye irritation/corrosion Inclusion rules by BfR
    "eac042c5-58d0-4f61-944d-74f800f15cac",  # Oral absorption
    "a073010f-e717-4b16-97be-2329de725403",  # Biodegradation ultimate (Biowin 3)
    "cbcd33a4-bd5a-4769-a7ae-503277a8c562",  # Hydrolysis half-life (Kb, pH 8)(Hydrowin)
    "32656e9d-5057-4019-ba83-d2713ef60605",  # rtER Expert System - USEPA
    "ff16af4c-efe9-4a7e-8a2f-a172104c2614",  # Substance type
    "799fd171-ca3b-4c61-8e99-a2fcbdde5ccc",  # OECD HPV Chemical Categories
    "995f48fb-21ad-4ae8-9e33-42e4c58ead97",  # Biodegradation probability (Biowin 6)
    "205eaaae-5679-47ad-b6e7-b544b95436ae",  # Skin irritation/corrosion Inclusion rules by BfR
    "288b2dcd-153b-4c3b-bf23-01e39be21f9b",  # Biodegradation probability (Biowin 7)
    "5a23498e-11ea-4569-b120-b4a0f2cad786",  # Acute Oral Toxicity
    "97f51119-d6a1-4597-b9a4-5389763d3cfe",  # Example Prioritization Scheme (PBT)
    "92343cf5-043c-444b-bf10-0be0ccfc134e",  # Lipinski Rule Oasis
    "12fb37d0-fdde-4985-8092-34dd6aa32023",  # in vivo mutagenicity (Micronucleus) alerts by ISS
    "1680d436-1615-40b9-9417-4bce5fe787fa",  # Protein binding alerts for skin sensitization according to GHS
    "dfd344a8-775a-4999-9ccc-2cdc94fdf785",  # Database Affiliation
    "65f6a209-e50c-40f8-809a-3ae6baba33d4",  # Ionization at pH = 7.4
    "8118699b-f496-41fa-b0bd-d1fab56af3d9",  # Toxic hazard classification by Cramer (extended)_v.2.5
    "62d7947a-d473-45d6-8f1c-ebd45cac6267",  # Bioaccumulation - metabolism alerts
    "48ddf1f2-fae6-4672-8f71-6c13e6f1008e",  # Blood brain barrier
    "5e181bd3-3d8d-4d9e-b981-54c4ca97bb8b",  # Respiratory sensitisation
    "1a058730-3759-47c0-9f79-273598cf85f1",  # US-EPA New Chemical Categories
    "98258538-a931-408f-b459-15d769090db9",  # Biodegradation probability (Biowin 1)
    "2256b12d-5ef8-49a2-af9a-b453b3f23d96",  # Carcinogenicity (genotox and nongenotox) alerts by ISS
    "4fbdaa0a-0d8d-4b7e-8851-b8817bf50b03",  # Organic functional groups (nested)
    "a76653ce-8e5e-49a5-83e6-58c6139ca7fe",  # Protein binding potency GSH
    "585c3344-45f5-4c22-8466-7324383a3c30",  # All terpenes_R7
    "9259b6a8-a0ee-4344-bc18-ef5e394d7274",  # Organic functional groups, Norbert Haider (checkmol)
    "e0f6c1c4-04bc-47cb-bf31-86948f38a607",  # Biodeg BioHC half-life (Biowin)
    "4ced55c3-7eb8-4f9e-9c8f-3e43c6b8053c",  # Ionization at pH = 4
    "723eb011-3e5b-4565-9358-4c3d8620ba5d",  # Protein binding by OECD
    "a06271f5-944e-4892-b0ad-fa5f7217ec14",  # Acute aquatic toxicity classification by Verhaar (Modified)
    "043c65bc-a16d-472c-aa76-7e3d0b3f48eb",  # Protein Binding Potency h-CLAT
    "2782c679-745d-4ae5-8e91-e18daf8c93e0",  # DNA binding by OECD
    "2dd5c9d5-59ad-4158-b620-0e6cf8660e4f",  # Protein binding alerts for Chromosomal aberration by OASIS
    "6b981ca5-a945-4331-9e92-a8948cddb1e9",  # Protein binding potency Cys (DPRA 13%)
    "fa31b3b4-853e-470d-9a71-eded98e60c4e",  # Organic functional groups (US EPA)
    "042065de-c0a9-459d-a256-9a572b5bdedb",  # Biodegradation probability (Biowin 5)
    "f6c44acb-1264-4ba6-9815-2ad226f858c3",  # Ionization at pH = 9
    "528c9b02-d816-4996-891a-94d150d7e374",  # Biodegradation probability (Biowin 2)
    "269cbe29-b1e2-484b-905f-bbd83a380899",  # Retinoic Acid Receptor Binding
    "aa645923-a592-46fe-9069-737a2e4a7ac6",  # Toxic hazard classification by Cramer
    "0d6c0447-e311-4201-933a-bb75cd845ca7",  # Biodegradation fragments (BioWIN MITI)
    "ae169ec2-d1b5-4fb7-a59c-9dd5eb00c7f7",  # Eye irritation/corrosion Exclusion rules by BfR
    "0e9ebafd-9dcd-4899-b5bb-11b85ef32b23",  # Organic functional groups
    "f5975ac5-ef52-44e9-9c2d-ec676c724c4b",  # Hydrolysis half-life (Ka, pH 7)(Hydrowin)
    "e6fd8c5f-32b0-4f16-a8ed-95b03298e83e",  # Protein binding potency Lys (DPRA 13%)
    "34996c50-e065-4407-8aa9-e09c9e485cea",  # Toxic hazard classification by Cramer_v.2.5
    "6e1f573f-c6d3-42e5-a800-7c118f442a14",  # Tautomers unstable
    "5f241597-c420-43f9-8ff7-af33dff99c60",  # Acute aquatic toxicity MOA by OASIS
    "82022987-367a-4aca-b962-9ec7127e3040",  # Estrogen Receptor Binding
    "d00b495e-31d6-4640-b5d5-64b7f4e25f97",  # Groups of elements
    "04d1717a-70a8-4e1f-8a6d-08338bdd7ec4",  # Hydrolysis half-life (pH 6.5-7.4)
    "cba502bd-358a-426d-ab7a-688a7202f091",  # DNA alerts for AMES, CA and MNT by OASIS
    "ddd59664-8509-4a0d-9d24-99b37d91ca85",  # Hydrolysis half-life (Ka, pH 8)(Hydrowin)
    "71bf1896-8d07-4ff8-a7eb-5298fe3bd1b9",  # Protein binding by OASIS
    "7d637c92-6764-476b-903e-09735c1d3119",  # DART scheme
    "ee991334-f544-4b4c-b5d9-ff5ca5935101",  # Chemical elements
    "35894a1a-1570-44a3-aa8e-4de2ba44ad38",  # Hydrolysis half-life (Kb, pH 7)(Hydrowin)
    "3e8e16a0-7723-4649-80cd-90c21f39ca01",  # Inventory Affiliation
    "23d69016-71e4-4dbf-8306-2faad3071f99",  # DNA binding by OASIS
    "98f8a7b9-0740-4b71-af83-f108fb4f722d",  # Protein binding alerts for skin sensitization by OASIS
    "6e91aee2-2c3e-4e4b-8b44-5d14018bb4eb",  # All TERPENES
    "66ce7511-d556-4052-81db-5eaf2d370e7a",  # Bioaccumulation - metabolism half-lives
    "5486d6ff-3241-401c-82a3-d7eb9a1b2be1",  # Ionization at pH = 1
    "f662ac67-684c-4c3f-8a66-160d2b63f435",  # Aquatic toxicity classification by ECOSAR
    "e8aeedd7-72d8-4003-b2ed-cd6529677b8f",  # Keratinocyte gene expression
    "50d215a3-adb9-4b37-b72b-6bc04f313056",  # in vitro mutagenicity (Ames test) alerts by ISS
    "9aed1a13-4f47-4302-9e23-edfeb3c4283f",  # Ultimate biodeg
    "52e345cb-8f65-4524-9f33-64b96ed13878",  # Oncologic Primary Classification
)

EXTENDED_PROFILER_GUIDS = (
    "561749ad-2c8b-1ad4-b56a-dc418a9438b2",  # iSafeRat® Mechanisms of toxic Action profiler
    "5b1bc853-0ec0-423a-9200-21c440c9a7cc",  # Repeated dose (HESS)
    "b78ae496-5748-49d1-93d1-7530d5e2e367",  # Skin irritation/corrosion Exclusion rules by BfR
)

SLOW_PROFILER_GUIDS = (
    "1f1091d6-b4dd-4400-8dcf-bdbc1397a3de",  # 1 ECHA P profiler
    "6db1fc83-858a-4fc7-9289-753c7c6b38f7",  # 2 ECHA B profiler
    "30153132-a1f3-4a5a-846b-3aac62ddf453",  # 3 ECHA T profiler (ENV)
)

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

        # Internal caches for expensive discovery endpoints
        self._qsar_model_catalog: list[dict[str, Any]] | None = None

    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None, timeout: Any = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Making %s request to: %s with params: %s", method, url, params)
        
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if data else None,
                timeout=effective_timeout
            )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Response status: %s", response.status_code)
                preview = ""
                if response.content:
                    try:
                        preview = response.text[:200]
                    except Exception:  # pragma: no cover - fallback for binary payloads
                        preview = ""
                if preview:
                    logger.debug("Response content preview: %s...", preview)
            
            if not response.ok:
                raise QSARResponseError(f"API returned status {response.status_code}: {response.text}")
            
            if not response.content:
                return None

            # Try JSON first, but some endpoints may return plain text
            try:
                return response.json()
            except ValueError:
                # Fallback: return text payload (e.g., version endpoints return "4.7")
                try:
                    return response.text
                except Exception:
                    return None
            
        except requests.exceptions.Timeout as e:
            raise QSARTimeoutError(f"Request timed out: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            raise QSARConnectionError(f"Failed to connect to QSAR Toolbox API: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise QSARResponseError(f"Request failed: {str(e)}")

    def _make_request_with_robust_retry(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None, max_retries: int = None, timeout: Any = None) -> Any:
        """Make request with robust retry mechanism specifically for 500 errors"""
        if max_retries is None:
            max_retries = self.max_retries
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(
            "Making %s request to: %s with robust retry (max_retries=%s)",
            method,
            url,
            max_retries,
        )
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} for {url}")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data if data else None,
                    timeout=timeout if timeout is not None else self.timeout
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
    def search_by_cas(self, cas_number: str) -> List[Dict[str, Any]]:
        """Search chemical by CAS number (digits only as required by the Toolbox API)."""
        if not isinstance(cas_number, str):
            cas_number = str(cas_number or "")

        digits_only = "".join(ch for ch in cas_number if ch.isdigit())
        if not digits_only:
            logger.debug("CAS search skipped – identifier %s has no digits.", cas_number)
            return []

        original_timeout = self.timeout
        search_timeout = (5, 15)  # Keep CAS search responsive

        try:
            self.timeout = search_timeout
            try:
                result = self._make_request(f'search/cas/{digits_only}/false')
            except QSARResponseError as exc:
                # 400 responses indicate an invalid CAS format; treat as no hits
                if "status 400" in str(exc):
                    logger.debug("CAS search for %s returned 400 response; treating as no hits.", digits_only)
                    return []
                raise
            if isinstance(result, dict):
                return [result]
            return result or []
        finally:
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

    # UPDATED: Added record limiting + key-study preservation
    @lru_cache(maxsize=32)
    def get_all_chemical_data(
        self,
        chem_id: str,
        include_metadata: bool = True,
        *,
        record_limit: Optional[int] = None,
        keep_key_studies: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experimental data for a chemical.

        When metadata is requested we stream the response so that we can keep
        only the newest records (default 500) while preserving all key studies.
        This avoids the 20k-record metadata payload that was timing out.
        """
        original_timeout = self.timeout
        # Configurable timeouts via env for robustness in publication runs
        read_to = int(os.getenv("OQT_EXPERIMENTAL_READ_TIMEOUT_S", "30"))  # seconds
        hard_to = int(os.getenv("OQT_EXPERIMENTAL_HARD_TIMEOUT_S", "120"))  # seconds
        data_timeout = (5, read_to)
        default_limit = DEFAULT_EXPERIMENTAL_RECORD_LIMIT if include_metadata else None
        limit = record_limit if record_limit is not None else default_limit
        if limit is not None and limit <= 0:
            limit = None  # treat non-positive values as "no limit"

        try:
            self.timeout = data_timeout

            if include_metadata:
                logger.info(
                    "Requesting experimental data for %s with metadata (limit=%s, keep_key_studies=%s).",
                    chem_id,
                    limit if limit is not None else "unbounded",
                    keep_key_studies,
                )
                return self._stream_experimental_data(
                    chem_id=chem_id,
                    limit=limit,
                    keep_key_studies=keep_key_studies,
                    hard_timeout_s=hard_to,
                )

            logger.info(
                "Requesting experimental data for %s without metadata (limit=%s).",
                chem_id,
                limit if limit is not None else "unbounded",
            )
            result = self._make_request(f'data/all/{chem_id}')
            if not result:
                return []

            if isinstance(result, dict):
                result = [result]

            if limit is not None and len(result) > limit:
                logger.info(
                    "Truncating experimental dataset for %s to %s records (from %s).",
                    chem_id,
                    limit,
                    len(result),
                )
                return result[:limit]

            return result

        finally:
            self.timeout = original_timeout

    def _stream_experimental_data(
        self,
        *,
        chem_id: str,
        limit: Optional[int],
        keep_key_studies: bool,
        hard_timeout_s: int,
    ) -> List[Dict[str, Any]]:
        """Stream metadata-rich experimental data and retain only the needed slice."""
        url = f"{self.base_url}/data/all/{chem_id}"
        params = {"includeMetadata": "true"}

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
                stream=True,
            )
        except requests.exceptions.Timeout as exc:
            raise QSARTimeoutError(f"Streaming request timed out for {chem_id}: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise QSARResponseError(f"Failed streaming request for {chem_id}: {exc}") from exc

        if not response.ok:
            response.close()
            raise QSARResponseError(
                f"API returned status {response.status_code} for experimental data ({chem_id})."
            )

        decoder = json.JSONDecoder()
        buffer = ""
        array_started = False
        records: List[Dict[str, Any]] = []
        kept_non_key = 0
        kept_key = 0
        dropped = 0
        total_seen = 0
        limit_enabled = limit is not None

        import time
        start_time = time.perf_counter()

        try:
            for chunk in response.iter_content(chunk_size=65536, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk

                while True:
                    buffer = buffer.lstrip()
                    if not buffer:
                        break

                    if not array_started:
                        if buffer.startswith('['):
                            array_started = True
                            buffer = buffer[1:]
                            continue
                        response.close()
                        raise QSARResponseError("Unexpected payload while parsing experimental data stream.")

                    if buffer.startswith(']'):
                        buffer = buffer[1:]
                        logger.info(
                            "Experimental data stream for %s complete: total=%s kept=%s (non-key=%s, key=%s) dropped=%s limit=%s",
                            chem_id,
                            total_seen,
                            len(records),
                            kept_non_key,
                            kept_key,
                            dropped,
                            limit if limit is not None else "unbounded",
                        )
                        return records

                    try:
                        item, idx = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        # Need more data
                        break

                    buffer = buffer[idx:]
                    buffer = buffer.lstrip()
                    if buffer.startswith(','):
                        buffer = buffer[1:]

                    if not isinstance(item, dict):
                        logger.debug("Skipping non-dict experimental entry of type %s", type(item))
                        continue

                    total_seen += 1
                    metadata = item.get("MetaData") or []
                    is_key = keep_key_studies and self._metadata_is_key_study(metadata)
                    item["IsKeyStudy"] = bool(is_key)

                    if is_key:
                        kept_key += 1
                        records.append(item)
                        continue

                    if not limit_enabled or kept_non_key < limit:
                        kept_non_key += 1
                        records.append(item)
                    else:
                        dropped += 1

                # Hard stop if streaming took too long; return what we have
                if hard_timeout_s and (time.perf_counter() - start_time) > hard_timeout_s:
                    logger.warning(
                        "Experimental data stream for %s reached hard timeout (>%ss). Returning %s records (non-key=%s, key=%s).",
                        chem_id,
                        hard_timeout_s,
                        len(records),
                        kept_non_key,
                        kept_key,
                    )
                    return records

            if records:
                logger.warning(
                    "Experimental data stream for %s terminated unexpectedly after %s records; returning partial dataset.",
                    chem_id,
                    len(records),
                )
                return records
            raise QSARResponseError(
                f"Incomplete experimental data stream for {chem_id}: JSON array not terminated."
            )
        finally:
            response.close()

    @staticmethod
    def _metadata_is_key_study(metadata: Any) -> bool:
        if not metadata:
            return False
        for entry in metadata:
            if not isinstance(entry, str):
                continue
            lower_entry = entry.lower()
            if any(marker in lower_entry for marker in KEY_STUDY_MARKERS):
                return True
            if "purpose flag" in lower_entry and "key" in lower_entry:
                return True
        return False

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
        selected_profiler_guids: tuple | None = None,
        include_slow_profilers: bool = False
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
                    # Fallback to optimized preset using measured timings
                    auto_guids = list(FAST_PROFILER_GUIDS + EXTENDED_PROFILER_GUIDS)
                    if include_slow_profilers:
                        auto_guids += list(SLOW_PROFILER_GUIDS)

                    seen = set()
                    for guid in auto_guids:
                        if guid in seen:
                            continue
                        prof = profilers_by_guid.get(guid)
                        if not prof:
                            continue
                        seen.add(guid)
                        label = prof.get("Caption") or prof.get("Name") or guid
                        run_pairs.append((guid, label))

                    if not include_slow_profilers:
                        missing = [g for g in SLOW_PROFILER_GUIDS if g in profilers_by_guid]
                        if missing:
                            logger.info(
                                "Skipping %d slow profilers (ECHA) by default. Enable include_slow_profilers to run them.",
                                len(missing)
                            )

                target_count = len(run_pairs)

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
                status = 'Success' if len(successful_results) == target_count else 'Partial success'
                note = f"Successfully profiled with {len(successful_results)}/{target_count} selected profilers."
                if not selected_profiler_guids:
                    if include_slow_profilers:
                        note += " (includes ECHA profilers)."
                    elif any(g in SLOW_PROFILER_GUIDS for g in profilers_by_guid):
                        note += " (ECHA profilers skipped by default)."
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

    # ========== QPRF/RAAF Metadata Methods ==========

    def get_about_object(self, object_guid: str) -> Dict[str, Any]:
        """Get detailed metadata about any Toolbox object (model, calculator, profiler, simulator)
        
        Returns: { Name, Description, Donator, Disclaimer, Authors, Url, AdditionalInfo }
        Maps to QPRF §3.2 (Software), model credits, disclaimers
        """
        try:
            result = self._make_request(f'about/object/{object_guid}')
            return result or {}
        except Exception as e:
            logger.error(f"Error getting about info for object {object_guid}: {str(e)}")
            return {}

    def get_about_object_html(self, object_guid: str) -> str:
        """Get HTML-formatted about information for an object"""
        try:
            result = self._make_request(f'about/object/{object_guid}/html')
            return result if isinstance(result, str) else ""
        except Exception as e:
            logger.error(f"Error getting HTML about info for object {object_guid}: {str(e)}")
            return ""

    def get_toolbox_version(self) -> Dict[str, Any]:
        """Get OECD QSAR Toolbox version information
        
        Maps to QPRF §3.2 (Software name/version)
        """
        try:
            result = self._make_request('about/toolbox/version')
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                return {"Version": result}
            return {}
        except Exception as e:
            logger.error(f"Error getting Toolbox version: {str(e)}")
            return {}

    def get_webapi_version(self) -> Dict[str, Any]:
        """Get WebAPI version information
        
        Maps to QPRF §3.2 (Software name/version)
        """
        try:
            result = self._make_request('about/webapi/version')
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                return {"Version": result}
            return {}
        except Exception as e:
            logger.error(f"Error getting WebAPI version: {str(e)}")
            return {}

    # --- IUCLID Integration ---

    def search_iuclid_by_cas(self, cas: str, section_names: List[str] = None) -> Dict[str, Any]:
        """Search IUCLID database by CAS number
        
        Returns: {
            Chemical: { Id, IUCLIDIds: [{ IUCLIDEntityId, IUCLIDName, Type, SourceId }] },
            ECHAEntityUrl, ECHASectionUrl, SourceType
        }
        Maps to QPRF §2.3 (Other regulatory numerical identifiers)
        """
        try:
            data = {"SectionNames": section_names or []}
            result = self._make_request(f'iuclid/cas/{cas}', method='POST', data=data)
            return result or {}
        except Exception as e:
            logger.error(f"Error searching IUCLID by CAS {cas}: {str(e)}")
            return {}

    def search_iuclid_by_ec(self, ec_number: str, section_names: List[str] = None) -> Dict[str, Any]:
        """Search IUCLID database by EC number
        
        Maps to QPRF §2.3 (Other regulatory numerical identifiers)
        """
        try:
            data = {"SectionNames": section_names or []}
            result = self._make_request(f'iuclid/ecnumber/{ec_number}', method='POST', data=data)
            return result or {}
        except Exception as e:
            logger.error(f"Error searching IUCLID by EC {ec_number}: {str(e)}")
            return {}

    def search_iuclid_by_name(self, name: str, section_names: List[str] = None) -> Dict[str, Any]:
        """Search IUCLID database by chemical name
        
        Maps to QPRF §2.3 (Other regulatory numerical identifiers)
        """
        try:
            data = {"Name": name, "SectionNames": section_names or []}
            result = self._make_request('iuclid/name', method='POST', data=data)
            return result or {}
        except Exception as e:
            logger.error(f"Error searching IUCLID by name {name}: {str(e)}")
            return {}

    def search_iuclid_by_smiles(self, smiles: str, is_subfragment_search: bool = False, 
                                section_names: List[str] = None) -> Dict[str, Any]:
        """Search IUCLID database by SMILES
        
        Maps to QPRF §2.3 (Other regulatory numerical identifiers)
        """
        try:
            data = {"Smiles": smiles, "SectionNames": section_names or []}
            result = self._make_request(
                f'iuclid/smiles/{str(is_subfragment_search).lower()}', 
                method='POST', 
                data=data
            )
            return result or {}
        except Exception as e:
            logger.error(f"Error searching IUCLID by SMILES: {str(e)}")
            return {}

    def get_iuclid_databases(self) -> List[Dict[str, Any]]:
        """Get list of available IUCLID databases"""
        try:
            result = self._make_request('iuclid/databases')
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting IUCLID databases: {str(e)}")
            return []

    # --- Structure Normalization ---

    def canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string
        
        Maps to QPRF §2.6, §5.1 (Input structure normalization)
        """
        try:
            result = self._make_request('structure/canonize', params={'smiles': smiles})
            return result if isinstance(result, str) else smiles
        except Exception as e:
            logger.error(f"Error canonicalizing SMILES: {str(e)}")
            return smiles

    def get_connectivity(self, smiles: str) -> str:
        """Get connectivity string for SMILES
        
        Maps to QPRF §2.6, §5.1 (Stereochemistry/connectivity documentation)
        """
        try:
            result = self._make_request('structure/connectivity', params={'smiles': smiles})
            return result if isinstance(result, str) else ""
        except Exception as e:
            logger.error(f"Error getting connectivity: {str(e)}")
            return ""

    # --- Endpoint Taxonomy ---

    @lru_cache(maxsize=1)
    def get_endpoint_tree(self) -> Dict[str, Any]:
        """Get complete endpoint taxonomy tree
        
        Maps to QPRF §4.1 (Endpoint classification)
        """
        try:
            result = self._make_request('data/endpointtree')
            return result or {}
        except Exception as e:
            logger.error(f"Error getting endpoint tree: {str(e)}")
            return {}

    def get_endpoint_info(self, position: str) -> Dict[str, Any]:
        """Get information about a specific endpoint"""
        try:
            result = self._make_request('data/endpoint', params={'position': position})
            return result or {}
        except Exception as e:
            logger.error(f"Error getting endpoint info for {position}: {str(e)}")
            return {}

    def get_endpoint_units(self, position: str, endpoint: str) -> List[Dict[str, Any]]:
        """Get available units for an endpoint
        
        Maps to QPRF §4.2 (Units)
        """
        try:
            result = self._make_request('data/units', params={'position': position, 'endpoint': endpoint})
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting units for endpoint: {str(e)}")
            return []

    def get_endpoint_databases(self, position: str, endpoint: str) -> List[Dict[str, Any]]:
        """Get databases available for an endpoint
        
        Returns: [{ SourceId, UrlBase, Caption, Guid }]
        Maps to provenance documentation
        """
        try:
            result = self._make_request('data/databases', params={'position': position, 'endpoint': endpoint})
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting databases for endpoint: {str(e)}")
            return []

    # --- Calculator Metadata ---

    @lru_cache(maxsize=1)
    def get_all_calculators(self) -> List[Dict[str, Any]]:
        """Get list of all available calculators with metadata
        
        Returns: [{ Caption, Guid, Units, Is3D, IsExperimental, Description }]
        Maps to QPRF §3.1 (Model), §4.1-4.2 (Predicted property)
        """
        try:
            result = self._make_request('calculation')
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting calculators: {str(e)}")
            return []

    def get_calculator_info(self, calculator_guid: str) -> Dict[str, Any]:
        """Get detailed information about a specific calculator
        
        Maps to QPRF §3.1, §5.3 (Model details, settings)
        """
        try:
            result = self._make_request(f'calculation/{calculator_guid}/info')
            return result or {}
        except Exception as e:
            logger.error(f"Error getting calculator info for {calculator_guid}: {str(e)}")
            return {}

    # --- QSAR Model Metadata ---

    def get_qsar_models(self, position: str) -> List[Dict[str, Any]]:
        """Get QSAR models for a specific endpoint position
        
        Returns: [{ Caption, Guid, Position, Donator }]
        Maps to QPRF §3.1 (Model name/version)
        """
        try:
            encoded_position = urllib.parse.quote(position, safe="")
            result = self._make_request(f'qsar/list/{encoded_position}')
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting QSAR models for {position}: {str(e)}")
            return []

    def get_all_qsar_models_catalog(self) -> List[Dict[str, Any]]:
        """Discover the full catalog of QSAR models across all endpoint positions."""
        if self._qsar_model_catalog is not None:
            return self._qsar_model_catalog

        logger.info("Discovering QSAR model catalog across endpoint tree...")
        catalog: list[dict[str, Any]] = []
        seen_guids: set[str] = set()

        try:
            endpoint_tree = self.get_endpoint_tree() or []
        except Exception as exc:
            logger.error(f"Failed to retrieve endpoint tree for QSAR discovery: {exc}")
            endpoint_tree = []

        for position in endpoint_tree:
            if not isinstance(position, str):
                continue

            try:
                models = self.get_qsar_models(position) or []
            except Exception as exc:
                logger.debug(f"Skipping QSAR models for {position} due to error: {exc}")
                continue

            for entry in models:
                if not isinstance(entry, dict):
                    continue
                guid = entry.get("Guid")
                if not guid or guid in seen_guids:
                    continue
                seen_guids.add(guid)

                record = dict(entry)
                record.setdefault("RequestedPosition", position)
                catalog.append(record)

        logger.info("Discovered %d QSAR models spanning %d endpoint positions.", len(catalog), len(endpoint_tree))
        self._qsar_model_catalog = catalog
        return catalog

    def apply_qsar_model(self, qsar_guid: str, chem_id: str, *, timeout: Any = None) -> Dict[str, Any]:
        """Apply QSAR model with fail-fast retry (avoid long hangs on 500s).

        Returns a normalized dict with applicability domain and value fields.
        Maps to QPRF §4 (Prediction), §6 (Applicability Domain)
        """
        url = f"{self.base_url}/qsar/apply/{qsar_guid}/{chem_id}"

        # Interpret timeout param; default to client timeout
        effective_timeout = timeout if timeout is not None else self.timeout

        # Perform at most a couple of quick retries on transient 5xx
        attempts = 0
        max_attempts = 3
        backoff_seconds = [0.5, 1.0]  # total ~1.5s extra wait

        while attempts < max_attempts:
            attempts += 1
            try:
                # Bypass the session adapter to avoid built-in status retries
                resp = requests.get(url, timeout=effective_timeout, headers=self.session.headers)
                if resp.ok:
                    try:
                        data = resp.json()
                    except ValueError:
                        # Non-JSON payloads are unexpected here; treat as empty
                        data = {}
                    return data or {}

                # For 5xx, allow limited manual retries
                if 500 <= resp.status_code < 600 and attempts < max_attempts:
                    wait = backoff_seconds[min(attempts - 1, len(backoff_seconds) - 1)]
                    logger.warning(
                        "QSAR apply returned %s for %s (attempt %d/%d). Retrying in %.1fs...",
                        resp.status_code,
                        url,
                        attempts,
                        max_attempts,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # Other HTTP errors: do not retry
                logger.error("QSAR apply failed (%s): %s", resp.status_code, url)
                return {}

            except requests.exceptions.Timeout as e:
                # Respect per-model timeout; retry once quickly
                if attempts < max_attempts:
                    logger.warning("QSAR apply timed out (attempt %d/%d): %s", attempts, max_attempts, e)
                    continue
                raise QSARTimeoutError(f"QSAR apply timed out after {attempts} attempts: {e}")
            except requests.exceptions.ConnectionError as e:
                if attempts < max_attempts:
                    logger.warning("QSAR apply connection error (attempt %d/%d): %s", attempts, max_attempts, e)
                    continue
                raise QSARConnectionError(f"QSAR apply connection failed after {attempts} attempts: {e}")
            except requests.exceptions.RequestException as e:
                logger.error("QSAR apply request failed: %s", e)
                return {}
        return {}

    def get_qsar_domain(self, qsar_guid: str, chem_id: str) -> str:
        """Get applicability domain assessment for a QSAR prediction
        
        Maps to QPRF §6.1 (Applicability Domain)
        """
        try:
            result = self._make_request(f'qsar/domain/{qsar_guid}/{chem_id}')
            return result if isinstance(result, str) else ""
        except Exception as e:
            logger.error(f"Error getting QSAR domain: {str(e)}")
            return ""

    def get_qmrf(self, qsar_id: str) -> Any:
        """Get QMRF (QSAR Model Reporting Format) document for a model
        
        Returns: Link or PDF content
        Maps to QPRF §3.1 (Reference to QMRF)
        """
        try:
            result = self._make_request(f'report/qmrf/{qsar_id}')
            return result
        except Exception as e:
            logger.error(f"Error getting QMRF for {qsar_id}: {str(e)}")
            return None

    # --- Profiler Metadata ---

    def get_profiler_info(self, profiler_guid: str) -> Dict[str, Any]:
        """Get detailed information about a profiler
        
        Maps to QPRF §7.3 (Mechanistic information)
        """
        try:
            result = self._make_request(f'profiling/{profiler_guid}/info')
            return result or {}
        except Exception as e:
            logger.error(f"Error getting profiler info for {profiler_guid}: {str(e)}")
            return {}

    def get_profiler_categories(self, profiler_guid: str) -> List[Dict[str, Any]]:
        """Get categories for a profiler"""
        try:
            result = self._make_request(f'profiling/{profiler_guid}')
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting profiler categories: {str(e)}")
            return []

    def get_profiler_literature(self, profiler_guid: str, category: str) -> List[Dict[str, Any]]:
        """Get literature references for a profiler category
        
        Maps to QPRF §7.3 (Literature citations)
        """
        try:
            result = self._make_request(
                f'profiling/{profiler_guid}/literature',
                params={'category': category}
            )
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting profiler literature: {str(e)}")
            return []

    # --- Metabolism Simulator Metadata ---

    def get_simulator_info(self, simulator_guid: str) -> Dict[str, Any]:
        """Get detailed information about a metabolism simulator
        
        Maps to QPRF §7.3.e (Metabolic considerations)
        """
        try:
            result = self._make_request(f'metabolism/{simulator_guid}/info')
            return result or {}
        except Exception as e:
            logger.error(f"Error getting simulator info for {simulator_guid}: {str(e)}")
            return {}

    def simulate_metabolism_from_smiles(self, simulator_guid: str, smiles: str) -> List[str]:
        """Simulate metabolism from SMILES directly
        
        Returns: List of metabolite SMILES
        Maps to QPRF §7.3.e (Metabolic considerations)
        """
        try:
            result = self._make_request(f'metabolism/{simulator_guid}', params={'smiles': smiles})
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error simulating metabolism from SMILES: {str(e)}")
            return []

    # --- Grouping/Analogues ---

    def get_analogues(self, chem_id: str, profiler_guid: str, database_ids: List[int] = None) -> List[Dict[str, Any]]:
        """Get analogues for a chemical using a profiler
        
        Maps to QPRF §7.4 (Analogues)
        """
        try:
            params = {}
            if database_ids:
                params['databaseIds'] = ','.join(map(str, database_ids))
            
            result = self._make_request(f'grouping/{chem_id}/{profiler_guid}', params=params)
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting analogues: {str(e)}")
            return []

    def get_data_by_ids(self, chem_ids: List[str], position: str = None, endpoint: str = None) -> List[Dict[str, Any]]:
        """Get experimental data for multiple chemicals by IDs
        
        Used for retrieving analogue data for QPRF §7.4
        """
        try:
            params = {'ids': ','.join(chem_ids)}
            if position:
                params['position'] = position
            if endpoint:
                params['endpoint'] = endpoint
            
            result = self._make_request('data/byids', params=params)
            if isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            logger.error(f"Error getting data by IDs: {str(e)}")
            return []

    # --- Provenance helpers (new) ---
    def get_metadata_hierarchy(self):
        return self._make_request("/data/metadatahierarchy")

    def get_database_catalog(self, position: str, endpoint: str):
        return self._make_request("/data/databases", params={"position": position, "endpoint": endpoint})

    def get_object_about(self, object_guid: str):
        return self._make_request(f"/about/object/{object_guid}")

    def get_profiler_literature(self, profiler_guid: str, category: str):
        return self._make_request(f"/profiling/{profiler_guid}/literature", params={"category": category})
