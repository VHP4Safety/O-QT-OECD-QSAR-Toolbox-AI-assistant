from typing import Optional, Dict, Any, List
import aiohttp
import asyncio
from pydantic import BaseModel

class QSARConnectionError(Exception):
    """Raised when connection to QSAR Toolbox API fails"""
    pass

class QSARTimeoutError(Exception):
    """Raised when API request times out"""
    pass

class QSARResponseError(Exception):
    """Raised when API returns invalid response"""
    pass

class ChemicalData(BaseModel):
    """Chemical data model"""
    ChemId: str
    Name: Optional[str]
    SMILES: Optional[str]
    InChI: Optional[str]
    CAS: Optional[str]
    Formula: Optional[str]

class QSARToolboxAPI:
    """Streamlined QSAR Toolbox API client"""
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: tuple = (5, 60), max_retries: int = 3):
        self.base_url = base_url
        self.connect_timeout, self.read_timeout = timeout
        self.max_retries = max_retries
        self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                connect=self.connect_timeout,
                total=self.read_timeout
            )
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make HTTP request with retry logic"""
        await self._ensure_session()
        
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                print(f"Attempting request to: {url}")  # Debug print
                async with getattr(self.session, method)(url, **kwargs) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Error response: {error_text}")  # Debug print
                        raise QSARResponseError(
                            f"API returned status {response.status}. URL: {url}. Response: {error_text}"
                        )
                    return await response.json()
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise QSARTimeoutError("Request timed out")
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise QSARConnectionError(f"Connection error: {str(e)}")

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_version(self) -> Dict[str, str]:
        """Get API version information"""
        return await self._request("get", "version")

    async def search_by_name(self, name: str) -> Optional[ChemicalData]:
        """Search chemical by name"""
        result = await self._request("get", f"search/name/{name}")
        return ChemicalData(**result[0]) if result else None

    async def search_by_smiles(self, smiles: str) -> Optional[ChemicalData]:
        """Search chemical by SMILES"""
        result = await self._request("get", f"search/smiles/{smiles}")
        return ChemicalData(**result[0]) if result else None

    async def apply_all_calculators(self, chem_id: str) -> Dict[str, Any]:
        """Get all calculated properties"""
        return await self._request("get", f"calculate/{chem_id}/all")

    async def get_all_chemical_data(self, chem_id: str) -> List[Dict[str, Any]]:
        """Get all experimental data"""
        return await self._request("get", f"experimental/{chem_id}")

    async def get_profilers(self) -> Dict[str, Any]:
        """Get profiling results"""
        return await self._request("get", "profilers")
