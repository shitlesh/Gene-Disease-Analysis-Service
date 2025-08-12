"""
NHS Scotland Open Data API Integration Service

This service provides async integration with NHS Scotland's open data API
to retrieve health and medical condition data that can support gene-disease analysis.
While specific genomic data is not available, related health datasets provide
valuable context for genetic research and analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
from ..config import settings

logger = logging.getLogger(__name__)


class NHSScotlandAPIError(Exception):
    """Custom exception for NHS Scotland API-related errors"""
    pass


class NHSScotlandService:
    """
    Service class for integrating with NHS Scotland Open Data API
    
    Provides methods to fetch health-related datasets that can inform
    gene-disease analysis, including congenital conditions, cancer data,
    and other medically relevant information.
    """
    
    def __init__(self):
        """Initialize the NHS Scotland service with HTTP client and cache"""
        self.base_url = settings.NHS_SCOTLAND_BASE_URL
        self.timeout = settings.NHS_SCOTLAND_TIMEOUT_SECONDS
        self.cache_ttl = settings.NHS_DATA_CACHE_TTL_SECONDS
        
        # Simple in-memory cache with TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # HTTP client configuration for API calls
        self._client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10),
            "headers": {
                "User-Agent": "Gene-Disease-Analysis-Tool/1.0.0",
                "Accept": "application/json"
            }
        }
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an async HTTP request to NHS Scotland API
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            Dict containing API response data
            
        Raises:
            NHSScotlandAPIError: If API request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with httpx.AsyncClient(**self._client_config) as client:
                logger.info(f"Making request to NHS Scotland API: {url}")
                
                response = await client.get(url, params=params or {})
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Successfully fetched data from {url}")
                
                return data
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error calling NHS Scotland API: {e}")
            raise NHSScotlandAPIError(f"API request timed out after {self.timeout} seconds")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling NHS Scotland API: {e.response.status_code} - {e.response.text}")
            raise NHSScotlandAPIError(f"API request failed with status {e.response.status_code}")
            
        except Exception as e:
            logger.error(f"Unexpected error calling NHS Scotland API: {e}")
            raise NHSScotlandAPIError(f"Unexpected error: {str(e)}")
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from endpoint and parameters"""
        params_str = str(sorted((params or {}).items()))
        return f"{endpoint}:{params_str}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cached data is still valid based on TTL"""
        if "timestamp" not in cache_entry:
            return False
        
        cache_time = cache_entry["timestamp"]
        expiry_time = cache_time + timedelta(seconds=self.cache_ttl)
        
        return datetime.utcnow() < expiry_time
    
    async def _cached_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make cached API request to avoid repeated calls
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            Dict containing API response data (from cache or fresh)
        """
        cache_key = self._get_cache_key(endpoint, params)
        
        # Check cache first
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            logger.info(f"Returning cached data for {endpoint}")
            return self._cache[cache_key]["data"]
        
        # Make fresh request
        data = await self._make_request(endpoint, params)
        
        # Cache the result
        self._cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        return data
    
    async def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Retrieve list of all available datasets from NHS Scotland
        
        Returns:
            List of dataset metadata dictionaries
        """
        try:
            response = await self._cached_request("action/package_list")
            
            if not response.get("success"):
                raise NHSScotlandAPIError("Failed to retrieve dataset list")
            
            dataset_names = response.get("result", [])
            
            # Get detailed info for each dataset (limited to avoid overwhelming API)
            datasets = []
            for name in dataset_names[:50]:  # Limit to first 50 for performance
                try:
                    dataset_info = await self.get_dataset_info(name)
                    if dataset_info:
                        datasets.append(dataset_info)
                except Exception as e:
                    logger.warning(f"Failed to get info for dataset {name}: {e}")
                    continue
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error fetching available datasets: {e}")
            raise NHSScotlandAPIError(f"Failed to fetch datasets: {str(e)}")
    
    async def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific dataset
        
        Args:
            dataset_name: Name/ID of the dataset
            
        Returns:
            Dict containing dataset metadata or None if not found
        """
        try:
            response = await self._cached_request("action/package_show", {"id": dataset_name})
            
            if not response.get("success"):
                logger.warning(f"Dataset {dataset_name} not found")
                return None
            
            dataset = response.get("result", {})
            
            # Extract relevant information
            return {
                "name": dataset.get("name"),
                "title": dataset.get("title"),
                "notes": dataset.get("notes", "")[:500],  # Limit description length
                "tags": [tag.get("display_name", tag.get("name")) for tag in dataset.get("tags", [])],
                "resources": len(dataset.get("resources", [])),
                "organization": dataset.get("organization", {}).get("title"),
                "metadata_created": dataset.get("metadata_created"),
                "metadata_modified": dataset.get("metadata_modified"),
            }
            
        except Exception as e:
            logger.error(f"Error fetching dataset info for {dataset_name}: {e}")
            return None
    
    async def search_datasets(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for datasets by query term
        
        Args:
            query: Search term (gene names, disease names, medical conditions)
            limit: Maximum number of results to return
            
        Returns:
            List of matching dataset dictionaries
        """
        try:
            params = {
                "q": query,
                "rows": limit,
                "sort": "score desc"
            }
            
            response = await self._cached_request("action/package_search", params)
            
            if not response.get("success"):
                raise NHSScotlandAPIError("Dataset search failed")
            
            result = response.get("result", {})
            datasets = result.get("results", [])
            
            # Transform results for frontend consumption
            formatted_datasets = []
            for dataset in datasets:
                formatted_datasets.append({
                    "name": dataset.get("name"),
                    "title": dataset.get("title"),
                    "notes": dataset.get("notes", "")[:300],  # Truncate description
                    "score": dataset.get("score", 0),
                    "tags": [tag.get("display_name", tag.get("name")) for tag in dataset.get("tags", [])],
                    "organization": dataset.get("organization", {}).get("title"),
                    "resources": len(dataset.get("resources", [])),
                    "metadata_modified": dataset.get("metadata_modified"),
                })
            
            return formatted_datasets
            
        except Exception as e:
            logger.error(f"Error searching datasets with query '{query}': {e}")
            raise NHSScotlandAPIError(f"Search failed: {str(e)}")
    
    async def get_congenital_conditions_data(self) -> Dict[str, Any]:
        """
        Retrieve congenital conditions dataset - most relevant for genetic analysis
        
        Returns:
            Dict containing congenital conditions data and metadata
        """
        try:
            dataset_info = await self.get_dataset_info("congenital-conditions")
            
            if not dataset_info:
                raise NHSScotlandAPIError("Congenital conditions dataset not found")
            
            # Get the full dataset with resources
            response = await self._cached_request("action/package_show", {"id": "congenital-conditions"})
            full_dataset = response.get("result", {})
            
            # Extract resource information (CSV files)
            resources = []
            for resource in full_dataset.get("resources", []):
                resources.append({
                    "name": resource.get("name"),
                    "description": resource.get("description"),
                    "format": resource.get("format"),
                    "url": resource.get("url"),
                    "created": resource.get("created"),
                    "last_modified": resource.get("last_modified"),
                })
            
            return {
                "dataset_info": dataset_info,
                "resources": resources,
                "conditions_categories": [
                    "Cardiovascular conditions",
                    "Nervous system conditions", 
                    "Genetic syndromes",
                    "Metabolic conditions",
                    "Skeletal conditions"
                ],
                "data_years": "2000-2022",
                "measurement_unit": "per 10,000 births"
            }
            
        except Exception as e:
            logger.error(f"Error fetching congenital conditions data: {e}")
            raise NHSScotlandAPIError(f"Failed to fetch congenital conditions: {str(e)}")
    
    async def get_cancer_incidence_data(self) -> Dict[str, Any]:
        """
        Retrieve cancer incidence dataset - relevant for cancer genetics research
        
        Returns:
            Dict containing cancer incidence data and metadata
        """
        try:
            dataset_info = await self.get_dataset_info("annual-cancer-incidence")
            
            if not dataset_info:
                raise NHSScotlandAPIError("Cancer incidence dataset not found")
            
            # Get the full dataset with resources
            response = await self._cached_request("action/package_show", {"id": "annual-cancer-incidence"})
            full_dataset = response.get("result", {})
            
            # Extract resource information
            resources = []
            for resource in full_dataset.get("resources", []):
                resources.append({
                    "name": resource.get("name"),
                    "description": resource.get("description"),
                    "format": resource.get("format"),
                    "url": resource.get("url"),
                    "created": resource.get("created"),
                    "last_modified": resource.get("last_modified"),
                })
            
            return {
                "dataset_info": dataset_info,
                "resources": resources,
                "cancer_types": [
                    "Bladder", "Bone and connective tissue", "Brain and central nervous system",
                    "Breast", "Colorectal", "Kidney", "Leukaemia", "Liver", "Lung and mesothelioma",
                    "Lymphoma", "Malignant melanoma", "Multiple myeloma", "Oesophageal", 
                    "Ovarian", "Pancreatic", "Prostate", "Stomach", "Uterine"
                ],
                "data_years": "1998-2022",
                "geographic_breakdown": ["Health Board", "Cancer Network Region"],
                "demographic_breakdown": ["Age group", "Sex"]
            }
            
        except Exception as e:
            logger.error(f"Error fetching cancer incidence data: {e}")
            raise NHSScotlandAPIError(f"Failed to fetch cancer incidence: {str(e)}")
    
    async def find_disease_related_datasets(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        Search for datasets related to a specific disease
        
        Args:
            disease_name: Name of disease to search for
            
        Returns:
            List of relevant datasets
        """
        try:
            # Search with various related terms
            search_terms = [
                disease_name,
                f"{disease_name} incidence",
                f"{disease_name} mortality",
                f"{disease_name} statistics"
            ]
            
            all_results = []
            for term in search_terms:
                try:
                    results = await self.search_datasets(term, limit=10)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    continue
            
            # Remove duplicates based on dataset name
            seen_names = set()
            unique_results = []
            for result in all_results:
                name = result.get("name")
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_results.append(result)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return unique_results[:15]  # Return top 15 most relevant
            
        except Exception as e:
            logger.error(f"Error finding disease-related datasets for '{disease_name}': {e}")
            raise NHSScotlandAPIError(f"Failed to find datasets for {disease_name}: {str(e)}")
    
    async def get_health_statistics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key health statistics available in NHS Scotland data
        
        Returns:
            Dict containing summary of available health data categories
        """
        try:
            # Search for key health categories
            categories = {
                "congenital_conditions": await self.search_datasets("congenital", limit=5),
                "cancer_data": await self.search_datasets("cancer", limit=5),
                "genetic_screening": await self.search_datasets("screening genetic", limit=5),
                "mortality_data": await self.search_datasets("mortality", limit=5),
                "disease_statistics": await self.search_datasets("disease statistics", limit=5)
            }
            
            return {
                "categories": categories,
                "total_datasets_found": sum(len(datasets) for datasets in categories.values()),
                "last_updated": datetime.utcnow().isoformat(),
                "data_source": "NHS Scotland Open Data Platform"
            }
            
        except Exception as e:
            logger.error(f"Error getting health statistics summary: {e}")
            raise NHSScotlandAPIError(f"Failed to get health statistics: {str(e)}")


# Global service instance for dependency injection
nhs_scotland_service = NHSScotlandService()