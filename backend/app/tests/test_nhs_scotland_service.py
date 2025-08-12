"""
Unit tests for NHS Scotland service integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from datetime import datetime, timedelta

from app.services.nhs_scotland_service import NHSScotlandService, NHSScotlandAPIError


class TestNHSScotlandService:
    """Test cases for NHS Scotland service"""

    @pytest.fixture
    def service(self):
        """Create service instance for testing"""
        return NHSScotlandService()

    @pytest.fixture
    def mock_response_data(self):
        """Mock response data from NHS Scotland API"""
        return {
            "result": [
                {
                    "name": "congenital-conditions",
                    "title": "Congenital Conditions Dataset",
                    "notes": "Data about birth defects and genetic conditions",
                    "tags": ["genetics", "congenital"],
                    "organization": "NHS Scotland",
                    "metadata_created": "2023-01-01T00:00:00Z",
                    "metadata_modified": "2023-06-01T00:00:00Z",
                    "resources": [{"url": "test.csv"}]
                }
            ],
            "count": 1
        }

    @pytest.mark.asyncio
    async def test_get_available_datasets_success(self, service, mock_response_data):
        """Test successful dataset retrieval"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            result = await service.get_available_datasets()
            
            assert len(result) == 1
            assert result[0]["name"] == "congenital-conditions"
            assert result[0]["title"] == "Congenital Conditions Dataset"
            mock_request.assert_called_once_with("package_list", {"limit": 100})

    @pytest.mark.asyncio
    async def test_get_available_datasets_with_limit(self, service, mock_response_data):
        """Test dataset retrieval with custom limit"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            await service.get_available_datasets(limit=50)
            
            mock_request.assert_called_once_with("package_list", {"limit": 50})

    @pytest.mark.asyncio
    async def test_search_datasets_success(self, service, mock_response_data):
        """Test successful dataset search"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            result = await service.search_datasets("genetic")
            
            assert len(result) == 1
            assert result[0]["name"] == "congenital-conditions"
            mock_request.assert_called_once_with("package_search", {
                "q": "genetic",
                "rows": 20,
                "sort": "score desc"
            })

    @pytest.mark.asyncio
    async def test_search_datasets_with_params(self, service, mock_response_data):
        """Test dataset search with custom parameters"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            await service.search_datasets("cancer", limit=10, sort_by="metadata_modified")
            
            mock_request.assert_called_once_with("package_search", {
                "q": "cancer",
                "rows": 10,
                "sort": "metadata_modified desc"
            })

    @pytest.mark.asyncio
    async def test_get_congenital_conditions_data_success(self, service):
        """Test successful congenital conditions data retrieval"""
        mock_data = {
            "name": "congenital-conditions",
            "title": "Congenital Anomalies in Scotland",
            "description": "Data on congenital conditions and birth defects",
            "categories": ["genetic_disorders"],
            "genetic_relevance": 0.8
        }
        
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"result": mock_data}
            
            result = await service.get_congenital_conditions_data()
            
            assert result["name"] == "congenital-conditions"
            assert result["title"] == "Congenital Anomalies in Scotland"
            assert result["genetic_relevance"] == 0.8

    @pytest.mark.asyncio
    async def test_get_cancer_incidence_data_success(self, service):
        """Test successful cancer incidence data retrieval"""
        mock_data = {
            "name": "cancer-incidence",
            "title": "Cancer Incidence in Scotland",
            "description": "Statistics on cancer incidence rates",
            "categories": ["cancer"],
            "genetic_relevance": 0.6
        }
        
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"result": mock_data}
            
            result = await service.get_cancer_incidence_data()
            
            assert result["name"] == "cancer-incidence"
            assert result["title"] == "Cancer Incidence in Scotland"

    @pytest.mark.asyncio
    async def test_make_request_success(self, service):
        """Test successful HTTP request"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "result": []}
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await service._make_request("test_endpoint")
            
            assert result == {"success": True, "result": []}
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_with_params(self, service):
        """Test HTTP request with parameters"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            await service._make_request("test_endpoint", {"param1": "value1"})
            
            # Verify the URL was constructed correctly
            call_args = mock_get.call_args
            assert "param1=value1" in str(call_args)

    @pytest.mark.asyncio
    async def test_make_request_http_error(self, service):
        """Test HTTP request with server error"""
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=MagicMock(status_code=500)
            )
            
            with pytest.raises(NHSScotlandAPIError):
                await service._make_request("test_endpoint")

    @pytest.mark.asyncio
    async def test_make_request_timeout_error(self, service):
        """Test HTTP request with timeout"""
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")
            
            with pytest.raises(NHSScotlandAPIError):
                await service._make_request("test_endpoint")

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, service):
        """Test HTTP request with connection error"""
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")
            
            with pytest.raises(NHSScotlandAPIError):
                await service._make_request("test_endpoint")

    def test_cache_functionality(self, service):
        """Test caching mechanism"""
        # Test cache miss
        assert service._get_from_cache("test_key") is None
        
        # Test cache set and hit
        test_data = {"test": "data"}
        service._set_cache("test_key", test_data)
        assert service._get_from_cache("test_key") == test_data
        
        # Test cache expiry
        service._set_cache("expire_key", test_data, ttl_seconds=0.1)
        import time
        time.sleep(0.2)
        assert service._get_from_cache("expire_key") is None

    def test_cache_size_limit(self, service):
        """Test cache size limitation"""
        # Fill cache beyond limit
        for i in range(service.cache_size + 10):
            service._set_cache(f"key_{i}", f"data_{i}")
        
        # Cache should not exceed size limit
        assert len(service._cache) <= service.cache_size

    @pytest.mark.asyncio
    async def test_get_health_statistics_summary(self, service):
        """Test health statistics summary generation"""
        mock_datasets = [
            {"name": "cancer-data", "tags": ["cancer"], "title": "Cancer Statistics"},
            {"name": "heart-data", "tags": ["cardiovascular"], "title": "Heart Disease Data"}
        ]
        
        with patch.object(service, 'get_available_datasets', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_datasets
            
            result = await service.get_health_statistics_summary()
            
            assert "categories" in result
            assert "total_datasets_found" in result
            assert result["total_datasets_found"] == 2
            assert result["data_source"] == "NHS Scotland Open Data Platform"

    @pytest.mark.asyncio
    async def test_api_error_propagation(self, service):
        """Test that API errors are properly propagated"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = NHSScotlandAPIError("API is down")
            
            with pytest.raises(NHSScotlandAPIError, match="API is down"):
                await service.get_available_datasets()

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, service):
        """Test handling of empty API responses"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"result": []}
            
            result = await service.get_available_datasets()
            
            assert result == []

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, service):
        """Test handling of malformed API responses"""
        with patch.object(service, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"unexpected": "format"}
            
            result = await service.get_available_datasets()
            
            # Should handle gracefully and return empty list
            assert result == []