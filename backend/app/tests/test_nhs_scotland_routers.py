"""
Unit tests for NHS Scotland FastAPI routers
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from fastapi import status

from app.main import app
from app.services.nhs_scotland_service import NHSScotlandAPIError


class TestNHSScotlandRouters:
    """Test cases for NHS Scotland API routers"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_datasets(self):
        """Mock dataset data for testing"""
        return [
            {
                "name": "genetic-conditions",
                "title": "Genetic Conditions Dataset",
                "notes": "Information about hereditary conditions",
                "tags": ["genetics", "health"],
                "metadata_created": "2023-01-01T00:00:00Z",
                "metadata_modified": "2023-06-01T00:00:00Z",
                "organization": "NHS Scotland",
                "genetic_relevance": 0.8,
                "disease_categories": ["Genetic Disorders"],
                "keywords": ["genetic", "hereditary", "conditions"]
            }
        ]

    def test_nhs_scotland_info_endpoint(self, client):
        """Test NHS Scotland information endpoint"""
        response = client.get("/nhs/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["service"] == "NHS Scotland Open Data Integration"
        assert "available_endpoints" in data
        assert "data_limitations" in data
        assert "best_for" in data

    @patch('app.routers.nhs_scotland.nhs_scotland_service.get_available_datasets')
    @patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info')
    @patch('app.routers.nhs_scotland.data_transformer.format_for_frontend')
    def test_get_available_datasets_success(self, mock_format, mock_enhance, mock_get_datasets, client, mock_datasets):
        """Test successful dataset retrieval"""
        # Setup mocks
        mock_get_datasets.return_value = mock_datasets
        mock_enhance.side_effect = lambda x: x  # Return dataset as-is
        mock_format.return_value = mock_datasets
        
        response = client.get("/nhs/datasets")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "datasets" in data
        assert "total_count" in data
        assert "categories" in data
        assert data["total_count"] == 1
        
        mock_get_datasets.assert_called_once()

    @patch('app.routers.nhs_scotland.nhs_scotland_service.get_available_datasets')
    def test_get_available_datasets_api_error(self, mock_get_datasets, client):
        """Test dataset retrieval with API error"""
        mock_get_datasets.side_effect = NHSScotlandAPIError("API unavailable")
        
        response = client.get("/nhs/datasets")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["error"] == "NHS_API_ERROR"

    @patch('app.routers.nhs_scotland.data_transformer.find_gene_related_conditions')
    @patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets')
    @patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info')
    @patch('app.routers.nhs_scotland.data_transformer.filter_datasets_by_relevance')
    @patch('app.routers.nhs_scotland.data_transformer.format_for_frontend')
    def test_get_gene_related_data_success(self, mock_format, mock_filter, mock_enhance, 
                                         mock_search, mock_find_conditions, client, mock_datasets):
        """Test successful gene-related data retrieval"""
        # Setup mocks
        mock_find_conditions.return_value = ["breast cancer", "ovarian cancer"]
        mock_search.return_value = mock_datasets
        mock_enhance.side_effect = lambda x: x
        mock_filter.return_value = mock_datasets
        mock_format.return_value = mock_datasets
        
        response = client.get("/nhs/gene/BRCA1")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["gene_name"] == "BRCA1"
        assert "related_datasets" in data
        assert "suggested_conditions" in data
        assert "total_datasets" in data
        assert "search_strategy" in data
        
        mock_find_conditions.assert_called_once_with("BRCA1")

    def test_get_gene_related_data_invalid_gene(self, client):
        """Test gene endpoint with invalid gene name"""
        # Too short
        response = client.get("/nhs/gene/A")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Too long
        response = client.get("/nhs/gene/" + "A" * 25)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.routers.nhs_scotland.data_transformer.expand_disease_search_terms')
    @patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets')
    @patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info')
    @patch('app.routers.nhs_scotland.data_transformer.filter_datasets_by_relevance')
    @patch('app.routers.nhs_scotland.data_transformer.format_for_frontend')
    def test_get_disease_related_data_success(self, mock_format, mock_filter, mock_enhance,
                                            mock_search, mock_expand_terms, client, mock_datasets):
        """Test successful disease-related data retrieval"""
        # Setup mocks
        mock_expand_terms.return_value = ["cancer", "carcinoma", "malignancy"]
        mock_search.return_value = mock_datasets
        mock_enhance.side_effect = lambda x: x
        mock_filter.return_value = mock_datasets
        mock_format.return_value = mock_datasets
        
        response = client.get("/nhs/disease/cancer")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["disease_name"] == "Cancer"
        assert "relevant_datasets" in data
        assert "data_categories" in data
        assert "geographic_coverage" in data
        assert "total_datasets" in data
        
        mock_expand_terms.assert_called_once_with("cancer")

    def test_get_disease_related_data_invalid_disease(self, client):
        """Test disease endpoint with invalid disease name"""
        # Too short
        response = client.get("/nhs/disease/a")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Too long
        response = client.get("/nhs/disease/" + "a" * 105)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.routers.nhs_scotland.nhs_scotland_service.get_congenital_conditions_data')
    def test_get_congenital_conditions_success(self, mock_get_conditions, client):
        """Test successful congenital conditions retrieval"""
        mock_data = {
            "name": "congenital-conditions",
            "title": "Congenital Anomalies",
            "description": "Birth defects and genetic conditions"
        }
        mock_get_conditions.return_value = mock_data
        
        response = client.get("/nhs/congenital-conditions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == "congenital-conditions"
        assert data["title"] == "Congenital Anomalies"

    @patch('app.routers.nhs_scotland.nhs_scotland_service.get_cancer_incidence_data')
    def test_get_cancer_incidence_success(self, mock_get_cancer, client):
        """Test successful cancer incidence retrieval"""
        mock_data = {
            "name": "cancer-incidence",
            "title": "Cancer Statistics",
            "description": "Cancer incidence rates"
        }
        mock_get_cancer.return_value = mock_data
        
        response = client.get("/nhs/cancer-incidence")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == "cancer-incidence"
        assert data["title"] == "Cancer Statistics"

    @patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets')
    @patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info')
    @patch('app.routers.nhs_scotland.data_transformer.filter_datasets_by_relevance')
    @patch('app.routers.nhs_scotland.data_transformer.format_for_frontend')
    def test_search_datasets_success(self, mock_format, mock_filter, mock_enhance,
                                   mock_search, client, mock_datasets):
        """Test successful dataset search"""
        # Setup mocks
        mock_search.return_value = mock_datasets
        mock_enhance.side_effect = lambda x: x
        mock_filter.return_value = mock_datasets
        mock_format.return_value = mock_datasets
        
        response = client.get("/nhs/search", params={"query": "genetic"})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert "message" in data
        assert "total_results" in data
        
        mock_search.assert_called_once_with("genetic", limit=20)

    def test_search_datasets_with_params(self, client):
        """Test dataset search with custom parameters"""
        with patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets') as mock_search, \
             patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info') as mock_enhance, \
             patch('app.routers.nhs_scotland.data_transformer.filter_datasets_by_relevance') as mock_filter, \
             patch('app.routers.nhs_scotland.data_transformer.format_for_frontend') as mock_format:
            
            mock_search.return_value = []
            mock_enhance.side_effect = lambda x: x
            mock_filter.return_value = []
            mock_format.return_value = []
            
            response = client.get("/nhs/search", params={
                "query": "cancer",
                "limit": 10,
                "min_relevance": 0.5
            })
            
            assert response.status_code == status.HTTP_200_OK
            mock_search.assert_called_once_with("cancer", limit=10)

    def test_search_datasets_invalid_params(self, client):
        """Test dataset search with invalid parameters"""
        # Query too short
        response = client.get("/nhs/search", params={"query": "a"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Query too long
        response = client.get("/nhs/search", params={"query": "a" * 205})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid limit
        response = client.get("/nhs/search", params={"query": "test", "limit": 0})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = client.get("/nhs/search", params={"query": "test", "limit": 100})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.routers.nhs_scotland.nhs_scotland_service.get_health_statistics_summary')
    @patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info')
    @patch('app.routers.nhs_scotland.data_transformer.format_for_frontend')
    def test_get_health_statistics_summary_success(self, mock_format, mock_enhance,
                                                 mock_get_summary, client):
        """Test successful health statistics summary retrieval"""
        mock_summary_data = {
            "categories": {
                "cancer": [{"name": "cancer-data", "title": "Cancer Stats"}]
            },
            "total_datasets_found": 10,
            "last_updated": "2023-06-01",
            "data_source": "NHS Scotland Open Data Platform"
        }
        mock_get_summary.return_value = mock_summary_data
        mock_enhance.side_effect = lambda x: x
        mock_format.return_value = [{"name": "cancer-data", "title": "Cancer Stats"}]
        
        response = client.get("/nhs/health-summary")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "categories" in data
        assert "total_datasets_found" in data
        assert data["total_datasets_found"] == 10

    @patch('app.routers.nhs_scotland.data_transformer.create_analysis_suggestions')
    def test_get_analysis_suggestions_gene(self, mock_suggestions, client):
        """Test analysis suggestions for gene"""
        mock_suggestions.return_value = {
            "search_strategies": ["Search for BRCA1 related conditions"],
            "related_datasets": ["breast cancer", "ovarian cancer"],
            "analysis_approaches": ["Check cancer data"],
            "data_limitations": ["No direct genetic data"]
        }
        
        response = client.get("/nhs/analysis-suggestions", params={"gene_name": "BRCA1"})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        
        mock_suggestions.assert_called_once_with(gene_name="BRCA1", disease_name=None)

    @patch('app.routers.nhs_scotland.data_transformer.create_analysis_suggestions')
    def test_get_analysis_suggestions_disease(self, mock_suggestions, client):
        """Test analysis suggestions for disease"""
        mock_suggestions.return_value = {
            "search_strategies": ["Search cancer datasets"],
            "analysis_approaches": ["Look for incidence data"],
            "data_limitations": ["Scotland-specific data only"]
        }
        
        response = client.get("/nhs/analysis-suggestions", params={"disease_name": "cancer"})
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        mock_suggestions.assert_called_once_with(gene_name=None, disease_name="cancer")

    def test_get_analysis_suggestions_no_params(self, client):
        """Test analysis suggestions with no parameters"""
        response = client.get("/nhs/analysis-suggestions")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Either gene_name or disease_name must be provided" in data["detail"]

    @patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets')
    def test_generic_api_error_handling(self, mock_search, client):
        """Test generic API error handling"""
        mock_search.side_effect = Exception("Unexpected error")
        
        response = client.get("/nhs/search", params={"query": "test"})
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "INTERNAL_ERROR"

    def test_endpoint_parameter_validation(self, client):
        """Test parameter validation across endpoints"""
        # Test gene endpoint path parameter validation
        response = client.get("/nhs/gene/")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test disease endpoint path parameter validation  
        response = client.get("/nhs/disease/")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test search endpoint query parameter validation
        response = client.get("/nhs/search")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.routers.nhs_scotland.nhs_scotland_service.search_datasets')
    def test_search_with_relevance_filtering(self, mock_search, client, mock_datasets):
        """Test search with relevance filtering"""
        with patch('app.routers.nhs_scotland.data_transformer.enhance_dataset_info') as mock_enhance, \
             patch('app.routers.nhs_scotland.data_transformer.filter_datasets_by_relevance') as mock_filter, \
             patch('app.routers.nhs_scotland.data_transformer.format_for_frontend') as mock_format:
            
            mock_search.return_value = mock_datasets
            mock_enhance.side_effect = lambda x: x
            mock_filter.return_value = mock_datasets
            mock_format.return_value = mock_datasets
            
            response = client.get("/nhs/search", params={
                "query": "genetic",
                "min_relevance": 0.3
            })
            
            assert response.status_code == status.HTTP_200_OK
            mock_filter.assert_called_once_with(mock_datasets, 0.3)