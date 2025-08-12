"""
Unit tests for NHS Scotland Pydantic models
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from app.models.nhs_scotland import (
    DatasetInfo, CongenitalConditionsData, CancerIncidenceData,
    HealthStatisticsSummary, GeneRelatedDataResponse, DiseaseDataResponse,
    AvailableDatasetsResponse, NHSDataResponse, ErrorResponse,
    GeneSearchRequest, DiseaseSearchRequest, DatasetSearchRequest
)


class TestNHSScotlandModels:
    """Test cases for NHS Scotland Pydantic models"""

    def test_dataset_info_valid(self):
        """Test DatasetInfo with valid data"""
        data = {
            "name": "test-dataset",
            "title": "Test Dataset",
            "notes": "A test dataset for genetic analysis",
            "resources": 3,
            "organization": "NHS Scotland",
            "tags": ["genetics", "health", "scotland"]
        }
        
        dataset = DatasetInfo(**data)
        
        assert dataset.name == "test-dataset"
        assert dataset.title == "Test Dataset"
        assert dataset.notes == "A test dataset for genetic analysis"
        assert dataset.resources == 3
        assert len(dataset.tags) == 3

    def test_dataset_info_optional_fields(self):
        """Test DatasetInfo with minimal required fields"""
        data = {
            "name": "minimal-dataset",
            "title": "Minimal Dataset",
            "notes": "Minimal description",
            "resources": 1
        }
        
        dataset = DatasetInfo(**data)
        
        assert dataset.name == "minimal-dataset"
        assert dataset.title == "Minimal Dataset"
        assert dataset.notes == "Minimal description"
        assert dataset.tags == []
        assert dataset.resources == 1

    def test_dataset_info_validation_errors(self):
        """Test DatasetInfo validation errors"""
        # Missing required fields
        with pytest.raises(ValidationError):
            DatasetInfo()
        
        # Missing notes field
        with pytest.raises(ValidationError):
            DatasetInfo(name="test", title="Test", resources=1)

    def test_congenital_conditions_data_valid(self):
        """Test CongenitalConditionsData with valid data"""
        dataset_info = {
            "name": "congenital-conditions",
            "title": "Congenital Anomalies in Scotland",
            "notes": "Data about birth defects and genetic conditions",
            "resources": 3
        }
        
        data = {
            "dataset_info": dataset_info,
            "resources": [],
            "conditions_categories": ["Heart defects", "Neural tube defects"],
            "data_years": "2020-2023",
            "measurement_unit": "per 10,000 births"
        }
        
        conditions = CongenitalConditionsData(**data)
        
        assert conditions.dataset_info.name == "congenital-conditions"
        assert conditions.dataset_info.title == "Congenital Anomalies in Scotland"
        assert "Heart defects" in conditions.conditions_categories

    def test_cancer_incidence_data_valid(self):
        """Test CancerIncidenceData with valid data"""
        dataset_info = {
            "name": "cancer-incidence",
            "title": "Cancer Incidence in Scotland",
            "notes": "Annual cancer incidence statistics",
            "resources": 5
        }
        
        data = {
            "dataset_info": dataset_info,
            "resources": [],
            "cancer_types": ["breast", "lung", "colorectal"],
            "data_years": "2018-2022",
            "geographic_breakdown": ["Scotland", "Health Boards"],
            "demographic_breakdown": ["Age", "Sex"]
        }
        
        cancer = CancerIncidenceData(**data)
        
        assert cancer.dataset_info.name == "cancer-incidence"
        assert len(cancer.cancer_types) == 3
        assert "breast" in cancer.cancer_types

    def test_gene_related_data_response_valid(self):
        """Test GeneRelatedDataResponse with valid data"""
        from app.models.nhs_scotland import DatasetSearchResult
        
        dataset = DatasetSearchResult(
            name="breast-cancer-data",
            title="Breast Cancer Statistics",
            notes="Breast cancer incidence data",
            score=0.9,
            resources=2
        )
        
        data = {
            "gene_name": "BRCA1",
            "related_datasets": [dataset],
            "suggested_conditions": ["breast cancer", "ovarian cancer"],
            "total_datasets": 1,
            "search_strategy": "Searched for BRCA1 and related conditions"
        }
        
        response = GeneRelatedDataResponse(**data)
        
        assert response.gene_name == "BRCA1"
        assert len(response.related_datasets) == 1
        assert "breast cancer" in response.suggested_conditions
        assert response.total_datasets == 1

    def test_disease_data_response_valid(self):
        """Test DiseaseDataResponse with valid data"""
        from app.models.nhs_scotland import DatasetSearchResult
        
        dataset = DatasetSearchResult(
            name="cancer-statistics",
            title="Cancer Statistics",
            notes="Cancer incidence data",
            score=0.8,
            resources=3
        )
        
        data = {
            "disease_name": "Cancer",
            "relevant_datasets": [dataset],
            "data_categories": ["Cancer", "Oncology"],
            "geographic_coverage": ["Scotland"],
            "total_datasets": 1
        }
        
        response = DiseaseDataResponse(**data)
        
        assert response.disease_name == "Cancer"
        assert len(response.relevant_datasets) == 1
        assert "Cancer" in response.data_categories
        assert "Scotland" in response.geographic_coverage

    def test_available_datasets_response_valid(self):
        """Test AvailableDatasetsResponse with valid data"""
        dataset = DatasetInfo(
            name="test-dataset",
            title="Test Dataset",
            notes="Test description",
            resources=1
        )
        
        data = {
            "datasets": [dataset],
            "total_count": 1,
            "categories": ["Genetic Disorders", "Cancer"]
        }
        
        response = AvailableDatasetsResponse(**data)
        
        assert len(response.datasets) == 1
        assert response.total_count == 1
        assert len(response.categories) == 2

    def test_nhs_data_response_success(self):
        """Test NHSDataResponse for successful response"""
        data = {
            "success": True,
            "data": {"test": "data"},
            "message": "Operation successful",
            "total_results": 5
        }
        
        response = NHSDataResponse(**data)
        
        assert response.success is True
        assert response.data == {"test": "data"}
        assert response.message == "Operation successful"
        assert response.total_results == 5

    def test_nhs_data_response_minimal(self):
        """Test NHSDataResponse with minimal fields"""
        response = NHSDataResponse(success=False, data={})
        
        assert response.success is False
        assert response.data == {}
        assert response.message is None
        assert response.total_results is None

    def test_error_response_valid(self):
        """Test ErrorResponse with valid data"""
        data = {
            "error": "API_ERROR",
            "message": "NHS Scotland API is temporarily unavailable"
        }
        
        error = ErrorResponse(**data)
        
        assert error.error == "API_ERROR"
        assert error.message == "NHS Scotland API is temporarily unavailable"
        assert error.success is False

    def test_gene_search_request_valid(self):
        """Test GeneSearchRequest validation"""
        data = {
            "gene_name": "BRCA1",
            "include_related_conditions": True
        }
        
        request = GeneSearchRequest(**data)
        
        assert request.gene_name == "BRCA1"
        assert request.include_related_conditions is True

    def test_gene_search_request_validation(self):
        """Test GeneSearchRequest validation errors"""
        # Invalid gene name length
        with pytest.raises(ValidationError):
            GeneSearchRequest(gene_name="A")  # Too short
        
        with pytest.raises(ValidationError):
            GeneSearchRequest(gene_name="A" * 25)  # Too long

    def test_disease_search_request_valid(self):
        """Test DiseaseSearchRequest validation"""
        data = {
            "disease_name": "breast cancer",
            "limit": 10
        }
        
        request = DiseaseSearchRequest(**data)
        
        assert request.disease_name == "breast cancer"
        assert request.limit == 10

    def test_disease_search_request_validation(self):
        """Test DiseaseSearchRequest validation errors"""
        # Invalid disease name length
        with pytest.raises(ValidationError):
            DiseaseSearchRequest(disease_name="a")  # Too short
        
        with pytest.raises(ValidationError):
            DiseaseSearchRequest(disease_name="a" * 105)  # Too long

    def test_dataset_search_request_valid(self):
        """Test DatasetSearchRequest validation"""
        from app.models.nhs_scotland import HealthCategory
        
        data = {
            "query": "genetic conditions",
            "limit": 25,
            "category": HealthCategory.CONGENITAL_CONDITIONS
        }
        
        request = DatasetSearchRequest(**data)
        
        assert request.query == "genetic conditions"
        assert request.limit == 25
        assert request.category == HealthCategory.CONGENITAL_CONDITIONS

    def test_dataset_search_request_validation(self):
        """Test DatasetSearchRequest validation errors"""
        # Invalid query length
        with pytest.raises(ValidationError):
            DatasetSearchRequest(query="a")  # Too short
        
        # Invalid limit
        with pytest.raises(ValidationError):
            DatasetSearchRequest(query="test", limit=0)  # Too small
        
        with pytest.raises(ValidationError):
            DatasetSearchRequest(query="test", limit=200)  # Too large

    def test_health_statistics_summary_valid(self):
        """Test HealthStatisticsSummary with valid data"""
        from app.models.nhs_scotland import DatasetSearchResult
        
        cancer_dataset = DatasetSearchResult(
            name="cancer-data",
            title="Cancer Data",
            notes="Cancer statistics",
            score=0.8,
            resources=2
        )
        
        data = {
            "categories": {
                "cancer": [cancer_dataset]
            },
            "total_datasets_found": 50,
            "last_updated": "2023-07-15T10:30:00Z",
            "data_source": "NHS Scotland Open Data Platform"
        }
        
        summary = HealthStatisticsSummary(**data)
        
        assert len(summary.categories) == 1
        assert summary.total_datasets_found == 50
        assert summary.data_source == "NHS Scotland Open Data Platform"

    def test_model_serialization(self):
        """Test that models can be serialized to JSON"""
        dataset = DatasetInfo(
            name="test",
            title="Test Dataset",
            notes="Test description",
            resources=1
        )
        
        # Should be able to convert to dict
        dataset_dict = dataset.dict()
        assert isinstance(dataset_dict, dict)
        assert dataset_dict["name"] == "test"
        
        # Should be able to convert to JSON
        dataset_json = dataset.json()
        assert isinstance(dataset_json, str)
        assert "test" in dataset_json

    def test_model_field_aliases(self):
        """Test that field aliases work correctly"""
        data = {
            "name": "test-dataset",
            "title": "Test Dataset",
            "notes": "Test description",
            "resources": 1,
            "metadata_modified": "2023-06-01"  
        }
        
        dataset = DatasetInfo(**data)
        assert dataset.metadata_modified == "2023-06-01"

    def test_model_defaults(self):
        """Test that model defaults are applied correctly"""
        # Test DatasetInfo defaults
        dataset = DatasetInfo(name="test", title="Test", notes="Test notes", resources=1)
        assert dataset.tags == []
        assert dataset.organization is None
        
        # Test GeneSearchRequest defaults
        request = GeneSearchRequest(gene_name="BRCA1")
        assert request.include_related_conditions is True