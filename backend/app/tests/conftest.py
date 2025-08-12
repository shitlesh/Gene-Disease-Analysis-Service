"""
Pytest configuration and shared fixtures for NHS Scotland integration tests
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing HTTP requests"""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": []}
    mock_client.get.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_nhs_response():
    """Sample NHS Scotland API response for testing"""
    return {
        "help": "https://opendata.nhs.scot/api/3/action/package_list",
        "success": True,
        "result": [
            {
                "author": "NHS Scotland",
                "author_email": "opendata@nhs.scot",
                "creator_user_id": "test-user-id",
                "id": "12345678-1234-5678-9012-123456789abc",
                "isopen": True,
                "license_id": "uk-ogl",
                "license_title": "UK Open Government Licence",
                "maintainer": "NHS Scotland Open Data Team",
                "maintainer_email": "opendata@nhs.scot",
                "metadata_created": "2023-01-15T10:30:00.123456",
                "metadata_modified": "2023-06-20T14:45:30.654321",
                "name": "congenital-anomalies-scotland",
                "notes": "This dataset contains information about congenital anomalies (birth defects) recorded in Scotland. The data includes various types of congenital conditions and their prevalence rates across different health boards.",
                "num_resources": 3,
                "num_tags": 5,
                "organization": {
                    "id": "nhs-scotland-org-id",
                    "name": "nhs-scotland",
                    "title": "NHS Scotland",
                    "type": "organization",
                    "description": "National Health Service Scotland",
                    "image_url": "",
                    "created": "2020-01-01T00:00:00.000000",
                    "is_organization": True,
                    "approval_status": "approved",
                    "state": "active"
                },
                "owner_org": "nhs-scotland-org-id",
                "private": False,
                "state": "active",
                "title": "Congenital Anomalies in Scotland",
                "type": "dataset",
                "url": "",
                "version": "1.2",
                "resources": [
                    {
                        "id": "resource-1-id",
                        "name": "Congenital Anomalies Data 2020-2023",
                        "description": "Main dataset file",
                        "format": "CSV",
                        "url": "https://www.opendata.nhs.scot/dataset/congenital-anomalies/resource/resource-1-id",
                        "created": "2023-01-15T10:30:00.123456",
                        "last_modified": "2023-06-20T14:45:30.654321",
                        "size": 1024000
                    }
                ],
                "tags": [
                    {
                        "id": "tag-genetics-id",
                        "name": "genetics",
                        "display_name": "Genetics",
                        "state": "active",
                        "vocabulary_id": None
                    },
                    {
                        "id": "tag-congenital-id", 
                        "name": "congenital",
                        "display_name": "Congenital",
                        "state": "active",
                        "vocabulary_id": None
                    },
                    {
                        "id": "tag-birth-defects-id",
                        "name": "birth-defects",
                        "display_name": "Birth Defects", 
                        "state": "active",
                        "vocabulary_id": None
                    }
                ],
                "groups": [],
                "relationships_as_subject": [],
                "relationships_as_object": []
            },
            {
                "id": "87654321-4321-8765-2109-876543210fed",
                "name": "cancer-incidence-scotland",
                "title": "Cancer Incidence in Scotland",
                "notes": "Annual cancer incidence statistics for Scotland, including data on different cancer types, age groups, and geographic distribution.",
                "author": "NHS Scotland",
                "maintainer": "Scottish Cancer Registry",
                "metadata_created": "2022-03-10T09:15:00.000000",
                "metadata_modified": "2023-07-15T16:20:00.000000",
                "organization": {
                    "name": "nhs-scotland",
                    "title": "NHS Scotland"
                },
                "tags": [
                    {"name": "cancer", "display_name": "Cancer"},
                    {"name": "incidence", "display_name": "Incidence"},
                    {"name": "oncology", "display_name": "Oncology"}
                ],
                "resources": [
                    {
                        "id": "cancer-resource-id",
                        "name": "Cancer Incidence Data",
                        "format": "CSV",
                        "url": "https://www.opendata.nhs.scot/dataset/cancer-incidence/resource/cancer-resource-id"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_enhanced_dataset():
    """Sample enhanced dataset for testing transformations"""
    return {
        "name": "test-genetic-dataset",
        "title": "Genetic Conditions Test Dataset", 
        "notes": "This dataset contains information about hereditary genetic conditions and inherited disorders affecting the Scottish population.",
        "tags": ["genetics", "hereditary", "conditions"],
        "metadata_created": "2023-01-01T00:00:00Z",
        "metadata_modified": "2023-06-01T12:00:00Z",
        "organization": "NHS Scotland",
        "genetic_relevance": 0.85,
        "disease_categories": ["Genetic Disorders", "Inherited Conditions"],
        "keywords": ["genetic", "hereditary", "inherited", "disorders", "conditions"],
        "metadata_created_formatted": "2023-01-01",
        "metadata_modified_formatted": "2023-06-01"
    }


@pytest.fixture
def mock_gene_conditions_mapping():
    """Mock gene-condition mapping for testing"""
    return {
        "BRCA1": ["breast cancer", "ovarian cancer", "hereditary cancer syndrome"],
        "BRCA2": ["breast cancer", "ovarian cancer", "prostate cancer"],
        "CFTR": ["cystic fibrosis", "respiratory disorders"],
        "TP53": ["li-fraumeni syndrome", "various cancers", "tumor suppressor disorders"]
    }


@pytest.fixture
def mock_disease_synonyms():
    """Mock disease synonyms for testing"""
    return {
        "cancer": ["carcinoma", "malignancy", "tumor", "neoplasm"],
        "heart disease": ["cardiovascular disease", "cardiac disorder", "coronary disease"],
        "diabetes": ["diabetes mellitus", "diabetic condition", "glucose disorder"]
    }


@pytest.fixture
def test_client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_nhs_service_success():
    """Mock NHS Scotland service with successful responses"""
    mock_service = AsyncMock()
    
    # Mock successful API responses
    mock_service.get_available_datasets.return_value = [
        {
            "name": "test-dataset",
            "title": "Test Dataset",
            "notes": "Test description",
            "tags": ["test"],
            "organization": "NHS Scotland"
        }
    ]
    
    mock_service.search_datasets.return_value = [
        {
            "name": "search-result",
            "title": "Search Result",
            "notes": "Search result description",
            "tags": ["search"],
            "organization": "NHS Scotland"
        }
    ]
    
    mock_service.get_congenital_conditions_data.return_value = {
        "name": "congenital-conditions",
        "title": "Congenital Conditions",
        "description": "Birth defects data"
    }
    
    mock_service.get_cancer_incidence_data.return_value = {
        "name": "cancer-incidence", 
        "title": "Cancer Incidence",
        "description": "Cancer statistics"
    }
    
    return mock_service


@pytest.fixture
def mock_nhs_service_error():
    """Mock NHS Scotland service with error responses"""
    from app.services.nhs_scotland_service import NHSScotlandAPIError
    
    mock_service = AsyncMock()
    mock_service.get_available_datasets.side_effect = NHSScotlandAPIError("API Error")
    mock_service.search_datasets.side_effect = NHSScotlandAPIError("Search Error")
    
    return mock_service


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset any caches between tests"""
    # This fixture runs automatically before each test
    # Add any cache reset logic here if needed
    yield
    # Cleanup after test if needed


class MockAsyncContextManager:
    """Mock async context manager for testing"""
    
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass