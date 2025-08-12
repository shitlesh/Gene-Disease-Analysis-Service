"""
Pydantic models for NHS Scotland Open Data API integration

These models define the request and response schemas for interacting with
NHS Scotland's health and medical datasets, providing validation and
serialization for gene-disease analysis support data.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class DatasetFormat(str, Enum):
    """Supported dataset file formats"""
    CSV = "CSV"
    JSON = "JSON"
    XML = "XML"
    PDF = "PDF"


class HealthCategory(str, Enum):
    """Categories of health data available"""
    CONGENITAL_CONDITIONS = "congenital_conditions"
    CANCER_DATA = "cancer_data"
    GENETIC_SCREENING = "genetic_screening"
    MORTALITY_DATA = "mortality_data"
    DISEASE_STATISTICS = "disease_statistics"


class DatasetResource(BaseModel):
    """
    Model representing a dataset resource (file/data source)
    Each dataset can have multiple resources (e.g., different years, regions)
    """
    name: str = Field(..., description="Resource name/title")
    description: Optional[str] = Field(None, description="Resource description")
    format: str = Field(..., description="File format (CSV, JSON, etc.)")
    url: str = Field(..., description="Direct URL to download the resource")
    created: Optional[str] = Field(None, description="Creation timestamp")
    last_modified: Optional[str] = Field(None, description="Last modification timestamp")
    size: Optional[int] = Field(None, description="Resource size in bytes")


class DatasetInfo(BaseModel):
    """
    Model for NHS Scotland dataset metadata
    Contains core information about available health datasets
    """
    name: str = Field(..., description="Dataset identifier")
    title: str = Field(..., description="Human-readable dataset title")
    notes: str = Field(..., description="Dataset description and context")
    tags: List[str] = Field(default=[], description="Keywords and categories")
    resources: int = Field(..., description="Number of available resources")
    organization: Optional[str] = Field(None, description="Publishing organization")
    metadata_created: Optional[str] = Field(None, description="Dataset creation date")
    metadata_modified: Optional[str] = Field(None, description="Last modification date")
    
    @validator('notes')
    def truncate_notes(cls, v):
        """Ensure description is not too long for frontend display"""
        return v[:500] + "..." if len(v) > 500 else v


class DatasetSearchResult(BaseModel):
    """
    Model for dataset search results with relevance scoring
    Used when searching for specific diseases or conditions
    """
    name: str = Field(..., description="Dataset identifier")
    title: str = Field(..., description="Dataset title")
    notes: str = Field(..., description="Brief description")
    score: float = Field(..., description="Search relevance score")
    tags: List[str] = Field(default=[], description="Associated tags")
    organization: Optional[str] = Field(None, description="Data source organization")
    resources: int = Field(..., description="Number of data files available")
    metadata_modified: Optional[str] = Field(None, description="Last update date")


class CongenitalConditionsData(BaseModel):
    """
    Model for congenital conditions dataset - most relevant for genetic analysis
    Contains information about birth defects and genetic conditions
    """
    dataset_info: DatasetInfo = Field(..., description="Basic dataset metadata")
    resources: List[DatasetResource] = Field(..., description="Available data files")
    conditions_categories: List[str] = Field(..., description="Types of conditions covered")
    data_years: str = Field(..., description="Time period covered")
    measurement_unit: str = Field(..., description="How rates are measured")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_info": {
                    "name": "congenital-conditions",
                    "title": "Congenital Conditions",
                    "notes": "Data on congenital conditions in Scotland",
                    "tags": ["genetics", "birth defects", "health"],
                    "resources": 4,
                    "organization": "Public Health Scotland"
                },
                "conditions_categories": [
                    "Cardiovascular conditions",
                    "Nervous system conditions",
                    "Genetic syndromes"
                ],
                "data_years": "2000-2022",
                "measurement_unit": "per 10,000 births"
            }
        }


class CancerIncidenceData(BaseModel):
    """
    Model for cancer incidence dataset - relevant for cancer genetics
    Contains cancer statistics that may correlate with genetic factors
    """
    dataset_info: DatasetInfo = Field(..., description="Basic dataset metadata")
    resources: List[DatasetResource] = Field(..., description="Available data files")
    cancer_types: List[str] = Field(..., description="Types of cancers tracked")
    data_years: str = Field(..., description="Years of data coverage")
    geographic_breakdown: List[str] = Field(..., description="Geographic data divisions")
    demographic_breakdown: List[str] = Field(..., description="Population breakdowns available")


class HealthStatisticsSummary(BaseModel):
    """
    Model for health statistics overview
    Provides summary of available health data categories
    """
    categories: Dict[str, List[DatasetSearchResult]] = Field(
        ..., 
        description="Health data organized by category"
    )
    total_datasets_found: int = Field(..., description="Total number of relevant datasets")
    last_updated: str = Field(..., description="When this summary was generated")
    data_source: str = Field(..., description="Source of the data")


# Request Models

class DiseaseSearchRequest(BaseModel):
    """
    Request model for searching disease-related datasets
    """
    disease_name: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="Name of disease to search for"
    )
    limit: Optional[int] = Field(
        20, 
        ge=1, 
        le=50,
        description="Maximum number of results to return"
    )
    
    @validator('disease_name')
    def sanitize_disease_name(cls, v):
        """Clean and standardize disease name for search"""
        return v.strip().lower()


class GeneSearchRequest(BaseModel):
    """
    Request model for searching gene-related health data
    Note: Direct gene datasets may not be available, but related health conditions are
    """
    gene_name: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Gene name to search for related conditions"
    )
    include_related_conditions: bool = Field(
        True,
        description="Whether to include conditions associated with this gene"
    )
    
    @validator('gene_name')
    def sanitize_gene_name(cls, v):
        """Standardize gene name format"""
        return v.strip().upper()  # Gene names are typically uppercase


class DatasetSearchRequest(BaseModel):
    """
    Generic request model for searching NHS Scotland datasets
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Search query for datasets"
    )
    limit: Optional[int] = Field(
        20,
        ge=1,
        le=50,
        description="Maximum number of results"
    )
    category: Optional[HealthCategory] = Field(
        None,
        description="Optional category filter"
    )


# Response Models

class NHSDataResponse(BaseModel):
    """
    Generic response wrapper for NHS Scotland data
    Provides consistent response structure with metadata
    """
    success: bool = Field(..., description="Whether the request was successful")
    data: Union[
        DatasetInfo,
        List[DatasetSearchResult],
        CongenitalConditionsData,
        CancerIncidenceData,
        HealthStatisticsSummary,
        Dict[str, Any]
    ] = Field(..., description="Response data payload")
    message: Optional[str] = Field(None, description="Additional information or error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    total_results: Optional[int] = Field(None, description="Total results available")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """
    Error response model for API failures
    """
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error description")


# Specialized response models for specific endpoints

class GeneRelatedDataResponse(BaseModel):
    """
    Response model for gene-related health data
    Since direct gene datasets aren't available, returns related health conditions
    """
    gene_name: str = Field(..., description="Queried gene name")
    related_datasets: List[DatasetSearchResult] = Field(
        ..., 
        description="Health datasets potentially related to this gene"
    )
    suggested_conditions: List[str] = Field(
        default=[],
        description="Known conditions associated with this gene"
    )
    total_datasets: int = Field(..., description="Number of related datasets found")
    search_strategy: str = Field(
        ...,
        description="Explanation of how related data was found"
    )


class DiseaseDataResponse(BaseModel):
    """
    Response model for disease-specific health data
    """
    disease_name: str = Field(..., description="Queried disease name")
    relevant_datasets: List[DatasetSearchResult] = Field(
        ...,
        description="Datasets containing information about this disease"
    )
    data_categories: List[str] = Field(
        default=[],
        description="Types of data available (incidence, mortality, etc.)"
    )
    geographic_coverage: List[str] = Field(
        default=["Scotland"],
        description="Geographic areas covered by the data"
    )
    total_datasets: int = Field(..., description="Number of relevant datasets found")


class AvailableDatasetsResponse(BaseModel):
    """
    Response model for listing all available datasets
    """
    datasets: List[DatasetInfo] = Field(..., description="All available NHS Scotland datasets")
    total_count: int = Field(..., description="Total number of datasets")
    categories: List[str] = Field(..., description="Available data categories")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this list was generated"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }