"""
FastAPI routers for NHS Scotland Open Data integration

This module provides REST endpoints for accessing NHS Scotland health data
to support gene-disease analysis. While direct genomic data isn't available,
these endpoints provide access to related health statistics and medical conditions.
"""

from fastapi import APIRouter, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging

from ..models.nhs_scotland import (
    NHSDataResponse, ErrorResponse, GeneRelatedDataResponse, DiseaseDataResponse,
    AvailableDatasetsResponse, CongenitalConditionsData, CancerIncidenceData,
    HealthStatisticsSummary, GeneSearchRequest, DiseaseSearchRequest, DatasetSearchRequest
)
from ..services.nhs_scotland_service import nhs_scotland_service, NHSScotlandAPIError
from ..utils.data_transformer import data_transformer

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/nhs",
    tags=["nhs-scotland"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


async def handle_api_error(error: Exception, context: str) -> JSONResponse:
    """
    Centralized error handling for NHS Scotland API calls
    
    Args:
        error: The exception that occurred
        context: Description of what operation failed
        
    Returns:
        JSONResponse with appropriate error details
    """
    if isinstance(error, NHSScotlandAPIError):
        logger.warning(f"NHS Scotland API error in {context}: {error}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="NHS_API_ERROR",
                message=f"NHS Scotland API error: {str(error)}"
            ).dict()
        )
    
    logger.error(f"Unexpected error in {context}: {error}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR", 
            message=f"Internal server error in {context}"
        ).dict()
    )


@router.get("/", response_model=dict)
async def nhs_scotland_info():
    """
    Get information about NHS Scotland integration capabilities
    
    Returns basic information about what data is available and how to use the endpoints.
    """
    return {
        "service": "NHS Scotland Open Data Integration",
        "description": "Access to Scottish health datasets for gene-disease analysis support",
        "available_endpoints": {
            "datasets": "GET /nhs/datasets - List all available datasets",
            "gene_search": "GET /nhs/gene/{gene_name} - Find data related to a specific gene",
            "disease_search": "GET /nhs/disease/{disease_name} - Find data related to a disease",
            "congenital": "GET /nhs/congenital-conditions - Genetic conditions data",
            "cancer": "GET /nhs/cancer-incidence - Cancer statistics",
            "search": "GET /nhs/search - Search across all datasets",
            "health_summary": "GET /nhs/health-summary - Overview of available health data"
        },
        "data_limitations": [
            "No direct genetic sequence data available",
            "Individual patient records not accessible",
            "Data is Scotland-specific",
            "Some datasets may have temporal limitations"
        ],
        "best_for": [
            "Population health statistics",
            "Disease incidence and prevalence",
            "Congenital condition rates",
            "Cancer epidemiology",
            "Public health surveillance data"
        ]
    }


@router.get("/datasets", response_model=AvailableDatasetsResponse)
async def get_available_datasets():
    """
    Retrieve all available NHS Scotland datasets
    
    Returns a comprehensive list of datasets available through the NHS Scotland
    open data platform, enhanced with genetic relevance scoring.
    """
    try:
        logger.info("Fetching available NHS Scotland datasets")
        
        # Get raw datasets from NHS Scotland API
        raw_datasets = await nhs_scotland_service.get_available_datasets()
        
        # Transform and enhance dataset information
        enhanced_datasets = []
        categories = set()
        
        for dataset in raw_datasets:
            enhanced = data_transformer.enhance_dataset_info(dataset)
            enhanced_datasets.append(enhanced)
            categories.update(enhanced.get('disease_categories', []))
        
        # Format for frontend consumption
        formatted_datasets = data_transformer.format_for_frontend(enhanced_datasets)
        
        return AvailableDatasetsResponse(
            datasets=formatted_datasets,
            total_count=len(formatted_datasets),
            categories=sorted(list(categories))
        )
        
    except Exception as e:
        return await handle_api_error(e, "get_available_datasets")


@router.get("/gene/{gene_name}", response_model=GeneRelatedDataResponse)
async def get_gene_related_data(
    gene_name: str = Path(..., description="Gene name to search for", min_length=2, max_length=20)
):
    """
    Find health datasets related to a specific gene
    
    Since direct gene datasets aren't available in NHS Scotland open data,
    this endpoint searches for health conditions and datasets that are
    associated with the specified gene.
    """
    try:
        logger.info(f"Searching for data related to gene: {gene_name}")
        
        # Clean and standardize gene name
        clean_gene = gene_name.strip().upper()
        
        # Find known conditions associated with this gene
        related_conditions = data_transformer.find_gene_related_conditions(clean_gene)
        
        # Search for datasets using gene name and related conditions
        all_datasets = []
        search_terms = [clean_gene] + related_conditions
        
        for term in search_terms[:5]:  # Limit searches to avoid overwhelming API
            try:
                results = await nhs_scotland_service.search_datasets(term, limit=10)
                enhanced_results = [data_transformer.enhance_dataset_info(r) for r in results]
                all_datasets.extend(enhanced_results)
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {e}")
                continue
        
        # Remove duplicates and filter by relevance
        unique_datasets = {}
        for dataset in all_datasets:
            name = dataset.get('name')
            if name and name not in unique_datasets:
                unique_datasets[name] = dataset
        
        relevant_datasets = data_transformer.filter_datasets_by_relevance(
            list(unique_datasets.values()), 
            min_relevance=0.05  # Lower threshold for gene searches
        )
        
        formatted_datasets = data_transformer.format_for_frontend(relevant_datasets)
        
        # Create search strategy explanation
        search_strategy = f"Searched for '{clean_gene}' and related conditions"
        if related_conditions:
            search_strategy += f" including: {', '.join(related_conditions[:3])}"
        
        return GeneRelatedDataResponse(
            gene_name=clean_gene,
            related_datasets=formatted_datasets,
            suggested_conditions=related_conditions,
            total_datasets=len(formatted_datasets),
            search_strategy=search_strategy
        )
        
    except Exception as e:
        return await handle_api_error(e, f"get_gene_related_data for {gene_name}")


@router.get("/disease/{disease_name}", response_model=DiseaseDataResponse)
async def get_disease_related_data(
    disease_name: str = Path(..., description="Disease name to search for", min_length=2, max_length=100)
):
    """
    Find health datasets related to a specific disease
    
    Searches NHS Scotland open data for datasets containing information
    about the specified disease, including synonyms and related terms.
    """
    try:
        logger.info(f"Searching for data related to disease: {disease_name}")
        
        # Clean and expand search terms
        clean_disease = disease_name.strip().lower()
        expanded_terms = data_transformer.expand_disease_search_terms(clean_disease)
        
        # Search using expanded terms
        all_results = []
        for term in expanded_terms[:8]:  # Limit to prevent API overload
            try:
                results = await nhs_scotland_service.search_datasets(term, limit=10)
                enhanced_results = [data_transformer.enhance_dataset_info(r) for r in results]
                all_results.extend(enhanced_results)
            except Exception as e:
                logger.warning(f"Search failed for disease term '{term}': {e}")
                continue
        
        # Remove duplicates based on dataset name
        unique_datasets = {}
        for dataset in all_results:
            name = dataset.get('name')
            if name and name not in unique_datasets:
                unique_datasets[name] = dataset
        
        # Filter and sort by relevance
        relevant_datasets = data_transformer.filter_datasets_by_relevance(
            list(unique_datasets.values()), 
            min_relevance=0.1
        )
        
        formatted_datasets = data_transformer.format_for_frontend(relevant_datasets)
        
        # Identify data categories
        data_categories = set()
        geographic_coverage = {"Scotland"}  # NHS Scotland is Scotland-specific
        
        for dataset in relevant_datasets:
            data_categories.update(dataset.get('disease_categories', []))
        
        return DiseaseDataResponse(
            disease_name=clean_disease.title(),
            relevant_datasets=formatted_datasets,
            data_categories=sorted(list(data_categories)),
            geographic_coverage=list(geographic_coverage),
            total_datasets=len(formatted_datasets)
        )
        
    except Exception as e:
        return await handle_api_error(e, f"get_disease_related_data for {disease_name}")


@router.get("/congenital-conditions", response_model=CongenitalConditionsData)
async def get_congenital_conditions():
    """
    Get congenital conditions dataset - most relevant for genetic analysis
    
    Returns information about the congenital conditions dataset, which contains
    data about birth defects and genetic conditions in Scotland.
    """
    try:
        logger.info("Fetching congenital conditions dataset")
        
        conditions_data = await nhs_scotland_service.get_congenital_conditions_data()
        return CongenitalConditionsData(**conditions_data)
        
    except Exception as e:
        return await handle_api_error(e, "get_congenital_conditions")


@router.get("/cancer-incidence", response_model=CancerIncidenceData)
async def get_cancer_incidence():
    """
    Get cancer incidence dataset - relevant for cancer genetics research
    
    Returns information about the cancer incidence dataset, which contains
    cancer statistics that may be useful for genetic cancer research.
    """
    try:
        logger.info("Fetching cancer incidence dataset")
        
        cancer_data = await nhs_scotland_service.get_cancer_incidence_data()
        return CancerIncidenceData(**cancer_data)
        
    except Exception as e:
        return await handle_api_error(e, "get_cancer_incidence")


@router.get("/search", response_model=NHSDataResponse)
async def search_datasets(
    query: str = Query(..., description="Search query", min_length=2, max_length=200),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    min_relevance: Optional[float] = Query(0.1, ge=0.0, le=1.0, description="Minimum genetic relevance score")
):
    """
    Search across all NHS Scotland datasets
    
    Performs a general search across all available datasets with optional
    filtering by genetic relevance score.
    """
    try:
        logger.info(f"Performing dataset search for query: '{query}'")
        
        # Search datasets
        search_results = await nhs_scotland_service.search_datasets(query, limit=limit)
        
        # Enhance results with genetic relevance scoring
        enhanced_results = []
        for result in search_results:
            enhanced = data_transformer.enhance_dataset_info(result)
            enhanced_results.append(enhanced)
        
        # Filter by relevance if specified
        if min_relevance > 0:
            enhanced_results = data_transformer.filter_datasets_by_relevance(
                enhanced_results, 
                min_relevance
            )
        
        # Format for frontend
        formatted_results = data_transformer.format_for_frontend(enhanced_results)
        
        return NHSDataResponse(
            success=True,
            data=formatted_results,
            message=f"Found {len(formatted_results)} datasets matching '{query}'",
            total_results=len(formatted_results)
        )
        
    except Exception as e:
        return await handle_api_error(e, f"search_datasets for '{query}'")


@router.get("/health-summary", response_model=HealthStatisticsSummary)
async def get_health_statistics_summary():
    """
    Get overview of available health data categories
    
    Returns a comprehensive summary of the types of health data available
    through NHS Scotland, organized by category.
    """
    try:
        logger.info("Generating health statistics summary")
        
        summary_data = await nhs_scotland_service.get_health_statistics_summary()
        
        # Enhance categories with additional metadata
        enhanced_categories = {}
        for category, datasets in summary_data.get('categories', {}).items():
            enhanced_datasets = []
            for dataset in datasets:
                enhanced = data_transformer.enhance_dataset_info(dataset)
                enhanced_datasets.append(enhanced)
            
            # Format for frontend
            enhanced_categories[category] = data_transformer.format_for_frontend(enhanced_datasets)
        
        return HealthStatisticsSummary(
            categories=enhanced_categories,
            total_datasets_found=summary_data.get('total_datasets_found', 0),
            last_updated=summary_data.get('last_updated', ''),
            data_source=summary_data.get('data_source', 'NHS Scotland Open Data Platform')
        )
        
    except Exception as e:
        return await handle_api_error(e, "get_health_statistics_summary")


@router.get("/analysis-suggestions")
async def get_analysis_suggestions(
    gene_name: Optional[str] = Query(None, description="Gene name for suggestions"),
    disease_name: Optional[str] = Query(None, description="Disease name for suggestions")
):
    """
    Get analysis suggestions based on gene or disease input
    
    Provides suggestions for how to approach gene-disease analysis using
    available NHS Scotland datasets, including search strategies and limitations.
    """
    try:
        if not gene_name and not disease_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either gene_name or disease_name must be provided"
            )
        
        logger.info(f"Generating analysis suggestions for gene: {gene_name}, disease: {disease_name}")
        
        suggestions = data_transformer.create_analysis_suggestions(
            gene_name=gene_name,
            disease_name=disease_name
        )
        
        return NHSDataResponse(
            success=True,
            data=suggestions,
            message="Analysis suggestions generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return await handle_api_error(e, "get_analysis_suggestions")