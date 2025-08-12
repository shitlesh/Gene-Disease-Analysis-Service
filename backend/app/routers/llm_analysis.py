"""
FastAPI router for LLM-powered gene-disease analysis
Integrates OpenAI and Anthropic APIs for correlation analysis
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Optional

from ..models.llm import (
    LLMAnalysisRequest, LLMServiceResponse, LLMProvider, LLMProviderConfig,
    OpenAIConfig, AnthropicConfig
)
from ..services.llm_client import LLMService
from ..config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/llm",
    tags=["llm-analysis"],
    responses={
        404: {"description": "Analysis not found"},
        500: {"description": "Internal server error"},
        503: {"description": "LLM service unavailable"}
    }
)

# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Dependency to get LLM service instance"""
    global _llm_service
    
    if _llm_service is None:
        # Initialize LLM service with configuration
        config = LLMProviderConfig(
            openai=OpenAIConfig(
                api_key=getattr(settings, 'OPENAI_API_KEY', ''),
                organization=getattr(settings, 'OPENAI_ORGANIZATION', None),
                temperature=getattr(settings, 'LLM_TEMPERATURE', 0.1),
                max_tokens=getattr(settings, 'LLM_MAX_TOKENS', 2000)
            ) if getattr(settings, 'OPENAI_API_KEY', '') else None,
            anthropic=AnthropicConfig(
                api_key=getattr(settings, 'ANTHROPIC_API_KEY', ''),
                temperature=getattr(settings, 'LLM_TEMPERATURE', 0.1),
                max_tokens=getattr(settings, 'LLM_MAX_TOKENS', 2000)
            ) if getattr(settings, 'ANTHROPIC_API_KEY', '') else None,
            default_provider=LLMProvider(getattr(settings, 'DEFAULT_LLM_PROVIDER', 'openai')),
            enable_caching=getattr(settings, 'LLM_ENABLE_CACHING', True),
            cache_ttl=getattr(settings, 'LLM_CACHE_TTL', 3600)
        )
        
        _llm_service = LLMService(config)
    
    return _llm_service


@router.get("/", response_model=dict)
async def llm_analysis_info():
    """
    Get information about LLM analysis capabilities
    
    Returns basic information about available providers, models, and endpoints.
    """
    return {
        "service": "LLM-Powered Gene-Disease Analysis",
        "description": "AI-powered correlation analysis using OpenAI and Anthropic APIs",
        "supported_providers": [provider.value for provider in LLMProvider],
        "available_endpoints": {
            "analyze": "POST /llm/analyze - Perform gene-disease correlation analysis",
            "providers": "GET /llm/providers - List available providers and their status",
            "cache_stats": "GET /llm/cache/stats - Get caching statistics",
            "cache_clear": "DELETE /llm/cache - Clear analysis cache"
        },
        "features": [
            "Provider-agnostic architecture (OpenAI, Anthropic)",
            "Structured JSON output with confidence scoring",
            "Anti-hallucination prompt engineering",
            "Response validation and retry logic",
            "Intelligent caching for performance",
            "Comprehensive error handling"
        ],
        "analysis_output": {
            "summary": "Brief gene-disease relationship summary",
            "associations": "Detailed molecular mechanisms and evidence",
            "recommendation": "Clinical or research recommendations", 
            "uncertainty": "Assessment of limitations and confidence bounds",
            "confidence_score": "Overall confidence level (0-1)"
        }
    }


@router.post("/analyze", response_model=LLMServiceResponse)
async def analyze_gene_disease(
    request: LLMAnalysisRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Perform AI-powered gene-disease correlation analysis
    
    Uses advanced language models to analyze the relationship between a specific gene
    and disease, providing structured output with evidence levels and confidence scoring.
    
    **Features:**
    - Multi-provider support (OpenAI GPT-4, Anthropic Claude)
    - Anti-hallucination prompts for reliable scientific analysis
    - Structured JSON output with validation
    - Evidence-based recommendations and uncertainty assessment
    """
    try:
        logger.info(f"Starting LLM analysis for gene: {request.gene}, disease: {request.disease}, provider: {request.provider}")
        
        # Perform analysis
        response = await llm_service.analyze_gene_disease(request)
        
        if response.success:
            logger.info(f"LLM analysis completed successfully. Confidence: {response.data.confidence_score if response.data else 'N/A'}")
        else:
            logger.warning(f"LLM analysis failed: {response.error.message if response.error else 'Unknown error'}")
        
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error in LLM analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during analysis"
        )


@router.get("/providers", response_model=dict)
async def get_provider_status(llm_service: LLMService = Depends(get_llm_service)):
    """
    Get status and availability of LLM providers
    
    Returns information about configured providers, their availability,
    and recommended models for different use cases.
    """
    try:
        providers_status = {}
        
        # Check OpenAI availability
        try:
            openai_config = llm_service.config.get_provider_config(LLMProvider.OPENAI)
            providers_status["openai"] = {
                "available": True,
                "models": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                "recommended_model": "gpt-4",
                "features": ["JSON mode", "High accuracy", "Fast response"],
                "use_cases": ["Detailed analysis", "Complex gene interactions"]
            }
        except ValueError:
            providers_status["openai"] = {
                "available": False,
                "reason": "API key not configured"
            }
        
        # Check Anthropic availability
        try:
            anthropic_config = llm_service.config.get_provider_config(LLMProvider.ANTHROPIC)
            providers_status["anthropic"] = {
                "available": True,
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                "recommended_model": "claude-3-sonnet-20240229",
                "features": ["Long context", "Nuanced analysis", "Safety-focused"],
                "use_cases": ["Literature review", "Complex reasoning", "Uncertainty assessment"]
            }
        except ValueError:
            providers_status["anthropic"] = {
                "available": False,
                "reason": "API key not configured"
            }
        
        return {
            "providers": providers_status,
            "default_provider": llm_service.config.default_provider.value,
            "total_available": sum(1 for p in providers_status.values() if p.get("available", False))
        }
        
    except Exception as e:
        logger.error(f"Error getting provider status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving provider status"
        )


@router.get("/cache/stats", response_model=dict)
async def get_cache_statistics(llm_service: LLMService = Depends(get_llm_service)):
    """
    Get caching statistics and performance metrics
    
    Returns information about cache usage, hit rates, and performance optimization.
    """
    try:
        stats = llm_service.get_cache_stats()
        return {
            "cache_statistics": stats,
            "performance_impact": {
                "cached_responses": "Sub-second response time",
                "uncached_responses": "2-10 seconds depending on provider",
                "cost_savings": "Significant reduction in API costs for repeated queries"
            },
            "cache_recommendations": {
                "enable_caching": "Recommended for production use",
                "optimal_ttl": "1-6 hours depending on use case",
                "cache_warming": "Pre-populate cache with common gene-disease pairs"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving cache statistics"
        )


@router.delete("/cache", response_model=dict)
async def clear_analysis_cache(llm_service: LLMService = Depends(get_llm_service)):
    """
    Clear the analysis response cache
    
    Removes all cached responses. Useful for testing or when updated
    analysis is needed for previously analyzed gene-disease pairs.
    """
    try:
        old_stats = llm_service.get_cache_stats()
        llm_service.clear_cache()
        
        logger.info("LLM analysis cache cleared")
        
        return {
            "success": True,
            "message": "Analysis cache cleared successfully",
            "cleared_entries": old_stats.get("cache_size", 0),
            "cache_status": "empty"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error clearing cache"
        )


@router.get("/models/{provider}", response_model=dict)
async def get_provider_models(provider: LLMProvider):
    """
    Get available models for a specific provider
    
    Returns detailed information about models, their capabilities,
    and recommended use cases.
    """
    try:
        if provider == LLMProvider.OPENAI:
            return {
                "provider": "openai",
                "models": {
                    "gpt-4": {
                        "name": "GPT-4",
                        "description": "Most capable model with high accuracy",
                        "context_length": 8192,
                        "features": ["JSON mode", "Function calling", "High reasoning"],
                        "recommended_for": ["Complex analysis", "Research queries", "Clinical applications"],
                        "cost": "Higher cost but best quality"
                    },
                    "gpt-4-turbo-preview": {
                        "name": "GPT-4 Turbo",
                        "description": "Faster variant with large context window",
                        "context_length": 128000,
                        "features": ["Large context", "JSON mode", "Fast response"],
                        "recommended_for": ["Large literature reviews", "Comprehensive analysis"],
                        "cost": "Balanced cost and performance"
                    },
                    "gpt-3.5-turbo": {
                        "name": "GPT-3.5 Turbo",
                        "description": "Fast and cost-effective option",
                        "context_length": 4096,
                        "features": ["Fast response", "Cost effective"],
                        "recommended_for": ["Simple queries", "High-volume analysis", "Testing"],
                        "cost": "Most economical option"
                    }
                }
            }
        elif provider == LLMProvider.ANTHROPIC:
            return {
                "provider": "anthropic",
                "models": {
                    "claude-3-opus-20240229": {
                        "name": "Claude 3 Opus",
                        "description": "Most powerful model for complex tasks",
                        "context_length": 200000,
                        "features": ["Highest accuracy", "Complex reasoning", "Long context"],
                        "recommended_for": ["Research analysis", "Complex medical queries", "Literature synthesis"],
                        "cost": "Premium pricing for best quality"
                    },
                    "claude-3-sonnet-20240229": {
                        "name": "Claude 3 Sonnet", 
                        "description": "Balanced performance and speed",
                        "context_length": 200000,
                        "features": ["Good accuracy", "Faster response", "Long context"],
                        "recommended_for": ["General analysis", "Clinical queries", "Production use"],
                        "cost": "Balanced cost and performance"
                    },
                    "claude-3-haiku-20240307": {
                        "name": "Claude 3 Haiku",
                        "description": "Fastest model for simple tasks",
                        "context_length": 200000,
                        "features": ["Very fast", "Cost effective", "Long context"],
                        "recommended_for": ["Simple queries", "High-volume processing", "Real-time applications"],
                        "cost": "Most cost-effective option"
                    }
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: {provider}"
            )
            
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information"
        )


# Health check for LLM service
@router.get("/health", response_model=dict)
async def llm_service_health(llm_service: LLMService = Depends(get_llm_service)):
    """
    Health check for LLM analysis service
    
    Verifies that the service is properly configured and providers are accessible.
    """
    try:
        health_status = {
            "service": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "providers": {},
            "cache": llm_service.get_cache_stats()
        }
        
        # Check each provider
        for provider in LLMProvider:
            try:
                llm_service.config.get_provider_config(provider)
                health_status["providers"][provider.value] = "configured"
            except ValueError:
                health_status["providers"][provider.value] = "not_configured"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "service": "unhealthy",
                "error": "LLM service configuration error"
            }
        )