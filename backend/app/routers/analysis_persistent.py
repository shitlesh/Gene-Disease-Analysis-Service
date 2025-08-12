"""
FastAPI router for persistent analysis operations using database storage
Provides endpoints for creating, tracking, and retrieving analyses with full persistence
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from ..db.session import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.crud import UserSessionCRUD, AnalysisCRUD, AnalysisChunkCRUD, AnalyticsQueries
from ..db.models import LLMProviderType, AnalysisStatus
from ..models.llm import LLMAnalysisRequest
from ..services.worker_pool import submit_llm_job, get_job_status, get_pool_status, JobPriority

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/persistent",
    tags=["persistent-analysis"],
    responses={404: {"description": "Not found"}}
)


# Pydantic models for API requests/responses

class CreateSessionRequest(BaseModel):
    """Request model for creating a user session"""
    username: str = Field(..., min_length=1, max_length=255)
    provider: str = Field(default="internal", max_length=50)
    external_id: Optional[str] = Field(default=None, max_length=255)
    preferences: Optional[Dict[str, Any]] = Field(default={})


class SessionResponse(BaseModel):
    """Response model for user session"""
    id: int
    username: str
    provider: str
    session_token: Optional[str] = None
    created_at: datetime
    last_active_at: datetime
    total_analyses: int
    total_tokens_used: int
    is_active: bool


class CreateAnalysisRequest(BaseModel):
    """Request model for creating an analysis"""
    session_id: int
    gene: str = Field(..., min_length=1, max_length=100)
    disease: str = Field(..., min_length=1, max_length=500)
    provider: LLMProviderType
    model: str = Field(..., max_length=100)
    context: Optional[str] = None
    priority: Optional[str] = Field(default="normal", description="Job priority: low, normal, high, critical")
    metadata: Optional[Dict[str, Any]] = Field(default={})


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    id: int
    session_id: int
    gene: str
    disease: str
    provider: str
    model: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    result_summary: Optional[str] = None
    confidence_score: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class AnalysisHistoryResponse(BaseModel):
    """Response model for analysis history"""
    analyses: List[AnalysisResponse]
    total_count: int
    page: int
    page_size: int


class UsageStatsResponse(BaseModel):
    """Response model for usage statistics"""
    period_days: int
    total_analyses: int
    completed_analyses: int
    success_rate: float
    average_processing_time: float
    total_tokens_used: int
    popular_genes: List[Dict[str, Any]]
    popular_diseases: List[Dict[str, Any]]


# Session Management Endpoints

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    db: AsyncSession = Depends(get_async_session)
):
    """
    Create a new user session for persistent analysis tracking
    
    Creates a session that will track all analyses performed by the user,
    including usage statistics, preferences, and analysis history.
    """
    try:
        # Check if user already has an active session
        existing_session = await UserSessionCRUD.get_session_by_username(
            session=db,
            username=request.username,
            provider=request.provider
        )
        
        if existing_session:
            # Update last active time
            await UserSessionCRUD.update_last_active(db, existing_session.id)
            
            return SessionResponse(
                id=existing_session.id,
                username=existing_session.username,
                provider=existing_session.provider,
                session_token=existing_session.session_token,
                created_at=existing_session.created_at,
                last_active_at=existing_session.last_active_at,
                total_analyses=existing_session.total_analyses,
                total_tokens_used=existing_session.total_tokens_used,
                is_active=existing_session.is_active
            )
        
        # Create new session
        user_session = await UserSessionCRUD.create_session(
            session=db,
            username=request.username,
            provider=request.provider,
            external_id=request.external_id,
            preferences=request.preferences
        )
        
        logger.info(f"Created new user session for {request.username} (ID: {user_session.id})")
        
        return SessionResponse(
            id=user_session.id,
            username=user_session.username,
            provider=user_session.provider,
            session_token=user_session.session_token,
            created_at=user_session.created_at,
            last_active_at=user_session.last_active_at,
            total_analyses=user_session.total_analyses,
            total_tokens_used=user_session.total_tokens_used,
            is_active=user_session.is_active
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user session"
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    """Get user session by ID"""
    try:
        user_session = await UserSessionCRUD.get_session_by_id(db, session_id)
        
        if not user_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return SessionResponse(
            id=user_session.id,
            username=user_session.username,
            provider=user_session.provider,
            session_token=user_session.session_token,
            created_at=user_session.created_at,
            last_active_at=user_session.last_active_at,
            total_analyses=user_session.total_analyses,
            total_tokens_used=user_session.total_tokens_used,
            is_active=user_session.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )


# Analysis Management Endpoints

@router.post("/analyses", response_model=AnalysisResponse)
async def create_analysis(
    request: CreateAnalysisRequest,
    db: AsyncSession = Depends(get_async_session)
):
    """
    Create a new persistent analysis
    
    Creates an analysis entry in the database and submits it to the worker pool
    for processing. The analysis will be tracked through all stages of processing
    with full persistence and streaming chunk support.
    """
    try:
        # Verify session exists
        user_session = await UserSessionCRUD.get_session_by_id(db, request.session_id)
        if not user_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {request.session_id} not found"
            )
        
        # Generate prompt (simplified for this example)
        from ..services.prompt_templates import get_analysis_prompt
        system_message, user_message = get_analysis_prompt(
            gene=request.gene,
            disease=request.disease,
            context=request.context
        )
        prompt_text = f"System: {system_message}\n\nUser: {user_message}"
        
        # Create analysis in database
        analysis = await AnalysisCRUD.create_analysis(
            session=db,
            session_id=request.session_id,
            gene=request.gene,
            disease=request.disease,
            provider=request.provider,
            model=request.model,
            prompt_text=prompt_text,
            context=request.context,
            metadata=request.metadata
        )
        
        # Parse priority
        priority_mapping = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "critical": JobPriority.CRITICAL
        }
        job_priority = priority_mapping.get(request.priority.lower(), JobPriority.NORMAL)
        
        # Submit to worker pool
        llm_request = LLMAnalysisRequest(
            gene=request.gene,
            disease=request.disease,
            provider=request.provider,
            model=request.model,
            context=request.context
        )
        
        job_id = await submit_llm_job(
            request=llm_request,
            session_id=f"db_{request.session_id}",
            priority=job_priority
        )
        
        # Update analysis with job ID in metadata
        analysis.analysis_metadata["worker_job_id"] = job_id
        await db.commit()
        
        logger.info(f"Created analysis {analysis.id} and submitted worker job {job_id}")
        
        return AnalysisResponse(
            id=analysis.id,
            session_id=analysis.session_id,
            gene=analysis.gene,
            disease=analysis.disease,
            provider=analysis.provider.value,
            model=analysis.model,
            status=analysis.status.value,
            created_at=analysis.created_at,
            started_at=analysis.started_at,
            completed_at=analysis.completed_at,
            processing_time_seconds=analysis.processing_time_seconds,
            result_summary=analysis.result_summary,
            confidence_score=analysis.confidence_score,
            input_tokens=analysis.input_tokens,
            output_tokens=analysis.output_tokens,
            total_tokens=analysis.total_tokens,
            estimated_cost=analysis.estimated_cost,
            error_message=analysis.error_message,
            retry_count=analysis.retry_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create analysis"
        )


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    """Get analysis by ID with current status"""
    try:
        analysis = await AnalysisCRUD.get_analysis_by_id(db, analysis_id)
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_id} not found"
            )
        
        # Check worker job status if available
        if "worker_job_id" in analysis.analysis_metadata:
            job_id = analysis.analysis_metadata["worker_job_id"]
            try:
                job_status = await get_job_status(job_id)
                if job_status and job_status.status.value != analysis.status.value:
                    # Sync database with worker status
                    if job_status.status.value == "completed" and job_status.result:
                        await AnalysisCRUD.complete_analysis(
                            session=db,
                            analysis_id=analysis.id,
                            result_summary=job_status.result.data.summary,
                            confidence_score=job_status.result.data.confidence_score,
                            total_tokens=job_status.result.usage_stats.total_tokens if job_status.result.usage_stats else None
                        )
                        # Refresh analysis
                        analysis = await AnalysisCRUD.get_analysis_by_id(db, analysis_id)
                    elif job_status.status.value == "failed":
                        await AnalysisCRUD.fail_analysis(
                            session=db,
                            analysis_id=analysis.id,
                            error_message=job_status.error or "Job failed"
                        )
                        # Refresh analysis
                        analysis = await AnalysisCRUD.get_analysis_by_id(db, analysis_id)
            except Exception as e:
                logger.warning(f"Failed to sync worker job status: {e}")
        
        return AnalysisResponse(
            id=analysis.id,
            session_id=analysis.session_id,
            gene=analysis.gene,
            disease=analysis.disease,
            provider=analysis.provider.value,
            model=analysis.model,
            status=analysis.status.value,
            created_at=analysis.created_at,
            started_at=analysis.started_at,
            completed_at=analysis.completed_at,
            processing_time_seconds=analysis.processing_time_seconds,
            result_summary=analysis.result_summary,
            confidence_score=analysis.confidence_score,
            input_tokens=analysis.input_tokens,
            output_tokens=analysis.output_tokens,
            total_tokens=analysis.total_tokens,
            estimated_cost=analysis.estimated_cost,
            error_message=analysis.error_message,
            retry_count=analysis.retry_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


@router.get("/sessions/{session_id}/analyses", response_model=AnalysisHistoryResponse)
async def get_analysis_history(
    session_id: int,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(default=None, description="Filter by status"),
    gene_filter: Optional[str] = Query(default=None, description="Filter by gene"),
    disease_filter: Optional[str] = Query(default=None, description="Filter by disease"),
    db: AsyncSession = Depends(get_async_session)
):
    """Get analysis history for a user session with pagination and filtering"""
    try:
        # Verify session exists
        user_session = await UserSessionCRUD.get_session_by_id(db, session_id)
        if not user_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        # Parse status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = AnalysisStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status filter: {status_filter}"
                )
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get analysis history
        analyses, total_count = await AnalysisCRUD.get_user_analysis_history(
            session=db,
            session_id=session_id,
            limit=page_size,
            offset=offset,
            status_filter=status_enum,
            gene_filter=gene_filter,
            disease_filter=disease_filter
        )
        
        # Convert to response models
        analysis_responses = []
        for analysis in analyses:
            analysis_responses.append(AnalysisResponse(
                id=analysis.id,
                session_id=analysis.session_id,
                gene=analysis.gene,
                disease=analysis.disease,
                provider=analysis.provider.value,
                model=analysis.model,
                status=analysis.status.value,
                created_at=analysis.created_at,
                started_at=analysis.started_at,
                completed_at=analysis.completed_at,
                processing_time_seconds=analysis.processing_time_seconds,
                result_summary=analysis.result_summary,
                confidence_score=analysis.confidence_score,
                input_tokens=analysis.input_tokens,
                output_tokens=analysis.output_tokens,
                total_tokens=analysis.total_tokens,
                estimated_cost=analysis.estimated_cost,
                error_message=analysis.error_message,
                retry_count=analysis.retry_count
            ))
        
        return AnalysisHistoryResponse(
            analyses=analysis_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis history for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis history"
        )


@router.get("/analyses/search")
async def search_analyses(
    q: str = Query(..., min_length=1, description="Search term"),
    session_id: Optional[int] = Query(default=None, description="Filter by session ID"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_async_session)
):
    """Search analyses by gene, disease, or content"""
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Search analyses
        analyses, total_count = await AnalysisCRUD.search_analyses(
            session=db,
            search_term=q,
            session_id=session_id,
            limit=page_size,
            offset=offset
        )
        
        # Convert to response models
        analysis_responses = []
        for analysis in analyses:
            analysis_responses.append(AnalysisResponse(
                id=analysis.id,
                session_id=analysis.session_id,
                gene=analysis.gene,
                disease=analysis.disease,
                provider=analysis.provider.value,
                model=analysis.model,
                status=analysis.status.value,
                created_at=analysis.created_at,
                started_at=analysis.started_at,
                completed_at=analysis.completed_at,
                processing_time_seconds=analysis.processing_time_seconds,
                result_summary=analysis.result_summary,
                confidence_score=analysis.confidence_score,
                input_tokens=analysis.input_tokens,
                output_tokens=analysis.output_tokens,
                total_tokens=analysis.total_tokens,
                estimated_cost=analysis.estimated_cost,
                error_message=analysis.error_message,
                retry_count=analysis.retry_count
            ))
        
        return AnalysisHistoryResponse(
            analyses=analysis_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to search analyses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search analyses"
        )


# Analytics Endpoints

@router.get("/sessions/{session_id}/stats", response_model=UsageStatsResponse)
async def get_session_usage_stats(
    session_id: int,
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include"),
    db: AsyncSession = Depends(get_async_session)
):
    """Get usage statistics for a specific session"""
    try:
        # Verify session exists
        user_session = await UserSessionCRUD.get_session_by_id(db, session_id)
        if not user_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        # Get usage statistics
        stats = await AnalyticsQueries.get_usage_statistics(
            session=db,
            session_id=session_id,
            days=days
        )
        
        return UsageStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage stats for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics"
        )


@router.get("/stats/global")
async def get_global_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include"),
    db: AsyncSession = Depends(get_async_session)
):
    """Get global usage statistics across all sessions"""
    try:
        # Get global usage statistics
        usage_stats = await AnalyticsQueries.get_usage_statistics(
            session=db,
            days=days
        )
        
        # Get provider statistics
        provider_stats = await AnalyticsQueries.get_provider_statistics(
            session=db,
            days=days
        )
        
        return {
            "usage_statistics": usage_stats,
            "provider_statistics": provider_stats,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to get global stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve global statistics"
        )


# Health and Status Endpoints

@router.get("/health")
async def persistent_storage_health(db: AsyncSession = Depends(get_async_session)):
    """Health check for persistent storage system"""
    try:
        from ..db.session import check_database_health, get_database_stats
        
        # Check database health
        db_health = await check_database_health()
        db_stats = await get_database_stats()
        
        # Get worker pool status
        try:
            pool_status = await get_pool_status()
            worker_health = "healthy" if pool_status.get("running") else "unhealthy"
        except Exception:
            worker_health = "unknown"
            pool_status = {"error": "Worker pool status unavailable"}
        
        overall_status = "healthy"
        if db_health.get("status") != "healthy" or worker_health != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_health,
            "database_stats": db_stats,
            "worker_pool": {
                "status": worker_health,
                "details": pool_status
            }
        }
        
    except Exception as e:
        logger.error(f"Persistent storage health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )