"""
FastAPI router for worker pool management and job monitoring
Provides endpoints for submitting jobs, checking status, and monitoring the worker pool
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..models.llm import LLMAnalysisRequest, LLMServiceResponse
from ..services.worker_pool import (
    submit_llm_job, get_job_status, cancel_job, get_pool_status,
    JobPriority, JobStatus, QueuedJob
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/workers",
    tags=["worker-management"],
    responses={
        404: {"description": "Job not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Worker pool unavailable"}
    }
)


@router.get("/", response_model=dict)
async def worker_management_info():
    """
    Get information about worker pool management capabilities
    
    Returns basic information about the worker pool, rate limiting, and job management.
    """
    return {
        "service": "LLM Worker Pool Management",
        "description": "Asynchronous job processing with concurrency control and rate limiting",
        "features": [
            "Configurable concurrency limits",
            "Per-provider rate limiting with token bucket algorithm",
            "FIFO job processing per user session",
            "Exponential backoff with jitter for retries",
            "Job status tracking and monitoring",
            "Graceful shutdown and recovery"
        ],
        "available_endpoints": {
            "submit": "POST /workers/jobs - Submit new LLM analysis job",
            "status": "GET /workers/jobs/{job_id} - Get job status",
            "cancel": "DELETE /workers/jobs/{job_id} - Cancel queued job",
            "pool_status": "GET /workers/pool - Get worker pool status",
            "statistics": "GET /workers/stats - Get processing statistics"
        },
        "job_priorities": [priority.name for priority in JobPriority],
        "job_statuses": [status.name for status in JobStatus]
    }


@router.post("/jobs", response_model=dict)
async def submit_job(
    request: LLMAnalysisRequest,
    session_id: Optional[str] = None,
    priority: Optional[str] = None,
    timeout: Optional[float] = None
):
    """
    Submit a new LLM analysis job to the worker pool
    
    Jobs are processed asynchronously with automatic retry logic and rate limiting.
    FIFO processing is maintained per session ID for fairness.
    
    **Priority Levels:**
    - LOW: Background processing
    - NORMAL: Standard priority (default)
    - HIGH: Expedited processing
    - CRITICAL: Highest priority
    """
    try:
        # Parse priority
        job_priority = JobPriority.NORMAL
        if priority:
            try:
                job_priority = JobPriority[priority.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid priority '{priority}'. Valid options: {[p.name for p in JobPriority]}"
                )
        
        # Submit job to worker pool
        job_id = await submit_llm_job(
            request=request,
            session_id=session_id,
            priority=job_priority,
            timeout=timeout
        )
        
        logger.info(f"Job {job_id} submitted successfully (session: {session_id}, priority: {job_priority.name})")
        
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "message": "Job submitted successfully",
            "session_id": session_id,
            "priority": job_priority.name,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit job: {e}", exc_info=True)
        if "queue full" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Worker pool queue is full. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit job"
            )


@router.get("/jobs/{job_id}", response_model=dict)
async def get_job_details(job_id: str):
    """
    Get detailed status and information for a specific job
    
    Returns current status, processing time, retry count, and results if completed.
    """
    try:
        job = await get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Build response
        response = {
            "job_id": job.id,
            "status": job.status.value,
            "priority": job.priority.name,
            "session_id": job.session_id,
            "created_at": job.created_at.isoformat(),
            "age_seconds": job.age_seconds,
            "timeout": job.timeout,
            "max_retries": job.max_retries,
            "current_retry": job.current_retry,
            "request": {
                "gene": job.request.gene,
                "disease": job.request.disease,
                "provider": job.request.provider.value,
                "model": job.request.model
            }
        }
        
        # Add processing information if available
        if job.started_at:
            response["started_at"] = job.started_at.isoformat()
            response["processing_time"] = job.processing_time
        
        if job.completed_at:
            response["completed_at"] = job.completed_at.isoformat()
        
        # Add results or error information
        if job.status == JobStatus.COMPLETED and job.result:
            response["result"] = {
                "success": job.result.success,
                "summary": job.result.data.summary if job.result.data else None,
                "confidence_score": job.result.data.confidence_score if job.result.data else None,
                "usage_stats": job.result.usage_stats.dict() if job.result.usage_stats else None
            }
        elif job.status == JobStatus.FAILED:
            response["error"] = job.error or (job.result.error.message if job.result and job.result.error else "Unknown error")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details for {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job details"
        )


@router.delete("/jobs/{job_id}", response_model=dict)
async def cancel_job_request(job_id: str):
    """
    Cancel a queued job
    
    Only jobs in QUEUED status can be cancelled. Processing or completed jobs cannot be cancelled.
    """
    try:
        success = await cancel_job(job_id)
        
        if success:
            logger.info(f"Job {job_id} cancelled successfully")
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job cancelled successfully",
                "cancelled_at": datetime.utcnow().isoformat()
            }
        else:
            # Check if job exists
            job = await get_job_status(job_id)
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Job {job_id} cannot be cancelled (current status: {job.status.value})"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )


@router.get("/pool", response_model=dict)
async def get_pool_status():
    """
    Get current worker pool status and configuration
    
    Returns information about active workers, queue status, rate limiters, and statistics.
    """
    try:
        status = await get_pool_status()
        
        # Add additional metadata
        status["timestamp"] = datetime.utcnow().isoformat()
        status["health"] = "healthy" if status["running"] else "stopped"
        
        # Calculate utilization metrics
        if status["concurrency_limit"] > 0:
            status["worker_utilization"] = status["active_jobs"] / status["concurrency_limit"]
        
        if status["queue_capacity"] > 0:
            status["queue_utilization"] = status["queue_size"] / status["queue_capacity"]
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting pool status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pool status"
        )


@router.get("/stats", response_model=dict)
async def get_processing_statistics():
    """
    Get detailed processing statistics and performance metrics
    
    Returns comprehensive statistics about job processing, performance, and system health.
    """
    try:
        pool_status = await get_pool_status()
        stats = pool_status.get("statistics", {})
        
        # Calculate derived metrics
        total_jobs = stats.get("jobs_processed", 0) + stats.get("jobs_failed", 0)
        success_rate = (stats.get("jobs_processed", 0) / total_jobs * 100) if total_jobs > 0 else 0
        
        avg_processing_time = 0
        if stats.get("jobs_processed", 0) > 0:
            avg_processing_time = stats.get("total_processing_time", 0) / stats.get("jobs_processed", 0)
        
        avg_wait_time = 0
        if total_jobs > 0:
            avg_wait_time = stats.get("total_wait_time", 0) / total_jobs
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "job_statistics": {
                "jobs_queued": stats.get("jobs_queued", 0),
                "jobs_processed": stats.get("jobs_processed", 0),
                "jobs_failed": stats.get("jobs_failed", 0),
                "jobs_cancelled": stats.get("jobs_cancelled", 0),
                "retry_count": stats.get("retry_count", 0),
                "success_rate_percent": round(success_rate, 2)
            },
            "performance_metrics": {
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "average_wait_time_seconds": round(avg_wait_time, 2),
                "total_processing_time_seconds": round(stats.get("total_processing_time", 0), 2),
                "total_wait_time_seconds": round(stats.get("total_wait_time", 0), 2)
            },
            "system_status": {
                "worker_pool_running": pool_status.get("running", False),
                "active_workers": pool_status.get("workers", 0),
                "concurrency_limit": pool_status.get("concurrency_limit", 0),
                "queue_size": pool_status.get("queue_size", 0),
                "queue_capacity": pool_status.get("queue_capacity", 0),
                "active_jobs": pool_status.get("active_jobs", 0),
                "queued_jobs": pool_status.get("queued_jobs", 0)
            },
            "rate_limiting": pool_status.get("rate_limiters", {})
        }
        
    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing statistics"
        )


@router.get("/health", response_model=dict)
async def worker_pool_health():
    """
    Health check for the worker pool system
    
    Returns health status and basic system information for monitoring.
    """
    try:
        pool_status = await get_pool_status()
        
        # Determine health status
        health = "healthy"
        issues = []
        
        if not pool_status.get("running", False):
            health = "unhealthy"
            issues.append("Worker pool is not running")
        
        if pool_status.get("active_jobs", 0) >= pool_status.get("concurrency_limit", 0):
            issues.append("All workers are busy")
        
        queue_utilization = 0
        if pool_status.get("queue_capacity", 0) > 0:
            queue_utilization = pool_status.get("queue_size", 0) / pool_status.get("queue_capacity", 0)
            if queue_utilization > 0.8:
                issues.append("Queue is nearly full")
        
        if issues and health == "healthy":
            health = "degraded"
        
        return {
            "status": health,
            "timestamp": datetime.utcnow().isoformat(),
            "worker_pool_running": pool_status.get("running", False),
            "active_workers": pool_status.get("workers", 0),
            "queue_utilization": round(queue_utilization * 100, 2),
            "issues": issues,
            "uptime_info": "Worker pool operational"
        }
        
    except Exception as e:
        logger.error(f"Worker pool health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check failed",
                "worker_pool_running": False
            }
        )