from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import json
import asyncio
import logging
from ..models.analysis import (
    AnalysisRequest, AnalysisResult, AnalysisHistoryResponse, 
    AnalysisProgress, AnalysisStatus
)
from ..storage.memory_store import storage
from ..services.analysis_service import analysis_service

logger = logging.getLogger(__name__)

# Create router with prefix and tags for API organization
router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}}
)


async def validate_session(session_id: str) -> None:
    """
    Helper function to validate session exists and is active
    
    Args:
        session_id: Session identifier to validate
        
    Raises:
        HTTPException: 401 if session is invalid
    """
    if not storage.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )


@router.post("/", response_model=AnalysisResult, status_code=status.HTTP_201_CREATED)
async def create_analysis(analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Initiates gene-disease analysis processing
    
    This endpoint:
    1. Validates the session and request parameters
    2. Starts analysis processing in the background
    3. Returns the initial analysis record
    4. Processing continues asynchronously
    
    Args:
        analysis_request: Contains gene, disease, and session_id
        background_tasks: FastAPI background task handler
        
    Returns:
        AnalysisResult: Initial analysis record with pending status
        
    Raises:
        HTTPException: 401 for invalid session, 400 for validation errors
    """
    try:
        logger.info(f"Creating analysis: {analysis_request.gene} vs {analysis_request.disease}")
        
        # Validate session exists
        await validate_session(analysis_request.session_id)
        
        # Create initial analysis record
        analysis = storage.create_analysis(
            session_id=analysis_request.session_id,
            gene=analysis_request.gene,
            disease=analysis_request.disease
        )
        
        # Start processing in background (non-blocking)
        background_tasks.add_task(
            run_analysis_background,
            analysis_request.session_id,
            analysis_request.gene,
            analysis_request.disease
        )
        
        logger.info(f"Analysis created: {analysis.analysis_id}")
        return analysis
        
    except HTTPException:
        # Re-raise HTTP exceptions (like invalid session)
        raise
        
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create analysis"
        )


async def run_analysis_background(session_id: str, gene: str, disease: str):
    """
    Background task to run analysis processing
    
    Args:
        session_id: Session identifier
        gene: Gene to analyze
        disease: Disease to analyze
    """
    try:
        await analysis_service.start_analysis(session_id, gene, disease)
    except Exception as e:
        logger.error(f"Background analysis failed: {str(e)}")


@router.get("/stream/{analysis_id}")
async def stream_analysis_progress(analysis_id: str):
    """
    Server-Sent Events endpoint for real-time analysis progress
    
    This endpoint provides streaming updates of analysis progress using SSE.
    The client can connect to this endpoint to receive real-time updates.
    
    Args:
        analysis_id: Analysis identifier to stream
        
    Returns:
        StreamingResponse: SSE stream of progress updates
        
    Raises:
        HTTPException: 404 if analysis not found
    """
    
    async def progress_generator() -> AsyncGenerator[str, None]:
        """
        Generates Server-Sent Events for analysis progress
        
        Yields:
            str: SSE-formatted progress updates
        """
        try:
            # Check if analysis exists
            analysis = storage.get_analysis(analysis_id)
            if not analysis:
                yield f"data: {json.dumps({'error': 'Analysis not found'})}\n\n"
                return
            
            # Stream progress until completion or failure
            last_status = None
            while True:
                current_analysis = storage.get_analysis(analysis_id)
                if not current_analysis:
                    break
                
                # Send update if status changed
                if current_analysis.status != last_status:
                    progress_data = {
                        "analysis_id": analysis_id,
                        "status": current_analysis.status.value,
                        "message": current_analysis.summary,
                        "timestamp": current_analysis.created_at.isoformat()
                    }
                    
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    last_status = current_analysis.status
                
                # Break if analysis is complete or failed
                if current_analysis.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                    break
                
                # Wait before next check
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in progress stream for {analysis_id}: {str(e)}")
            yield f"data: {json.dumps({'error': 'Stream error'})}\n\n"
    
    # Return SSE stream response
    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/history/{session_id}", response_model=AnalysisHistoryResponse)
async def get_analysis_history(session_id: str):
    """
    Retrieves analysis history for a specific session
    
    This endpoint:
    1. Validates the session exists
    2. Retrieves all analyses for the session
    3. Returns analyses in reverse chronological order
    
    Args:
        session_id: Session identifier
        
    Returns:
        AnalysisHistoryResponse: Session info and list of analyses
        
    Raises:
        HTTPException: 401 if session invalid, 500 for server errors
    """
    try:
        logger.info(f"Retrieving analysis history for session: {session_id}")
        
        # Validate session exists
        await validate_session(session_id)
        
        # Get session info for response context
        session_info = storage.get_session(session_id)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Retrieve all analyses for the session
        analyses = storage.get_session_analyses(session_id)
        analysis_count = len(analyses)
        
        logger.info(f"Found {analysis_count} analyses for session {session_id}")
        
        return AnalysisHistoryResponse(
            session_id=session_id,
            username=session_info.username,
            total_analyses=analysis_count,
            analyses=analyses
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving history for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis history"
        )


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(analysis_id: str):
    """
    Retrieves a specific analysis by ID
    
    This endpoint:
    1. Looks up the analysis by ID
    2. Returns complete analysis information
    3. Works for analyses in any status (pending, processing, completed, failed)
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        AnalysisResult: Complete analysis information
        
    Raises:
        HTTPException: 404 if analysis not found
    """
    try:
        logger.info(f"Retrieving analysis: {analysis_id}")
        
        analysis = storage.get_analysis(analysis_id)
        
        if not analysis:
            logger.warning(f"Analysis not found: {analysis_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        logger.info(f"Analysis retrieved: {analysis_id} (status: {analysis.status})")
        return analysis
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )