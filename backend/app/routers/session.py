from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from ..models.session import SessionCreateRequest, SessionResponse
from ..storage.memory_store import storage
import logging

# Configure logging for debugging and monitoring
logger = logging.getLogger(__name__)

# Create router with prefix and tags for API organization
router = APIRouter(
    prefix="/session",
    tags=["session"],
    responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(session_request: SessionCreateRequest):
    """
    Creates a new user session with username and API key validation
    
    This endpoint:
    1. Validates the username format and API key structure
    2. Creates a new session in memory storage
    3. Returns session ID for subsequent API calls
    
    Args:
        session_request: Contains username and api_key
        
    Returns:
        SessionResponse: Session ID, username, and creation timestamp
        
    Raises:
        HTTPException: 400 if validation fails, 500 if session creation fails
    """
    try:
        logger.info(f"Creating session for username: {session_request.username}")
        
        # Pydantic validation occurs automatically at this point
        # Additional business logic validation could go here
        
        # Create session in storage
        session_info = storage.create_session(
            username=session_request.username,
            api_key=session_request.api_key
        )
        
        logger.info(f"Session created successfully: {session_info.session_id}")
        
        # Return public session information (API key excluded for security)
        return SessionResponse(
            session_id=session_info.session_id,
            username=session_info.username,
            created_at=session_info.created_at
        )
        
    except ValueError as ve:
        # Pydantic validation errors or custom validation errors
        logger.warning(f"Validation error creating session: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    
    except Exception as e:
        # Unexpected errors in session creation
        logger.error(f"Unexpected error creating session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session. Please try again."
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """
    Retrieves basic session information by session ID
    
    This endpoint:
    1. Validates that the session exists and is active
    2. Returns public session information
    3. Updates the last accessed timestamp
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        SessionResponse: Session details without sensitive information
        
    Raises:
        HTTPException: 404 if session not found
    """
    try:
        logger.info(f"Retrieving session info for: {session_id}")
        
        # Retrieve session from storage (updates last_accessed)
        session_info = storage.get_session(session_id)
        
        if not session_info:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired"
            )
        
        logger.info(f"Session info retrieved for: {session_info.username}")
        
        # Return public information only
        return SessionResponse(
            session_id=session_info.session_id,
            username=session_info.username,
            created_at=session_info.created_at
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error retrieving session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session information"
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Deletes a session and all associated data
    
    This endpoint:
    1. Removes the session from storage
    2. Removes all analyses associated with the session
    3. Returns 204 No Content on success
    
    Args:
        session_id: UUID string identifying the session to delete
        
    Returns:
        204 No Content status on successful deletion
        
    Raises:
        HTTPException: 404 if session not found
    """
    try:
        logger.info(f"Deleting session: {session_id}")
        
        # Attempt to delete session from storage
        deleted = storage.delete_session(session_id)
        
        if not deleted:
            logger.warning(f"Attempted to delete non-existent session: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        logger.info(f"Session deleted successfully: {session_id}")
        
        # Return 204 No Content (no response body needed)
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error deleting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )