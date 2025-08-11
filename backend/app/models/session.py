from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Optional
import re


class SessionCreateRequest(BaseModel):
    """
    Request model for creating a new user session
    Validates username format and API key structure
    """
    username: str = Field(..., min_length=3, max_length=50, description="Username for the session")
    api_key: str = Field(..., min_length=10, description="OpenAI or Anthropic API key")
    
    @validator('username')
    def validate_username(cls, v):
        """
        Validates username contains only alphanumeric characters, hyphens, and underscores
        Prevents potential security issues with special characters
        """
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.strip()
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """
        Validates API key format for OpenAI (sk-) or Anthropic (sk-ant- or anthropic-)
        Basic format validation to prevent obviously invalid keys
        """
        v = v.strip()
        openai_pattern = r'^sk-[a-zA-Z0-9]{32,}$'
        anthropic_pattern = r'^(sk-ant-|anthropic-)[a-zA-Z0-9\-_]{20,}$'
        
        if not (re.match(openai_pattern, v) or re.match(anthropic_pattern, v)):
            raise ValueError('Invalid API key format. Must be OpenAI (sk-...) or Anthropic (sk-ant-... or anthropic-...)')
        return v


class SessionResponse(BaseModel):
    """
    Response model for successful session creation
    Returns session ID and basic session information
    """
    session_id: str = Field(..., description="Unique session identifier")
    username: str = Field(..., description="Username associated with the session")
    created_at: datetime = Field(..., description="Session creation timestamp")
    
    class Config:
        # Allows datetime serialization in JSON responses
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionInfo(BaseModel):
    """
    Internal model for storing session data in memory
    Contains sensitive information that should not be exposed in API responses
    """
    session_id: str
    username: str
    api_key: str  # Stored securely, never returned in responses
    created_at: datetime
    last_accessed: datetime