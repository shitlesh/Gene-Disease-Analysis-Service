from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum
import re


class AnalysisStatus(str, Enum):
    """
    Enumeration of possible analysis states
    Used for tracking processing status
    """
    PENDING = "pending"
    PROCESSING = "processing"  
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    """
    Request model for gene-disease analysis
    Validates gene and disease name formats
    """
    gene: str = Field(..., min_length=2, max_length=20, description="Gene name to analyze")
    disease: str = Field(..., min_length=3, max_length=100, description="Disease name to analyze")
    session_id: str = Field(..., description="Session identifier for authentication")
    
    @validator('gene')
    def validate_gene(cls, v):
        """
        Validates gene name contains only letters, numbers, hyphens
        Common gene naming conventions (e.g., BRCA1, TP53, CFTR)
        """
        v = v.strip().upper()  # Standardize to uppercase
        if not re.match(r'^[A-Z0-9-]+$', v):
            raise ValueError('Gene name can only contain letters, numbers, and hyphens')
        return v
    
    @validator('disease')
    def validate_disease(cls, v):
        """
        Validates disease name contains only alphanumeric characters and common punctuation
        Allows spaces for multi-word disease names
        """
        v = v.strip().lower()  # Standardize to lowercase
        if not re.match(r'^[a-z0-9\s\-\'\".,()]+$', v):
            raise ValueError('Disease name contains invalid characters')
        return v


class AnalysisProgress(BaseModel):
    """
    Model for streaming analysis progress updates
    Used in real-time progress endpoints
    """
    analysis_id: str = Field(..., description="Analysis identifier")
    status: AnalysisStatus = Field(..., description="Current analysis status")
    progress_message: str = Field(..., description="Human-readable progress description")
    progress_percentage: float = Field(..., ge=0, le=100, description="Completion percentage (0-100)")
    timestamp: datetime = Field(..., description="Progress update timestamp")


class AnalysisResult(BaseModel):
    """
    Complete analysis result model
    Contains all analysis findings and metadata
    """
    analysis_id: str = Field(..., description="Unique analysis identifier")
    session_id: str = Field(..., description="Session that requested the analysis")
    gene: str = Field(..., description="Analyzed gene name")
    disease: str = Field(..., description="Analyzed disease name")
    status: AnalysisStatus = Field(..., description="Analysis completion status")
    
    # Analysis findings
    summary: str = Field(..., description="Executive summary of analysis results")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence in analysis results (0-100)")
    key_findings: List[str] = Field(..., description="List of key research findings")
    pathway_analysis: Optional[str] = Field(None, description="Biological pathway involvement analysis")
    therapeutic_targets: Optional[List[str]] = Field(None, description="Potential therapeutic targets identified")
    
    # Metadata
    created_at: datetime = Field(..., description="Analysis start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion timestamp")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    
    class Config:
        # Ensures proper datetime serialization
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisHistoryResponse(BaseModel):
    """
    Response model for analysis history endpoint
    Contains paginated list of user's previous analyses
    """
    session_id: str = Field(..., description="Session identifier")
    username: str = Field(..., description="Username for context")
    total_analyses: int = Field(..., description="Total number of analyses for this session")
    analyses: List[AnalysisResult] = Field(..., description="List of analysis results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }