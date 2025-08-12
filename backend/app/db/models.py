"""
SQLModel database models for persistent storage of LLM analyses and user sessions
Future-proof schema design supporting embeddings, search, and cost tracking
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from sqlmodel import SQLModel, Field, Relationship, JSON, Column, String, Text
from sqlalchemy import Index, UniqueConstraint


class AnalysisStatus(str, Enum):
    """Status of an LLM analysis"""
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LLMProviderType(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Base classes for shared fields
class TimestampMixin(SQLModel):
    """Mixin for common timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, nullable=True)


class UserSession(TimestampMixin, table=True):
    """
    User session table for tracking authenticated users and their analysis sessions
    
    Future extensibility:
    - Can add fields for user preferences, settings, API quotas
    - Supports multiple authentication providers
    - Tracks session activity for analytics
    """
    __tablename__ = "user_sessions"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # User identification
    username: str = Field(min_length=1, max_length=255, index=True)
    
    # Authentication details
    provider: str = Field(max_length=50, default="internal")  # 'internal', 'oauth', etc.
    external_id: Optional[str] = Field(default=None, max_length=255)  # External provider user ID
    
    # Session management
    session_token: Optional[str] = Field(default=None, max_length=255, index=True)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)
    
    # User preferences (JSON for flexibility)
    preferences: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Activity tracking
    total_analyses: int = Field(default=0)
    total_tokens_used: int = Field(default=0)  # For cost tracking
    
    # Status
    is_active: bool = Field(default=True)
    
    # Relationships
    analyses: List["Analysis"] = Relationship(back_populates="session")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_username_provider", "username", "provider"),
        Index("idx_session_token", "session_token"),
        Index("idx_last_active", "last_active_at"),
    )


class Analysis(TimestampMixin, table=True):
    """
    Analysis table for storing LLM analysis requests and results
    
    Future extensibility:
    - Embeddings column for semantic search
    - Cost tracking per analysis
    - Performance metrics and feedback
    - Version tracking for prompt iterations
    """
    __tablename__ = "analyses"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Foreign key to user session
    session_id: int = Field(foreign_key="user_sessions.id", index=True)
    
    # Analysis request details
    gene: str = Field(min_length=1, max_length=100, index=True)
    disease: str = Field(min_length=1, max_length=500, index=True)
    context: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # LLM provider details
    provider: LLMProviderType = Field(index=True)
    model: str = Field(max_length=100)
    prompt_text: str = Field(sa_column=Column(Text))
    
    # Analysis status and timing
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, index=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    processing_time_seconds: Optional[float] = Field(default=None)
    
    # Results and metadata
    result_summary: Optional[str] = Field(default=None, sa_column=Column(Text))
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Structured metadata (JSON for flexibility)
    analysis_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Cost and usage tracking
    input_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)
    estimated_cost: Optional[float] = Field(default=None)
    
    # Future: Embeddings for semantic search (stored as JSON for now)
    embeddings: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Error handling
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    retry_count: int = Field(default=0)
    
    # Quality and feedback (for future ML improvements)
    user_rating: Optional[int] = Field(default=None, ge=1, le=5)
    user_feedback: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Version tracking
    prompt_version: str = Field(default="1.0", max_length=20)
    schema_version: int = Field(default=1)
    
    # Relationships
    session: Optional[UserSession] = Relationship(back_populates="analyses")
    chunks: List["AnalysisChunk"] = Relationship(back_populates="analysis")
    
    # Indexes for efficient queries
    __table_args__ = (
        Index("idx_gene_disease", "gene", "disease"),
        Index("idx_session_status", "session_id", "status"),
        Index("idx_provider_model", "provider", "model"),
        Index("idx_created_status", "created_at", "status"),
        Index("idx_gene_provider", "gene", "provider"),
        Index("idx_disease_provider", "disease", "provider"),
    )


class AnalysisChunk(TimestampMixin, table=True):
    """
    Analysis chunk table for storing streaming analysis results
    Supports real-time streaming and reconstruction of full analysis
    
    Future extensibility:
    - Chunk embeddings for fine-grained search
    - Chunk-level feedback and corrections
    - Support for multimedia chunks (images, charts)
    """
    __tablename__ = "analysis_chunks"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Foreign key to analysis
    analysis_id: int = Field(foreign_key="analyses.id", index=True)
    
    # Chunk ordering and content
    sequence_number: int = Field(ge=0)
    chunk_type: str = Field(max_length=50, default="text")  # 'text', 'json', 'metadata'
    chunk_text: str = Field(sa_column=Column(Text))
    
    # Chunk metadata
    chunk_metadata: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Streaming information
    token_count: Optional[int] = Field(default=None)
    streaming_complete: bool = Field(default=False)
    
    # Future: Individual chunk embeddings
    chunk_embeddings: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Quality metrics per chunk
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Relationships
    analysis: Optional[Analysis] = Relationship(back_populates="chunks")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("analysis_id", "sequence_number", name="uq_analysis_sequence"),
        Index("idx_analysis_sequence", "analysis_id", "sequence_number"),
        Index("idx_chunk_type", "chunk_type"),
        Index("idx_streaming_complete", "streaming_complete"),
    )


# Enhanced tables for advanced features

class CostTracking(TimestampMixin, table=True):
    """
    Cost tracking table for detailed usage and billing analytics
    Tracks costs per analysis, provider, and session
    """
    __tablename__ = "cost_tracking"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Foreign keys
    session_id: int = Field(foreign_key="user_sessions.id", index=True)
    analysis_id: Optional[int] = Field(foreign_key="analyses.id", index=True, default=None)
    
    # Provider and model information
    provider: LLMProviderType = Field(index=True)
    model: str = Field(max_length=100, index=True)
    
    # Token usage details
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    
    # Cost calculations
    input_cost_per_token: float = Field(default=0.0, ge=0)
    output_cost_per_token: float = Field(default=0.0, ge=0)
    total_cost: float = Field(default=0.0, ge=0)
    
    # Cost metadata
    currency: str = Field(default="USD", max_length=3)
    billing_tier: Optional[str] = Field(default=None, max_length=50)
    cost_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Performance metrics
    processing_time_ms: Optional[int] = Field(default=None, ge=0)
    queue_time_ms: Optional[int] = Field(default=None, ge=0)
    
    __table_args__ = (
        Index("idx_cost_session_date", "session_id", "created_at"),
        Index("idx_cost_provider_model", "provider", "model"),
        Index("idx_cost_date_provider", "created_at", "provider"),
        Index("idx_cost_analysis", "analysis_id"),
    )


class ContextAugmentation(TimestampMixin, table=True):
    """
    Context augmentation table for storing NHS Scotland data and external context
    Caches external data to reduce API calls and improve performance
    """
    __tablename__ = "context_augmentation"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Content identifiers
    gene: str = Field(min_length=1, max_length=100, index=True)
    disease: str = Field(min_length=1, max_length=500, index=True)
    data_source: str = Field(max_length=100, index=True)  # 'nhs_scotland', 'pubmed', etc.
    
    # Context data
    context_data: Dict[str, Any] = Field(sa_column=Column(JSON))
    raw_data: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Data metadata
    data_version: str = Field(max_length=50, default="1.0")
    source_url: Optional[str] = Field(default=None, max_length=1000)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Caching and expiration
    expires_at: Optional[datetime] = Field(default=None)
    last_validated_at: Optional[datetime] = Field(default=None)
    
    # Usage tracking
    usage_count: int = Field(default=0, ge=0)
    
    __table_args__ = (
        UniqueConstraint("gene", "disease", "data_source", name="uq_gene_disease_source"),
        Index("idx_gene_source", "gene", "data_source"),
        Index("idx_disease_source", "disease", "data_source"),
        Index("idx_expires_at", "expires_at"),
        Index("idx_usage_count", "usage_count"),
    )


class AnalysisVersion(TimestampMixin, table=True):
    """
    Analysis version table for tracking re-runs and analysis evolution
    Supports analysis improvement and comparison over time
    """
    __tablename__ = "analysis_versions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Original and parent analysis references
    original_analysis_id: int = Field(foreign_key="analyses.id", index=True)
    parent_version_id: Optional[int] = Field(foreign_key="analysis_versions.id", index=True, default=None)
    
    # Version information
    version_number: int = Field(ge=1, index=True)
    version_type: str = Field(max_length=50, default="rerun")  # 'rerun', 'improvement', 'correction'
    
    # Changes made in this version
    changes_description: Optional[str] = Field(default=None, sa_column=Column(Text))
    prompt_changes: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Version metadata
    triggered_by: str = Field(max_length=50, default="user")  # 'user', 'system', 'schedule'
    trigger_reason: Optional[str] = Field(default=None, max_length=500)
    
    # Performance comparison
    improvement_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    
    __table_args__ = (
        UniqueConstraint("original_analysis_id", "version_number", name="uq_analysis_version"),
        Index("idx_original_version", "original_analysis_id", "version_number"),
        Index("idx_version_type", "version_type"),
        Index("idx_triggered_by", "triggered_by"),
    )


class AdminMetrics(TimestampMixin, table=True):
    """
    Admin metrics table for storing aggregated statistics and dashboard data
    Pre-computed metrics for performance and administrative insights
    """
    __tablename__ = "admin_metrics"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Metric identification
    metric_name: str = Field(max_length=100, index=True)
    metric_type: str = Field(max_length=50, index=True)  # 'daily', 'hourly', 'realtime'
    time_period: str = Field(max_length=50, index=True)  # '2024-01-15', '2024-01-15T14', etc.
    
    # Metric data
    metric_value: float = Field()
    metric_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Aggregation metadata
    sample_count: int = Field(default=0, ge=0)
    calculation_method: str = Field(max_length=50, default="sum")
    
    # Data quality
    is_estimated: bool = Field(default=False)
    confidence_level: float = Field(default=1.0, ge=0.0, le=1.0)
    
    __table_args__ = (
        UniqueConstraint("metric_name", "metric_type", "time_period", name="uq_metric_period"),
        Index("idx_metric_name_type", "metric_name", "metric_type"),
        Index("idx_time_period", "time_period"),
        Index("idx_metric_value", "metric_value"),
    )


# Future tables for extensibility

class AnalysisTemplate(TimestampMixin, table=True):
    """
    Template table for storing reusable analysis prompts and configurations
    Future enhancement for prompt management and A/B testing
    """
    __tablename__ = "analysis_templates"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, unique=True)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Template configuration
    provider: LLMProviderType = Field()
    model: str = Field(max_length=100)
    prompt_template: str = Field(sa_column=Column(Text))
    
    # Template metadata
    parameters: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    version: str = Field(max_length=20, default="1.0")
    
    # Usage tracking
    usage_count: int = Field(default=0)
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Status
    is_active: bool = Field(default=True)
    is_default: bool = Field(default=False)
    
    __table_args__ = (
        Index("idx_provider_active", "provider", "is_active"),
        Index("idx_usage_count", "usage_count"),
    )


class SearchIndex(TimestampMixin, table=True):
    """
    Search index table for full-text search and semantic search capabilities
    Future enhancement for advanced search features
    """
    __tablename__ = "search_indices"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="analyses.id", index=True)
    
    # Search content
    searchable_text: str = Field(sa_column=Column(Text))
    keywords: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Search metadata
    language: str = Field(max_length=10, default="en")
    content_type: str = Field(max_length=50, default="analysis")
    
    # Search performance
    search_weight: float = Field(default=1.0)
    last_indexed_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_content_type", "content_type"),
        Index("idx_last_indexed", "last_indexed_at"),
    )


# Utility functions for model operations

def create_analysis_from_request(
    session_id: int,
    gene: str,
    disease: str,
    provider: LLMProviderType,
    model: str,
    prompt_text: str,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Analysis:
    """
    Create a new Analysis instance from request parameters
    
    Args:
        session_id: User session ID
        gene: Gene symbol
        disease: Disease name
        provider: LLM provider
        model: Model name
        prompt_text: Generated prompt
        context: Additional context
        metadata: Additional metadata
        
    Returns:
        Analysis instance ready for database insertion
    """
    return Analysis(
        session_id=session_id,
        gene=gene.strip().upper(),
        disease=disease.strip().lower(),
        context=context,
        provider=provider,
        model=model,
        prompt_text=prompt_text,
        analysis_metadata=metadata or {},
        status=AnalysisStatus.PENDING
    )


def create_analysis_chunk(
    analysis_id: int,
    sequence_number: int,
    chunk_text: str,
    chunk_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> AnalysisChunk:
    """
    Create a new AnalysisChunk instance
    
    Args:
        analysis_id: Parent analysis ID
        sequence_number: Sequence number for ordering
        chunk_text: Chunk content
        chunk_type: Type of chunk (text, json, metadata)
        metadata: Chunk-specific metadata
        
    Returns:
        AnalysisChunk instance ready for database insertion
    """
    return AnalysisChunk(
        analysis_id=analysis_id,
        sequence_number=sequence_number,
        chunk_type=chunk_type,
        chunk_text=chunk_text,
        chunk_metadata=metadata or {},
        streaming_complete=False
    )