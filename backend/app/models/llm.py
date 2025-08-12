"""
Pydantic models for LLM integration requests and responses
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMModel(str, Enum):
    """Supported LLM models by provider"""
    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class GeneAssociation(BaseModel):
    """Model for gene-disease association data"""
    gene_function: str = Field(..., description="Primary function of the gene")
    mechanism: str = Field(..., description="How gene relates to disease")
    evidence_level: str = Field(..., description="Strength of evidence (strong/moderate/weak)")
    phenotypes: List[str] = Field(default=[], description="Associated clinical phenotypes")
    inheritance_pattern: Optional[str] = Field(None, description="Inheritance pattern if applicable")


class LLMAnalysisRequest(BaseModel):
    """Request model for LLM gene-disease analysis"""
    gene: str = Field(..., min_length=1, max_length=50, description="Gene symbol or name")
    disease: str = Field(..., min_length=1, max_length=200, description="Disease name or condition")
    provider: LLMProvider = Field(..., description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use (provider default if not specified)")
    context: Optional[str] = Field(None, description="Additional context for analysis")
    include_references: bool = Field(False, description="Whether to include literature references")
    
    @validator('gene')
    def validate_gene(cls, v):
        """Validate and normalize gene name format"""
        # First normalize the input
        normalized = v.strip().upper()
        
        # Then validate the normalized version
        if not normalized.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Gene name must be alphanumeric with optional hyphens or underscores")
        
        return normalized
    
    @validator('disease') 
    def validate_disease(cls, v):
        """Validate disease name"""
        return v.strip().lower()


class LLMAnalysisResponse(BaseModel):
    """Response model for LLM gene-disease analysis"""
    summary: str = Field(..., description="Brief summary of gene-disease relationship")
    associations: List[GeneAssociation] = Field(..., description="Detailed association information")
    recommendation: str = Field(..., description="Clinical or research recommendations")
    uncertainty: str = Field(..., description="Uncertainty assessment and limitations")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence (0-1)")
    references: Optional[List[str]] = Field(None, description="Literature references if requested")
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": "BRCA1 is strongly associated with hereditary breast and ovarian cancer syndrome",
                "associations": [
                    {
                        "gene_function": "DNA repair, tumor suppressor",
                        "mechanism": "Loss-of-function mutations impair homologous recombination",
                        "evidence_level": "strong",
                        "phenotypes": ["breast cancer", "ovarian cancer", "prostate cancer"],
                        "inheritance_pattern": "autosomal dominant"
                    }
                ],
                "recommendation": "Genetic counseling and screening recommended for carriers",
                "uncertainty": "Penetrance varies; environmental factors influence risk",
                "confidence_score": 0.95
            }
        }


class LLMError(BaseModel):
    """Model for LLM service errors"""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    provider: Optional[str] = Field(None, description="Provider that generated the error")
    status_code: Optional[int] = Field(None, description="HTTP status code if applicable")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")


class LLMUsageStats(BaseModel):
    """Model for LLM usage statistics"""
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    input_tokens: Optional[int] = Field(None, description="Input tokens consumed")
    output_tokens: Optional[int] = Field(None, description="Output tokens generated")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    request_duration: Optional[float] = Field(None, description="Request duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")


class LLMServiceResponse(BaseModel):
    """Complete response from LLM service including metadata"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[LLMAnalysisResponse] = Field(None, description="Analysis results if successful")
    error: Optional[LLMError] = Field(None, description="Error details if unsuccessful")
    usage_stats: Optional[LLMUsageStats] = Field(None, description="Usage and performance metrics")
    provider_response_id: Optional[str] = Field(None, description="Provider's response ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Provider-specific configuration models

class OpenAIConfig(BaseModel):
    """OpenAI-specific configuration"""
    api_key: str = Field(..., description="OpenAI API key")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API base URL")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2000, ge=1, description="Maximum tokens in response")


class AnthropicConfig(BaseModel):
    """Anthropic-specific configuration"""
    api_key: str = Field(..., description="Anthropic API key")
    base_url: str = Field("https://api.anthropic.com", description="Anthropic API base URL")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(2000, ge=1, description="Maximum tokens in response")


class LLMProviderConfig(BaseModel):
    """Configuration for all LLM providers"""
    openai: Optional[OpenAIConfig] = Field(None, description="OpenAI configuration")
    anthropic: Optional[AnthropicConfig] = Field(None, description="Anthropic configuration")
    default_provider: LLMProvider = Field(LLMProvider.OPENAI, description="Default provider to use")
    enable_caching: bool = Field(True, description="Whether to cache responses")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    
    def get_provider_config(self, provider: LLMProvider) -> Union[OpenAIConfig, AnthropicConfig]:
        """Get configuration for specific provider"""
        if provider == LLMProvider.OPENAI:
            if not self.openai:
                raise ValueError("OpenAI configuration not provided")
            return self.openai
        elif provider == LLMProvider.ANTHROPIC:
            if not self.anthropic:
                raise ValueError("Anthropic configuration not provided")
            return self.anthropic
        else:
            raise ValueError(f"Unsupported provider: {provider}")