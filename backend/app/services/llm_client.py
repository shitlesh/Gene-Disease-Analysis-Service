"""
Modular, provider-agnostic LLM integration layer
Supports OpenAI and Anthropic APIs with extensible architecture
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import httpx
from datetime import datetime, timedelta

from ..models.llm import (
    LLMProvider, LLMAnalysisRequest, LLMAnalysisResponse, LLMServiceResponse,
    LLMError, LLMUsageStats, LLMProviderConfig, OpenAIConfig, AnthropicConfig
)
from .prompt_templates import get_analysis_prompt, validate_json_response_format

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Authentication/API key errors"""
    pass


class LLMRateLimitError(LLMProviderError):
    """Rate limit errors"""
    pass


class LLMValidationError(LLMProviderError):
    """Response validation errors"""
    pass


class LLMClient(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self):
        """Setup the HTTP client for the provider"""
        pass
    
    @abstractmethod
    async def _make_request(self, system_message: str, user_message: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Make API request to the provider
        
        Returns:
            Tuple of (response_text, metadata)
        """
        pass
    
    @abstractmethod
    def _extract_usage_stats(self, response_data: Dict[str, Any], duration: float) -> LLMUsageStats:
        """Extract usage statistics from provider response"""
        pass
    
    async def analyze_gene_disease(self, request: LLMAnalysisRequest) -> LLMServiceResponse:
        """
        Analyze gene-disease correlation using the LLM provider
        
        Args:
            request: Analysis request with gene, disease, and parameters
            
        Returns:
            LLMServiceResponse with analysis results or error
        """
        start_time = time.time()
        
        try:
            # Generate prompts
            system_message, user_message = get_analysis_prompt(
                gene=request.gene,
                disease=request.disease,
                context=request.context,
                include_references=request.include_references
            )
            
            # Make API request with retries for validation
            response_text, metadata = await self._make_request_with_validation(
                system_message=system_message,
                user_message=user_message,
                model=request.model,
                max_retries=3
            )
            
            # Parse response
            analysis_data = json.loads(response_text)
            analysis_response = LLMAnalysisResponse(**analysis_data)
            
            # Calculate duration and extract usage stats
            duration = time.time() - start_time
            usage_stats = self._extract_usage_stats(metadata, duration)
            
            return LLMServiceResponse(
                success=True,
                data=analysis_response,
                usage_stats=usage_stats,
                provider_response_id=metadata.get('id')
            )
            
        except LLMAuthenticationError as e:
            return LLMServiceResponse(
                success=False,
                error=LLMError(
                    error_type="AUTHENTICATION_ERROR",
                    message=str(e),
                    provider=self.__class__.__name__,
                    status_code=401
                )
            )
        except LLMRateLimitError as e:
            return LLMServiceResponse(
                success=False,
                error=LLMError(
                    error_type="RATE_LIMIT_ERROR", 
                    message=str(e),
                    provider=self.__class__.__name__,
                    status_code=429,
                    retry_after=60
                )
            )
        except LLMValidationError as e:
            return LLMServiceResponse(
                success=False,
                error=LLMError(
                    error_type="VALIDATION_ERROR",
                    message=str(e),
                    provider=self.__class__.__name__
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error in LLM analysis: {e}", exc_info=True)
            return LLMServiceResponse(
                success=False,
                error=LLMError(
                    error_type="INTERNAL_ERROR",
                    message=f"Unexpected error: {str(e)}",
                    provider=self.__class__.__name__,
                    status_code=500
                )
            )
    
    async def _make_request_with_validation(
        self, 
        system_message: str, 
        user_message: str, 
        model: Optional[str] = None,
        max_retries: int = 3
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Make request with JSON validation and retries
        
        Args:
            system_message: System prompt
            user_message: User prompt
            model: Model to use
            max_retries: Maximum retry attempts for validation
            
        Returns:
            Tuple of (validated_response_text, metadata)
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response_text, metadata = await self._make_request(
                    system_message=system_message,
                    user_message=user_message,
                    model=model
                )
                
                # Validate JSON structure
                if validate_json_response_format(response_text):
                    return response_text, metadata
                else:
                    last_error = f"Invalid JSON format (attempt {attempt + 1})"
                    logger.warning(f"Invalid JSON response, attempt {attempt + 1}: {response_text[:200]}...")
                    
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {str(e)}"
                logger.warning(f"JSON decode error, attempt {attempt + 1}: {e}")
            except Exception as e:
                # Don't retry on non-validation errors
                raise
        
        raise LLMValidationError(f"Failed to get valid JSON response after {max_retries} attempts. Last error: {last_error}")
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()


class OpenAIClient(LLMClient):
    """OpenAI API client implementation"""
    
    def _setup_client(self):
        """Setup OpenAI HTTP client"""
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        self.timeout = self.config.get('timeout', 30)
        self.organization = self.config.get('organization')
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
        
        # Model defaults
        self.default_model = 'gpt-4'
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 2000)
    
    async def _make_request(self, system_message: str, user_message: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Make request to OpenAI API"""
        model = kwargs.get('model', self.default_model)
        
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message}
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'response_format': {'type': 'json_object'} if 'gpt-4' in model else None
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            response = await self.client.post('/chat/completions', json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown OpenAI error')
                error_type = data['error'].get('type', 'unknown')
                
                if 'invalid_api_key' in error_type or 'authentication' in error_type:
                    raise LLMAuthenticationError(f"OpenAI authentication failed: {error_msg}")
                elif 'rate_limit' in error_type:
                    raise LLMRateLimitError(f"OpenAI rate limit exceeded: {error_msg}")
                else:
                    raise LLMProviderError(f"OpenAI API error: {error_msg}")
            
            if not data.get('choices'):
                raise LLMProviderError("No choices returned from OpenAI API")
            
            response_text = data['choices'][0]['message']['content']
            return response_text, data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMAuthenticationError("Invalid OpenAI API key")
            elif e.response.status_code == 429:
                raise LLMRateLimitError("OpenAI rate limit exceeded")
            else:
                raise LLMProviderError(f"OpenAI API HTTP error: {e.response.status_code}")
        except httpx.TimeoutException:
            raise LLMProviderError("OpenAI API request timeout")
        except httpx.RequestError as e:
            raise LLMProviderError(f"OpenAI API request error: {str(e)}")
    
    def _extract_usage_stats(self, response_data: Dict[str, Any], duration: float) -> LLMUsageStats:
        """Extract usage statistics from OpenAI response"""
        usage = response_data.get('usage', {})
        model = response_data.get('model', 'unknown')
        
        return LLMUsageStats(
            provider="openai",
            model=model,
            input_tokens=usage.get('prompt_tokens'),
            output_tokens=usage.get('completion_tokens'),
            total_tokens=usage.get('total_tokens'),
            request_duration=duration
        )


class AnthropicClient(LLMClient):
    """Anthropic API client implementation"""
    
    def _setup_client(self):
        """Setup Anthropic HTTP client"""
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.base_url = self.config.get('base_url', 'https://api.anthropic.com')
        self.timeout = self.config.get('timeout', 30)
        
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
        
        # Model defaults
        self.default_model = 'claude-3-sonnet-20240229'
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 2000)
    
    async def _make_request(self, system_message: str, user_message: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Make request to Anthropic API"""
        model = kwargs.get('model', self.default_model)
        
        payload = {
            'model': model,
            'system': system_message,
            'messages': [
                {'role': 'user', 'content': user_message}
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        try:
            response = await self.client.post('/v1/messages', json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown Anthropic error')
                error_type = data['error'].get('type', 'unknown')
                
                if 'authentication' in error_type or 'invalid_api_key' in error_type:
                    raise LLMAuthenticationError(f"Anthropic authentication failed: {error_msg}")
                elif 'rate_limit' in error_type:
                    raise LLMRateLimitError(f"Anthropic rate limit exceeded: {error_msg}")
                else:
                    raise LLMProviderError(f"Anthropic API error: {error_msg}")
            
            if not data.get('content'):
                raise LLMProviderError("No content returned from Anthropic API")
            
            response_text = data['content'][0]['text']
            return response_text, data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMAuthenticationError("Invalid Anthropic API key")
            elif e.response.status_code == 429:
                raise LLMRateLimitError("Anthropic rate limit exceeded")
            else:
                raise LLMProviderError(f"Anthropic API HTTP error: {e.response.status_code}")
        except httpx.TimeoutException:
            raise LLMProviderError("Anthropic API request timeout")
        except httpx.RequestError as e:
            raise LLMProviderError(f"Anthropic API request error: {str(e)}")
    
    def _extract_usage_stats(self, response_data: Dict[str, Any], duration: float) -> LLMUsageStats:
        """Extract usage statistics from Anthropic response"""
        usage = response_data.get('usage', {})
        model = response_data.get('model', 'unknown')
        
        return LLMUsageStats(
            provider="anthropic",
            model=model,
            input_tokens=usage.get('input_tokens'),
            output_tokens=usage.get('output_tokens'),
            total_tokens=(usage.get('input_tokens', 0) + usage.get('output_tokens', 0)) or None,
            request_duration=duration
        )


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient
    }
    
    @classmethod
    def create_client(cls, provider: LLMProvider, config: Dict[str, Any]) -> LLMClient:
        """Create an LLM client for the specified provider"""
        if provider not in cls._clients:
            raise ValueError(f"Unsupported provider: {provider}")
        
        client_class = cls._clients[provider]
        return client_class(config)
    
    @classmethod
    def register_provider(cls, provider: LLMProvider, client_class: type):
        """Register a new provider client class"""
        cls._clients[provider] = client_class


class LLMService:
    """Main service for LLM operations with caching and configuration management"""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._clients: Dict[LLMProvider, LLMClient] = {}
        self._cache: Dict[str, Tuple[LLMServiceResponse, datetime]] = {}
        
    def _get_client(self, provider: LLMProvider) -> LLMClient:
        """Get or create client for provider"""
        if provider not in self._clients:
            provider_config = self.config.get_provider_config(provider)
            client_config = provider_config.dict()
            self._clients[provider] = LLMClientFactory.create_client(provider, client_config)
        
        return self._clients[provider]
    
    def _get_cache_key(self, request: LLMAnalysisRequest) -> str:
        """Generate cache key for request"""
        return f"{request.provider.value}:{request.gene}:{request.disease}:{request.model or 'default'}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMServiceResponse]:
        """Get cached response if valid"""
        if not self.config.enable_caching:
            return None
            
        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.config.cache_ttl):
                return response
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMServiceResponse):
        """Cache response with timestamp"""
        if self.config.enable_caching:
            self._cache[cache_key] = (response, datetime.utcnow())
    
    async def analyze_gene_disease(self, request: LLMAnalysisRequest) -> LLMServiceResponse:
        """
        Analyze gene-disease correlation with caching
        
        Args:
            request: Analysis request
            
        Returns:
            LLMServiceResponse with analysis results
        """
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Returning cached response for {cache_key}")
            return cached_response
        
        # Get client and make request
        client = self._get_client(request.provider)
        response = await client.analyze_gene_disease(request)
        
        # Cache successful responses
        if response.success:
            self._cache_response(cache_key, response)
        
        return response
    
    async def close(self):
        """Close all clients"""
        for client in self._clients.values():
            await client.close()
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl
        }


# Global LLM service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance"""
    global _llm_service
    
    if _llm_service is None:
        from ..config import settings
        
        # Create provider configuration from settings
        config = LLMProviderConfig(
            openai=OpenAIConfig(
                api_key=settings.OPENAI_API_KEY,
                organization=settings.OPENAI_ORGANIZATION,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            ) if settings.OPENAI_API_KEY else None,
            anthropic=AnthropicConfig(
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            ) if settings.ANTHROPIC_API_KEY else None,
            default_provider=LLMProvider(settings.DEFAULT_LLM_PROVIDER),
            enable_caching=settings.LLM_ENABLE_CACHING,
            cache_ttl=settings.LLM_CACHE_TTL
        )
        
        _llm_service = LLMService(config)
    
    return _llm_service


async def shutdown_llm_service():
    """Shutdown the global LLM service"""
    global _llm_service
    
    if _llm_service:
        await _llm_service.close()
        _llm_service = None