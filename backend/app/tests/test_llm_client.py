"""
Unit tests for LLM client integration layer
Tests provider-specific adapters with mocked APIs
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.models.llm import (
    LLMProvider, LLMAnalysisRequest, LLMAnalysisResponse, LLMProviderConfig, 
    OpenAIConfig, AnthropicConfig
)
from app.services.llm_client import (
    LLMService, OpenAIClient, AnthropicClient, LLMClientFactory,
    LLMAuthenticationError, LLMRateLimitError, LLMValidationError
)


class TestLLMModels:
    """Test Pydantic models for LLM integration"""
    
    def test_llm_analysis_request_validation(self):
        """Test request model validation"""
        # Valid request
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        assert request.gene == "BRCA1"
        assert request.disease == "breast cancer"
        assert request.provider == LLMProvider.OPENAI
        
        # Gene name normalization
        request = LLMAnalysisRequest(
            gene="  brca1  ",
            disease="cancer",
            provider=LLMProvider.OPENAI
        )
        assert request.gene == "BRCA1"
        
        # Disease name normalization
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="  Breast Cancer  ",
            provider=LLMProvider.OPENAI
        )
        assert request.disease == "breast cancer"
    
    def test_llm_analysis_response_structure(self):
        """Test response model structure"""
        response_data = {
            "summary": "BRCA1 is strongly associated with breast cancer",
            "associations": [
                {
                    "gene_function": "DNA repair",
                    "mechanism": "Loss of function",
                    "evidence_level": "strong",
                    "phenotypes": ["breast cancer", "ovarian cancer"],
                    "inheritance_pattern": "autosomal dominant"
                }
            ],
            "recommendation": "Genetic counseling recommended",
            "uncertainty": "Penetrance varies",
            "confidence_score": 0.9
        }
        
        response = LLMAnalysisResponse(**response_data)
        assert response.summary == "BRCA1 is strongly associated with breast cancer"
        assert len(response.associations) == 1
        assert response.confidence_score == 0.9


class TestPromptTemplates:
    """Test prompt template system"""
    
    def test_get_analysis_prompt(self):
        """Test prompt generation"""
        from app.services.prompt_templates import get_analysis_prompt
        
        system, user = get_analysis_prompt(
            gene="BRCA1",
            disease="breast cancer",
            context="Family history present"
        )
        
        assert "JSON" in system
        assert "BRCA1" in user
        assert "breast cancer" in user
        assert "Family history present" in user
    
    def test_validate_json_response_format(self):
        """Test JSON response validation"""
        from app.services.prompt_templates import validate_json_response_format
        
        # Valid response
        valid_response = json.dumps({
            "summary": "Test summary",
            "associations": [
                {
                    "gene_function": "Test function",
                    "mechanism": "Test mechanism", 
                    "evidence_level": "strong",
                    "phenotypes": ["test"]
                }
            ],
            "recommendation": "Test recommendation",
            "uncertainty": "Test uncertainty"
        })
        
        assert validate_json_response_format(valid_response) is True
        
        # Invalid response - missing fields
        invalid_response = json.dumps({"summary": "Test only"})
        assert validate_json_response_format(invalid_response) is False
        
        # Invalid JSON
        assert validate_json_response_format("not json") is False


class TestOpenAIClient:
    """Test OpenAI client implementation"""
    
    @pytest.fixture
    def openai_config(self):
        return {
            'api_key': 'test-api-key',
            'temperature': 0.1,
            'max_tokens': 2000
        }
    
    @pytest.fixture
    def mock_openai_response(self):
        return {
            'id': 'test-response-id',
            'model': 'gpt-4',
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "summary": "BRCA1 test summary",
                            "associations": [
                                {
                                    "gene_function": "DNA repair",
                                    "mechanism": "Homologous recombination",
                                    "evidence_level": "strong", 
                                    "phenotypes": ["breast cancer"],
                                    "inheritance_pattern": "autosomal dominant"
                                }
                            ],
                            "recommendation": "Test recommendation",
                            "uncertainty": "Test uncertainty",
                            "confidence_score": 0.9
                        })
                    }
                }
            ],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 150,
                'total_tokens': 250
            }
        }
    
    @pytest.mark.asyncio
    async def test_openai_successful_request(self, openai_config, mock_openai_response):
        """Test successful OpenAI API request"""
        client = OpenAIClient(openai_config)
        
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_openai_response
        mock_response.raise_for_status.return_value = None
        
        client.client.post = AsyncMock(return_value=mock_response)
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer", 
            provider=LLMProvider.OPENAI
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is True
        assert response.data is not None
        assert response.data.summary == "BRCA1 test summary"
        assert response.usage_stats.provider == "openai"
        assert response.usage_stats.total_tokens == 250
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_openai_authentication_error(self, openai_config):
        """Test OpenAI authentication error handling"""
        client = OpenAIClient(openai_config)
        
        # Mock authentication error
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized", 
                request=MagicMock(), 
                response=mock_response
            )
        )
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is False
        assert response.error.error_type == "AUTHENTICATION_ERROR"
        assert response.error.status_code == 401
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self, openai_config):
        """Test OpenAI rate limit error handling"""
        client = OpenAIClient(openai_config)
        
        # Mock rate limit error
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 429
        
        client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=MagicMock(),
                response=mock_response
            )
        )
        
        request = LLMAnalysisRequest(
            gene="BRCA1", 
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is False
        assert response.error.error_type == "RATE_LIMIT_ERROR"
        assert response.error.retry_after == 60
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_openai_invalid_json_retry(self, openai_config):
        """Test retry logic for invalid JSON responses"""
        client = OpenAIClient(openai_config)
        
        # Mock responses - first invalid, second valid
        invalid_response = {
            'choices': [{'message': {'content': 'invalid json'}}],
            'usage': {'total_tokens': 100}
        }
        
        valid_response = {
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "summary": "Valid response",
                            "associations": [
                                {
                                    "gene_function": "Test",
                                    "mechanism": "Test",
                                    "evidence_level": "strong",
                                    "phenotypes": ["test"]
                                }
                            ],
                            "recommendation": "Test",
                            "uncertainty": "Test"
                        })
                    }
                }
            ],
            'usage': {'total_tokens': 150}
        }
        
        mock_responses = [
            MagicMock(json=lambda: invalid_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: valid_response, raise_for_status=lambda: None)
        ]
        
        client.client.post = AsyncMock(side_effect=mock_responses)
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer", 
            provider=LLMProvider.OPENAI
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is True
        assert client.client.post.call_count == 2  # Retried once
        
        await client.close()


class TestAnthropicClient:
    """Test Anthropic client implementation"""
    
    @pytest.fixture
    def anthropic_config(self):
        return {
            'api_key': 'test-anthropic-key',
            'temperature': 0.1,
            'max_tokens': 2000
        }
    
    @pytest.fixture 
    def mock_anthropic_response(self):
        return {
            'id': 'test-anthropic-id',
            'model': 'claude-3-sonnet-20240229',
            'content': [
                {
                    'text': json.dumps({
                        "summary": "Claude test summary",
                        "associations": [
                            {
                                "gene_function": "Tumor suppressor",
                                "mechanism": "DNA damage response",
                                "evidence_level": "strong",
                                "phenotypes": ["breast cancer", "ovarian cancer"],
                                "inheritance_pattern": "autosomal dominant"
                            }
                        ],
                        "recommendation": "Claude recommendation",
                        "uncertainty": "Claude uncertainty",
                        "confidence_score": 0.85
                    })
                }
            ],
            'usage': {
                'input_tokens': 80,
                'output_tokens': 120
            }
        }
    
    @pytest.mark.asyncio
    async def test_anthropic_successful_request(self, anthropic_config, mock_anthropic_response):
        """Test successful Anthropic API request"""
        client = AnthropicClient(anthropic_config)
        
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_anthropic_response
        mock_response.raise_for_status.return_value = None
        
        client.client.post = AsyncMock(return_value=mock_response)
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.ANTHROPIC
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is True
        assert response.data.summary == "Claude test summary"
        assert response.usage_stats.provider == "anthropic"
        assert response.usage_stats.input_tokens == 80
        assert response.usage_stats.output_tokens == 120
        
        await client.close()


class TestLLMService:
    """Test main LLM service with caching"""
    
    @pytest.fixture
    def service_config(self):
        return LLMProviderConfig(
            openai=OpenAIConfig(api_key="test-openai-key"),
            anthropic=AnthropicConfig(api_key="test-anthropic-key"),
            default_provider=LLMProvider.OPENAI,
            enable_caching=True,
            cache_ttl=3600
        )
    
    @pytest.mark.asyncio
    async def test_service_caching(self, service_config):
        """Test response caching functionality"""
        service = LLMService(service_config)
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "summary": "Cached response",
                            "associations": [
                                {
                                    "gene_function": "Test",
                                    "mechanism": "Test", 
                                    "evidence_level": "strong",
                                    "phenotypes": ["test"]
                                }
                            ],
                            "recommendation": "Test",
                            "uncertainty": "Test"
                        })
                    }
                }
            ],
            'usage': {'total_tokens': 100}
        }
        mock_response.raise_for_status.return_value = None
        
        # Mock the client creation and HTTP calls
        with patch.object(LLMClientFactory, 'create_client') as mock_factory:
            mock_client = AsyncMock()
            mock_client.client.post = AsyncMock(return_value=mock_response)
            mock_client.analyze_gene_disease = AsyncMock(
                return_value=MagicMock(success=True, data=MagicMock())
            )
            mock_factory.return_value = mock_client
            
            request = LLMAnalysisRequest(
                gene="BRCA1",
                disease="breast cancer",
                provider=LLMProvider.OPENAI
            )
            
            # First request - should hit API
            response1 = await service.analyze_gene_disease(request)
            
            # Second request - should use cache
            response2 = await service.analyze_gene_disease(request)
            
            # Should only create client once
            mock_factory.assert_called_once()
            
            await service.close()
    
    def test_cache_key_generation(self, service_config):
        """Test cache key generation"""
        service = LLMService(service_config)
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        
        cache_key = service._get_cache_key(request)
        expected = "openai:BRCA1:breast cancer:default"
        
        assert cache_key == expected
    
    def test_cache_stats(self, service_config):
        """Test cache statistics"""
        service = LLMService(service_config)
        
        stats = service.get_cache_stats()
        
        assert stats["cache_enabled"] is True
        assert stats["cache_ttl"] == 3600
        assert stats["cache_size"] == 0


class TestLLMClientFactory:
    """Test LLM client factory"""
    
    def test_create_openai_client(self):
        """Test OpenAI client creation"""
        config = {'api_key': 'test-key'}
        client = LLMClientFactory.create_client(LLMProvider.OPENAI, config)
        
        assert isinstance(client, OpenAIClient)
        assert client.api_key == 'test-key'
    
    def test_create_anthropic_client(self):
        """Test Anthropic client creation"""
        config = {'api_key': 'test-key'}
        client = LLMClientFactory.create_client(LLMProvider.ANTHROPIC, config)
        
        assert isinstance(client, AnthropicClient)
        assert client.api_key == 'test-key'
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClientFactory.create_client("unsupported", {})


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test missing API key handling"""
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIClient({})
    
    @pytest.mark.asyncio
    async def test_provider_config_validation(self):
        """Test provider configuration validation"""
        config = LLMProviderConfig(default_provider=LLMProvider.OPENAI)
        
        with pytest.raises(ValueError, match="OpenAI configuration not provided"):
            config.get_provider_config(LLMProvider.OPENAI)
    
    @pytest.mark.asyncio
    async def test_json_validation_failure(self):
        """Test JSON validation failure after retries"""
        config = {'api_key': 'test-key'}
        client = OpenAIClient(config)
        
        # Mock invalid JSON responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'invalid json'}}],
            'usage': {'total_tokens': 100}
        }
        mock_response.raise_for_status.return_value = None
        
        client.client.post = AsyncMock(return_value=mock_response)
        
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        
        response = await client.analyze_gene_disease(request)
        
        assert response.success is False
        assert response.error.error_type == "VALIDATION_ERROR"
        
        await client.close()