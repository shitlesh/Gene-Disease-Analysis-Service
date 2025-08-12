# LLM Integration Layer - Implementation Guide

## Overview

This document describes the comprehensive LLM (Large Language Model) integration layer implemented for gene-disease correlation analysis. The system provides a provider-agnostic architecture supporting OpenAI and Anthropic APIs with extensible design for additional providers.

## Architecture

### Core Components

1. **Provider Abstraction Layer** (`services/llm_client.py`)
   - Abstract `LLMClient` base class
   - Provider-specific adapters (`OpenAIClient`, `AnthropicClient`)
   - `LLMClientFactory` for provider instantiation
   - `LLMService` for high-level operations

2. **Prompt Engineering System** (`services/prompt_templates.py`)
   - Template-based prompt generation
   - Anti-hallucination instructions
   - JSON-only output enforcement
   - Multiple analysis types support

3. **Data Models** (`models/llm.py`)
   - Pydantic models for request/response validation
   - Provider configuration management
   - Error handling structures
   - Usage statistics tracking

4. **API Endpoints** (`routers/llm_analysis.py`)
   - RESTful API for LLM analysis
   - Provider management endpoints
   - Caching and health monitoring
   - Comprehensive documentation

## Provider Support

### OpenAI Integration
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Features**: JSON mode, function calling, high accuracy
- **Use Cases**: Complex analysis, research queries, clinical applications

### Anthropic Integration  
- **Models**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Features**: Long context, nuanced analysis, safety-focused
- **Use Cases**: Literature review, complex reasoning, uncertainty assessment

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_ORGANIZATION=org-your-organization  # Optional

# Anthropic Configuration  
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# LLM Settings
DEFAULT_LLM_PROVIDER=openai  # or anthropic
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
LLM_ENABLE_CACHING=true
LLM_CACHE_TTL=3600
```

### Programmatic Configuration

```python
from app.models.llm import LLMProviderConfig, OpenAIConfig, AnthropicConfig
from app.services.llm_client import LLMService

config = LLMProviderConfig(
    openai=OpenAIConfig(
        api_key="your-openai-key",
        temperature=0.1,
        max_tokens=2000
    ),
    anthropic=AnthropicConfig(
        api_key="your-anthropic-key",
        temperature=0.1,
        max_tokens=2000
    ),
    default_provider=LLMProvider.OPENAI,
    enable_caching=True,
    cache_ttl=3600
)

llm_service = LLMService(config)
```

## Usage Examples

### Basic Gene-Disease Analysis

```python
from app.models.llm import LLMAnalysisRequest, LLMProvider
from app.services.llm_client import LLMService

# Create analysis request
request = LLMAnalysisRequest(
    gene="BRCA1",
    disease="breast cancer",
    provider=LLMProvider.OPENAI,
    model="gpt-4",  # Optional - uses provider default if not specified
    context="Patient has family history of breast cancer",
    include_references=True
)

# Perform analysis
response = await llm_service.analyze_gene_disease(request)

if response.success:
    analysis = response.data
    print(f"Summary: {analysis.summary}")
    print(f"Confidence: {analysis.confidence_score}")
    
    for assoc in analysis.associations:
        print(f"Mechanism: {assoc.mechanism}")
        print(f"Evidence: {assoc.evidence_level}")
else:
    error = response.error
    print(f"Error: {error.message}")
```

### API Endpoint Usage

```bash
# Perform gene-disease analysis
curl -X POST "http://localhost:8000/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "gene": "BRCA1",
    "disease": "breast cancer", 
    "provider": "openai",
    "model": "gpt-4",
    "context": "Family history present",
    "include_references": false
  }'

# Get provider status
curl "http://localhost:8000/llm/providers"

# Get available models for a provider
curl "http://localhost:8000/llm/models/openai"

# Get cache statistics
curl "http://localhost:8000/llm/cache/stats"

# Clear analysis cache
curl -X DELETE "http://localhost:8000/llm/cache"
```

## Response Format

The system returns structured JSON responses with the following format:

```json
{
  "success": true,
  "data": {
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
    "confidence_score": 0.95,
    "references": ["Author et al. Journal (Year)"]
  },
  "usage_stats": {
    "provider": "openai",
    "model": "gpt-4",
    "input_tokens": 150,
    "output_tokens": 200,
    "total_tokens": 350,
    "request_duration": 2.5
  },
  "provider_response_id": "chatcmpl-123456"
}
```

## Prompt Engineering

### Anti-Hallucination Features

1. **Explicit Instructions**: Clear directives to base responses only on established scientific evidence
2. **Uncertainty Acknowledgment**: Requirements to state when evidence is limited or conflicting
3. **Conservative Approach**: Instructions to use "unknown" or "insufficient evidence" when appropriate
4. **Evidence Level Classification**: Structured classification of evidence strength

### Template System

The system uses a template-based approach for consistent prompt generation:

```python
from app.services.prompt_templates import get_analysis_prompt

system_message, user_message = get_analysis_prompt(
    gene="BRCA1",
    disease="breast cancer",
    template_name="correlation",  # or "gene_function", "disease_genetics"
    context="Additional context",
    include_references=True
)
```

### Evidence Levels

- **Strong**: Multiple independent studies, functional validation, clinical guidelines
- **Moderate**: Some studies with consistent findings, plausible mechanism
- **Weak**: Limited studies, preliminary evidence, case reports
- **Insufficient**: No clear evidence or conflicting data

## Error Handling

### Error Types

1. **Authentication Errors**: Invalid API keys or authentication failures
2. **Rate Limit Errors**: API quota exceeded
3. **Validation Errors**: Invalid JSON response format after retries
4. **Provider Errors**: Service unavailable or HTTP errors
5. **Internal Errors**: Unexpected system failures

### Retry Logic

- **JSON Validation**: Automatic retry up to 3 times for invalid JSON responses
- **Rate Limiting**: Configurable backoff for rate limit errors
- **Graceful Degradation**: Detailed error reporting without system crashes

### Error Response Format

```json
{
  "success": false,
  "error": {
    "error_type": "RATE_LIMIT_ERROR",
    "message": "OpenAI rate limit exceeded",
    "provider": "OpenAIClient", 
    "status_code": 429,
    "retry_after": 60
  }
}
```

## Performance Features

### Caching System

- **In-Memory Cache**: Fast response for repeated queries
- **TTL-Based Expiration**: Configurable cache lifetime
- **Cache Key Generation**: Based on provider, gene, disease, and model
- **Cache Statistics**: Monitoring and performance metrics

### Performance Characteristics

- **Cached Responses**: Sub-second response time
- **Uncached Responses**: 2-10 seconds depending on provider and complexity
- **Cost Optimization**: Significant reduction in API costs for repeated queries
- **Scalability**: Stateless design supports horizontal scaling

## Extensibility

### Adding New Providers

1. **Create Client Class**: Inherit from `LLMClient` and implement abstract methods
2. **Register Provider**: Add to `LLMClientFactory._clients` registry
3. **Add Configuration**: Create provider-specific configuration model
4. **Update Endpoints**: Add provider to API documentation and model lists

Example new provider implementation:

```python
class CustomLLMClient(LLMClient):
    def _setup_client(self):
        # Initialize HTTP client and configuration
        pass
    
    async def _make_request(self, system_message: str, user_message: str, **kwargs):
        # Implement provider-specific API call
        pass
    
    def _extract_usage_stats(self, response_data: Dict[str, Any], duration: float):
        # Extract usage statistics from response
        pass

# Register the new provider
LLMClientFactory.register_provider(LLMProvider.CUSTOM, CustomLLMClient)
```

### Custom Templates

```python
from app.services.prompt_templates import PromptTemplate, template_registry

class CustomAnalysisTemplate(PromptTemplate):
    def format_system_message(self, **kwargs):
        return "Custom system message"
    
    def format_user_message(self, **kwargs):
        return "Custom user message"

# Register custom template
template_registry.register_template('custom_analysis', CustomAnalysisTemplate())
```

## Testing

### Unit Tests

The system includes comprehensive unit tests with mocked provider APIs:

```bash
# Run LLM-specific tests
poetry run pytest app/tests/test_llm_client.py -v

# Run integration tests
poetry run python test_llm_integration.py
```

### Test Coverage

- Provider client implementations
- Request/response validation
- Error handling scenarios
- Caching functionality
- Prompt template system
- JSON validation logic

## Security Considerations

### API Key Management

- Environment variable configuration
- No hardcoded credentials
- Secure key rotation support
- Organization-level access control (OpenAI)

### Input Validation

- Gene name format validation and normalization
- Disease name sanitization
- Context length limits
- Provider-specific parameter validation

### Rate Limiting Protection

- Built-in retry logic with exponential backoff
- Rate limit error handling
- Usage monitoring and statistics
- Cost optimization through caching

## Monitoring and Observability

### Usage Statistics

The system tracks comprehensive usage metrics:

```python
usage_stats = response.usage_stats
print(f"Provider: {usage_stats.provider}")
print(f"Model: {usage_stats.model}")
print(f"Tokens: {usage_stats.total_tokens}")
print(f"Duration: {usage_stats.request_duration}")
```

### Health Monitoring

```bash
# Check service health
curl "http://localhost:8000/llm/health"

# Monitor cache performance
curl "http://localhost:8000/llm/cache/stats"
```

### Logging

The system provides structured logging for:
- Request/response cycles
- Error conditions and retries
- Cache hit/miss statistics
- Performance metrics
- Provider-specific events

## Production Deployment

### Environment Setup

1. **API Keys**: Configure provider API keys securely
2. **Resource Limits**: Set appropriate memory and CPU limits
3. **Rate Limiting**: Implement application-level rate limiting if needed
4. **Monitoring**: Set up logging and metrics collection
5. **Caching**: Consider external cache (Redis) for multi-instance deployments

### Scaling Considerations

- **Horizontal Scaling**: Stateless design supports multiple instances
- **Load Balancing**: Distribute requests across instances
- **Cache Sharing**: Use external cache for shared state
- **Provider Failover**: Implement provider switching for reliability

## Conclusion

This LLM integration layer provides a robust, scalable, and extensible foundation for AI-powered gene-disease correlation analysis. The provider-agnostic architecture, comprehensive error handling, and performance optimizations make it suitable for both research and production environments.

The system's modular design enables easy addition of new providers and analysis types while maintaining consistency in the API interface and response format. The anti-hallucination prompt engineering ensures reliable, evidence-based analysis results for scientific and clinical applications.