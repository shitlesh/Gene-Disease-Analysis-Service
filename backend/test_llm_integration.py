#!/usr/bin/env python3
"""
Integration test for LLM gene-disease analysis
Tests basic functionality without making actual LLM API calls
"""

from fastapi.testclient import TestClient
from app.main import app


def test_llm_endpoints():
    """Test LLM analysis endpoints for basic functionality"""
    client = TestClient(app)
    
    print("Testing LLM Analysis API Integration...")
    
    # Test root endpoint
    response = client.get("/llm/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "LLM-Powered Gene-Disease Analysis" in data["service"]
    print("LLM analysis info endpoint works")
    
    # Test providers endpoint
    response = client.get("/llm/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    print("LLM providers endpoint works")
    
    # Test models endpoint for OpenAI
    response = client.get("/llm/models/openai")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai"
    assert "models" in data
    print("OpenAI models endpoint works")
    
    # Test models endpoint for Anthropic
    response = client.get("/llm/models/anthropic")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"
    assert "models" in data
    print("Anthropic models endpoint works")
    
    # Test cache stats endpoint
    response = client.get("/llm/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "cache_statistics" in data
    print("Cache statistics endpoint works")
    
    # Test health endpoint
    response = client.get("/llm/health")
    assert response.status_code == 200 or response.status_code == 503  # May be unhealthy without API keys
    print("LLM health endpoint works")
    
    print("\nLLM Analysis integration tests passed!")
    print("All endpoints are properly configured and accessible")


def test_llm_models_validation():
    """Test LLM models and providers validation"""
    from app.models.llm import LLMProvider, LLMAnalysisRequest
    from app.services.llm_client import LLMClientFactory
    
    print("\nTesting LLM Models and Validation...")
    
    # Test provider enum
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
    print("LLM provider enums work")
    
    # Test request validation
    request = LLMAnalysisRequest(
        gene="BRCA1",
        disease="breast cancer",
        provider=LLMProvider.OPENAI
    )
    assert request.gene == "BRCA1"
    assert request.disease == "breast cancer"
    print("LLM request validation works")
    
    # Test gene name normalization
    request = LLMAnalysisRequest(
        gene="  brca1  ",
        disease="cancer",
        provider=LLMProvider.OPENAI
    )
    assert request.gene == "BRCA1"
    print("Gene name normalization works")
    
    print("LLM models and validation working correctly!")


def test_prompt_templates():
    """Test prompt template system"""
    from app.services.prompt_templates import get_analysis_prompt, validate_json_response_format
    import json
    
    print("\nTesting Prompt Templates...")
    
    # Test prompt generation
    system, user = get_analysis_prompt(
        gene="BRCA1",
        disease="breast cancer",
        context="Family history of cancer"
    )
    
    assert "JSON" in system
    assert "BRCA1" in user
    assert "breast cancer" in user
    assert "Family history of cancer" in user
    print("Prompt generation works")
    
    # Test JSON validation
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
    print("JSON response validation works")
    
    # Test invalid JSON
    assert validate_json_response_format("invalid json") is False
    print("Invalid JSON detection works")
    
    print("Prompt templates working correctly!")


if __name__ == "__main__":
    print("Running LLM Integration Tests\n")
    
    try:
        test_llm_endpoints()
        test_llm_models_validation()
        test_prompt_templates()
        
        print(f"\n" + "="*60)
        print("ALL LLM INTEGRATION TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("FastAPI LLM endpoints configured")
        print("Provider abstraction working")
        print("Request/response validation functional")
        print("Prompt templates operational")
        print("JSON validation system active")
        
        print(f"\nLLM Analysis Features Ready:")
        print("Multi-provider support (OpenAI, Anthropic)")
        print("Structured gene-disease correlation analysis")
        print("Anti-hallucination prompt engineering")
        print("Response caching and error handling")
        print("Extensible architecture for new providers")
        
    except Exception as e:
        print(f"\nLLM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise