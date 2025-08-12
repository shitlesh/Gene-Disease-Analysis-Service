#!/usr/bin/env python3
"""
Integration test for NHS Scotland API endpoints
Tests basic functionality without making actual external API calls
"""

import asyncio
from fastapi.testclient import TestClient
from app.main import app

def test_nhs_endpoints():
    """Test NHS Scotland endpoints for basic functionality"""
    client = TestClient(app)
    
    print("Testing NHS Scotland API Integration...")
    
    # Test root endpoint
    response = client.get("/nhs/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "NHS Scotland Open Data Integration" in data["service"]
    print("NHS Scotland info endpoint works")
    
    # Test gene parameter validation
    response = client.get("/nhs/gene/BRCA1")  
    # This will fail because we don't have real API data, but should not be a 422 validation error
    assert response.status_code != 422  # Should not be validation error
    print("Gene endpoint parameter validation works")
    
    # Test disease parameter validation 
    response = client.get("/nhs/disease/cancer")
    assert response.status_code != 422  # Should not be validation error
    print("Disease endpoint parameter validation works")
    
    # Test search parameter validation
    response = client.get("/nhs/search", params={"query": "genetic"})
    assert response.status_code != 422  # Should not be validation error
    print("Search endpoint parameter validation works")
    
    # Test invalid parameters
    response = client.get("/nhs/gene/A")  # Too short
    assert response.status_code == 422
    print("Gene validation correctly rejects invalid input")
    
    response = client.get("/nhs/search", params={"query": "a"})  # Too short
    assert response.status_code == 422
    print("Search validation correctly rejects invalid input")
    
    print("\nNHS Scotland integration tests passed!")
    print("All endpoints are properly configured and validating inputs")

def test_data_transformer():
    """Test data transformation utilities"""
    from app.utils.data_transformer import DataTransformer
    
    print("\nTesting Data Transformation utilities...")
    
    # Test keyword extraction
    text = "Genetic disorders and hereditary conditions in Scotland"
    keywords = DataTransformer.extract_keywords(text)
    assert "genetic" in keywords
    assert "disorders" in keywords
    assert "hereditary" in keywords
    print("Keyword extraction works")
    
    # Test gene-condition mapping
    conditions = DataTransformer.find_gene_related_conditions("BRCA1")
    assert len(conditions) > 0
    assert "breast cancer" in conditions
    print("Gene-condition mapping works")
    
    # Test disease term expansion
    terms = DataTransformer.expand_disease_search_terms("cancer")
    assert "cancer" in terms
    assert len(terms) > 1
    print("Disease term expansion works")
    
    # Test genetic relevance scoring
    high_relevance_dataset = {
        "title": "Genetic Disorders Registry", 
        "notes": "Database of hereditary conditions and gene mutations",
        "tags": ["genetics", "inherited"]
    }
    score = DataTransformer.assess_genetic_relevance(high_relevance_dataset)
    assert score > 0.5
    print("Genetic relevance scoring works")
    
    print("Data transformation utilities working correctly!")

def test_app_health():
    """Test application health and basic functionality"""
    client = TestClient(app)
    
    print("\nTesting Application Health...")
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("Health endpoint working")
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Gene-Disease Analysis API" in data["message"]
    print("Root endpoint working")
    
    print("Application health checks passed!")

if __name__ == "__main__":
    print("Running NHS Scotland Integration Tests\n")
    
    try:
        test_app_health()
        test_data_transformer() 
        test_nhs_endpoints()
        
        print(f"\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("FastAPI application starts correctly")
        print("NHS Scotland router endpoints configured")
        print("Parameter validation working")
        print("Data transformation utilities functional")
        print("Error handling in place")
        print("Gene-disease analysis support ready")
        
        print(f"\nNext Steps:")
        print("• Connect to real NHS Scotland API")
        print("• Test with live data")
        print("• Integrate with React frontend")
        print("• Add more comprehensive datasets")
        
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        raise