#!/usr/bin/env python3
"""
Basic functionality test - testing core components without external API calls
"""

def test_data_transformer():
    """Test data transformation utilities"""
    from app.utils.data_transformer import DataTransformer
    
    print("Testing Data Transformation utilities...")
    
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

def test_models():
    """Test Pydantic models"""
    from app.models.nhs_scotland import DatasetInfo, DatasetSearchResult, GeneSearchRequest
    
    print("\nTesting Pydantic Models...")
    
    # Test DatasetInfo
    dataset = DatasetInfo(
        name="test-dataset",
        title="Test Dataset", 
        notes="Test description",
        resources=1
    )
    assert dataset.name == "test-dataset"
    print("DatasetInfo model works")
    
    # Test DatasetSearchResult
    search_result = DatasetSearchResult(
        name="search-result",
        title="Search Result",
        notes="Search description", 
        score=0.8,
        resources=2
    )
    assert search_result.score == 0.8
    print("DatasetSearchResult model works")
    
    # Test GeneSearchRequest
    gene_request = GeneSearchRequest(gene_name="BRCA1")
    assert gene_request.gene_name == "BRCA1"
    print("GeneSearchRequest model works")
    
    print("Pydantic models working correctly!")

def test_app_import():
    """Test that the FastAPI app can be imported"""
    from app.main import app
    from app.routers import nhs_scotland
    
    print("\nTesting FastAPI Application...")
    
    # Test app import
    assert app is not None
    print("FastAPI app imports successfully")
    
    # Test router import
    assert nhs_scotland.router is not None
    print("NHS Scotland router imports successfully")
    
    print("FastAPI application components working!")

def test_service_imports():
    """Test that services can be imported"""
    from app.services.nhs_scotland_service import NHSScotlandService, nhs_scotland_service
    
    print("\nTesting Service Imports...")
    
    # Test service class
    service = NHSScotlandService()
    assert service is not None
    print("NHSScotlandService class works")
    
    # Test service singleton
    assert nhs_scotland_service is not None
    print("NHS Scotland service singleton works")
    
    print("Service imports working!")

if __name__ == "__main__":
    print("Running Basic Component Tests\n")
    
    try:
        test_data_transformer()
        test_models()
        test_app_import()
        test_service_imports()
        
        print(f"\n" + "="*60)
        print("ALL BASIC TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("Data transformation utilities functional")
        print("Pydantic models working")
        print("FastAPI application imports")
        print("NHS Scotland service imports")
        print("All core components ready")
        
    except Exception as e:
        print(f"\nBasic test failed: {e}")
        import traceback
        traceback.print_exc()
        raise