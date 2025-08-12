"""
Unit tests for data transformation utilities
"""

import pytest
from datetime import datetime
from app.utils.data_transformer import DataTransformer, data_transformer


class TestDataTransformer:
    """Test cases for data transformation utilities"""

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing"""
        return {
            "name": "test-dataset",
            "title": "  test   dataset   title  ",
            "notes": "This is a genetic condition dataset with cancer information and hereditary disorders",
            "tags": ["genetics", "cancer", "health"],
            "metadata_created": "2023-01-01T10:00:00Z",
            "metadata_modified": "2023-06-01T15:30:00Z",
            "organization": "NHS Scotland"
        }

    def test_clean_dataset_title(self):
        """Test dataset title cleaning"""
        # Test normal case
        assert DataTransformer.clean_dataset_title("test dataset") == "Test Dataset"
        
        # Test extra whitespace
        assert DataTransformer.clean_dataset_title("  multiple   spaces  ") == "Multiple Spaces"
        
        # Test empty string
        assert DataTransformer.clean_dataset_title("") == "Unknown Dataset"
        
        # Test None
        assert DataTransformer.clean_dataset_title(None) == "Unknown Dataset"

    def test_extract_keywords(self):
        """Test keyword extraction from text"""
        text = "This dataset contains information about genetic disorders and cancer research data"
        keywords = DataTransformer.extract_keywords(text)
        
        assert "genetic" in keywords
        assert "disorders" in keywords
        assert "cancer" in keywords
        assert "research" in keywords
        
        # Should filter out common words
        assert "this" not in keywords
        assert "and" not in keywords

    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text"""
        assert DataTransformer.extract_keywords("") == []
        assert DataTransformer.extract_keywords(None) == []

    def test_assess_genetic_relevance(self):
        """Test genetic relevance scoring"""
        # High relevance dataset
        high_relevance = {
            "title": "Genetic Disorders Dataset",
            "notes": "Contains information about hereditary conditions and gene mutations",
            "tags": ["genomic", "inherited"]
        }
        score = DataTransformer.assess_genetic_relevance(high_relevance)
        assert score > 0.5

        # Low relevance dataset
        low_relevance = {
            "title": "Weather Data",
            "notes": "Daily temperature and rainfall measurements",
            "tags": ["climate", "weather"]
        }
        score = DataTransformer.assess_genetic_relevance(low_relevance)
        assert score < 0.1

        # Medium relevance dataset
        medium_relevance = {
            "title": "Birth Defects Registry",
            "notes": "Data about congenital conditions in newborns",
            "tags": ["birth", "screening"]
        }
        score = DataTransformer.assess_genetic_relevance(medium_relevance)
        assert 0.1 < score < 0.5

    def test_identify_disease_categories(self):
        """Test disease category identification"""
        dataset = {
            "title": "Cancer and Heart Disease Study",
            "notes": "Research on cancer incidence and cardiovascular conditions with genetic factors",
            "tags": ["oncology", "cardiac"]
        }
        
        categories = DataTransformer.identify_disease_categories(dataset)
        
        assert "Cancer" in categories
        assert "Cardiovascular" in categories
        assert "Genetic Disorders" in categories

    def test_find_gene_related_conditions(self):
        """Test gene-condition association lookup"""
        # Known gene
        conditions = DataTransformer.find_gene_related_conditions("BRCA1")
        assert "breast cancer" in conditions
        assert "ovarian cancer" in conditions

        # Unknown gene
        conditions = DataTransformer.find_gene_related_conditions("UNKNOWN_GENE")
        assert len(conditions) == 0

        # Case insensitive
        conditions = DataTransformer.find_gene_related_conditions("brca1")
        assert "breast cancer" in conditions

    def test_expand_disease_search_terms(self):
        """Test disease search term expansion"""
        terms = DataTransformer.expand_disease_search_terms("cancer")
        
        assert "cancer" in terms
        assert "carcinoma" in terms
        assert "malignancy" in terms
        assert "tumor" in terms

        # Test with heart disease synonym
        terms = DataTransformer.expand_disease_search_terms("heart disease")
        assert "cardiovascular" in terms
        assert "cardiac" in terms

    def test_enhance_dataset_info(self, sample_dataset):
        """Test dataset enhancement"""
        enhanced = DataTransformer.enhance_dataset_info(sample_dataset)
        
        # Check basic fields preserved
        assert enhanced["name"] == sample_dataset["name"]
        assert enhanced["organization"] == sample_dataset["organization"]
        
        # Check enhancements added
        assert "keywords" in enhanced
        assert "genetic_relevance" in enhanced
        assert "disease_categories" in enhanced
        assert "metadata_created_formatted" in enhanced
        assert "metadata_modified_formatted" in enhanced
        
        # Check title was cleaned
        assert enhanced["title"] == "Test Dataset Title"
        
        # Check genetic relevance is calculated
        assert isinstance(enhanced["genetic_relevance"], float)
        assert 0.0 <= enhanced["genetic_relevance"] <= 1.0

    def test_filter_datasets_by_relevance(self):
        """Test filtering datasets by relevance score"""
        datasets = [
            {"name": "high", "genetic_relevance": 0.8},
            {"name": "medium", "genetic_relevance": 0.5},
            {"name": "low", "genetic_relevance": 0.05},
            {"name": "none", "genetic_relevance": 0.0}
        ]
        
        # Filter with 0.1 threshold
        filtered = DataTransformer.filter_datasets_by_relevance(datasets, min_relevance=0.1)
        names = [d["name"] for d in filtered]
        
        assert "high" in names
        assert "medium" in names
        assert "low" not in names
        assert "none" not in names
        
        # Check sorting (highest first)
        assert filtered[0]["name"] == "high"
        assert filtered[1]["name"] == "medium"

    def test_format_for_frontend(self, sample_dataset):
        """Test frontend formatting"""
        enhanced_dataset = DataTransformer.enhance_dataset_info(sample_dataset)
        formatted = DataTransformer.format_for_frontend([enhanced_dataset])
        
        assert len(formatted) == 1
        frontend_dataset = formatted[0]
        
        # Check required frontend fields
        required_fields = [
            'id', 'title', 'description', 'keywords', 'categories', 
            'relevance', 'organization', 'lastUpdated', 'tags'
        ]
        
        for field in required_fields:
            assert field in frontend_dataset
        
        # Check field types
        assert isinstance(frontend_dataset['relevance'], float)
        assert isinstance(frontend_dataset['keywords'], list)
        assert isinstance(frontend_dataset['categories'], list)
        assert isinstance(frontend_dataset['tags'], list)
        
        # Check description truncation (if applicable)
        if len(sample_dataset.get('notes', '')) > 300:
            assert frontend_dataset['description'].endswith('...')

    def test_create_analysis_suggestions_gene(self):
        """Test analysis suggestions for gene input"""
        suggestions = DataTransformer.create_analysis_suggestions(gene_name="BRCA1")
        
        assert "search_strategies" in suggestions
        assert "related_datasets" in suggestions
        assert "analysis_approaches" in suggestions
        assert "data_limitations" in suggestions
        
        # Check gene-specific content
        assert len(suggestions["search_strategies"]) > 0
        assert "breast cancer" in str(suggestions["related_datasets"])
        assert len(suggestions["analysis_approaches"]) > 0

    def test_create_analysis_suggestions_disease(self):
        """Test analysis suggestions for disease input"""
        suggestions = DataTransformer.create_analysis_suggestions(disease_name="cancer")
        
        assert "search_strategies" in suggestions
        assert "analysis_approaches" in suggestions
        
        # Check disease-specific content
        strategy_text = str(suggestions["search_strategies"])
        assert "carcinoma" in strategy_text or "malignancy" in strategy_text

    def test_create_analysis_suggestions_both(self):
        """Test analysis suggestions for both gene and disease"""
        suggestions = DataTransformer.create_analysis_suggestions(
            gene_name="BRCA1", 
            disease_name="breast cancer"
        )
        
        # Should have suggestions for both
        assert len(suggestions["search_strategies"]) >= 2
        assert len(suggestions["analysis_approaches"]) > 0

    def test_global_transformer_instance(self):
        """Test that global transformer instance is available"""
        assert data_transformer is not None
        assert isinstance(data_transformer, DataTransformer)

    def test_gene_condition_mapping_coverage(self):
        """Test that gene-condition mapping covers important genes"""
        important_genes = ["BRCA1", "BRCA2", "TP53", "CFTR", "HTT"]
        
        for gene in important_genes:
            conditions = DataTransformer.find_gene_related_conditions(gene)
            assert len(conditions) > 0, f"No conditions found for {gene}"

    def test_disease_synonyms_coverage(self):
        """Test that disease synonyms cover important categories"""
        important_diseases = ["cancer", "heart disease", "diabetes", "stroke"]
        
        for disease in important_diseases:
            terms = DataTransformer.expand_disease_search_terms(disease)
            assert len(terms) > 1, f"No synonyms found for {disease}"

    def test_keyword_extraction_medical_terms(self):
        """Test that medical terms are properly extracted"""
        medical_text = "Cardiovascular disease genomics research with GWAS analysis"
        keywords = DataTransformer.extract_keywords(medical_text)
        
        assert "cardiovascular" in keywords
        assert "disease" in keywords
        assert "genomics" in keywords
        assert "research" in keywords
        assert "gwas" in keywords
        assert "analysis" in keywords

    def test_date_formatting_edge_cases(self):
        """Test date formatting with various input formats"""
        datasets = [
            {"metadata_created": "2023-01-01T10:00:00Z"},
            {"metadata_created": "2023-01-01T10:00:00+00:00"},
            {"metadata_created": "invalid-date"},
            {"metadata_created": None}
        ]
        
        for dataset in datasets:
            enhanced = DataTransformer.enhance_dataset_info(dataset)
            # Should not raise errors - enhanced should always contain the field
            if dataset.get("metadata_created"):
                # Only check for formatted field if original exists
                if dataset["metadata_created"] not in ["invalid-date", None]:
                    assert "metadata_created_formatted" in enhanced

    def test_relevance_score_bounds(self):
        """Test that relevance scores are always within bounds"""
        datasets = [
            {"title": "genetic genomic hereditary inherited congenital mutation", "notes": "", "tags": []},
            {"title": "", "notes": "", "tags": []},
            {"title": "weather data", "notes": "temperature rainfall", "tags": ["climate"]}
        ]
        
        for dataset in datasets:
            enhanced = DataTransformer.enhance_dataset_info(dataset)
            score = enhanced["genetic_relevance"]
            assert 0.0 <= score <= 1.0