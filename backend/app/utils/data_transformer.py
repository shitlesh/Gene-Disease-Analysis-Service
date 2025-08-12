"""
Data transformation utilities for NHS Scotland health data

This module provides functions to transform, filter, and enhance raw NHS Scotland
data to make it more suitable for gene-disease analysis and frontend consumption.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Utility class for transforming NHS Scotland health data
    Provides methods to clean, filter, and enhance raw API responses
    """
    
    # Known gene-disease associations for enhanced search results
    GENE_CONDITION_MAPPING = {
        "BRCA1": ["breast cancer", "ovarian cancer", "hereditary cancer"],
        "BRCA2": ["breast cancer", "ovarian cancer", "prostate cancer"],
        "TP53": ["li-fraumeni syndrome", "cancer", "tumor suppressor"],
        "CFTR": ["cystic fibrosis", "respiratory disease"],
        "HTT": ["huntington's disease", "neurological disorders"],
        "DMD": ["duchenne muscular dystrophy", "muscle disease"],
        "F8": ["hemophilia a", "bleeding disorders"],
        "F9": ["hemophilia b", "clotting disorders"],
        "HEXA": ["tay-sachs disease", "metabolic disorders"],
        "HBB": ["sickle cell disease", "thalassemia", "blood disorders"]
    }
    
    # Disease synonyms and alternative names for better search matching
    DISEASE_SYNONYMS = {
        "cancer": ["carcinoma", "malignancy", "tumor", "neoplasm"],
        "heart disease": ["cardiovascular", "cardiac", "coronary"],
        "diabetes": ["diabetic", "glucose", "insulin"],
        "stroke": ["cerebrovascular", "brain attack"],
        "mental health": ["psychiatric", "psychological", "depression", "anxiety"],
        "respiratory": ["lung", "breathing", "pulmonary"],
        "genetic": ["hereditary", "congenital", "inherited"],
        "neurological": ["nervous system", "brain", "neurologic"]
    }
    
    @staticmethod
    def clean_dataset_title(title: str) -> str:
        """
        Clean and standardize dataset titles
        
        Args:
            title: Raw dataset title from API
            
        Returns:
            Cleaned and formatted title
        """
        if not title:
            return "Unknown Dataset"
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', title.strip())
        
        # Capitalize first letter of each word for consistency
        return cleaned.title()
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract relevant keywords from dataset descriptions
        
        Args:
            text: Description or notes text
            
        Returns:
            List of relevant keywords
        """
        if not text:
            return []
        
        # Convert to lowercase and remove special characters
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter
        words = clean_text.split()
        
        # Filter out common stop words and keep medical/scientific terms
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that',
            'these', 'those', 'a', 'an', 'as', 'if', 'each', 'which', 'who', 'when',
            'where', 'how', 'what', 'why', 'data', 'dataset', 'information'
        }
        
        # Keep words that are likely medical or scientific terms
        keywords = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and
                not word.isdigit() and
                any(char.isalpha() for char in word)):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Return top 10 keywords
    
    @classmethod
    def enhance_dataset_info(cls, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance dataset information with additional metadata
        
        Args:
            dataset: Raw dataset dictionary from API
            
        Returns:
            Enhanced dataset with additional fields
        """
        enhanced = dataset.copy()
        
        # Clean the title
        if 'title' in enhanced:
            enhanced['title'] = cls.clean_dataset_title(enhanced['title'])
        
        # Extract keywords from description
        description = enhanced.get('notes', '')
        enhanced['keywords'] = cls.extract_keywords(description)
        
        # Add relevance indicators
        enhanced['genetic_relevance'] = cls.assess_genetic_relevance(enhanced)
        enhanced['disease_categories'] = cls.identify_disease_categories(enhanced)
        
        # Format dates for better readability
        for date_field in ['metadata_created', 'metadata_modified']:
            if date_field in enhanced and enhanced[date_field]:
                try:
                    # Parse and reformat date
                    date_obj = datetime.fromisoformat(enhanced[date_field].replace('Z', '+00:00'))
                    enhanced[f"{date_field}_formatted"] = date_obj.strftime('%Y-%m-%d')
                except Exception:
                    enhanced[f"{date_field}_formatted"] = enhanced[date_field]
        
        return enhanced
    
    @classmethod
    def assess_genetic_relevance(cls, dataset: Dict[str, Any]) -> float:
        """
        Assess how relevant a dataset is for genetic analysis
        
        Args:
            dataset: Dataset information dictionary
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        score = 0.0
        
        # Check title and description for genetic keywords
        text_fields = [
            dataset.get('title', ''),
            dataset.get('notes', ''),
            ' '.join(dataset.get('tags', []))
        ]
        
        genetic_keywords = [
            'genetic', 'gene', 'genomic', 'hereditary', 'inherited', 'congenital',
            'mutation', 'chromosome', 'dna', 'syndrome', 'disorder', 'condition',
            'screening', 'testing', 'family history', 'birth defect'
        ]
        
        full_text = ' '.join(text_fields).lower()
        
        for keyword in genetic_keywords:
            if keyword in full_text:
                if keyword in ['genetic', 'gene', 'genomic', 'hereditary']:
                    score += 0.2  # High value keywords
                elif keyword in ['congenital', 'inherited', 'syndrome']:
                    score += 0.15  # Medium value keywords
                else:
                    score += 0.1  # Lower value keywords
        
        # Cap at 1.0
        return min(score, 1.0)
    
    @classmethod
    def identify_disease_categories(cls, dataset: Dict[str, Any]) -> List[str]:
        """
        Identify disease categories present in a dataset
        
        Args:
            dataset: Dataset information dictionary
            
        Returns:
            List of identified disease categories
        """
        categories = []
        
        text_content = ' '.join([
            dataset.get('title', ''),
            dataset.get('notes', ''),
            ' '.join(dataset.get('tags', []))
        ]).lower()
        
        category_keywords = {
            'cancer': ['cancer', 'carcinoma', 'tumor', 'malignancy', 'oncology'],
            'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'stroke'],
            'neurological': ['brain', 'neurological', 'nervous', 'mental', 'cognitive'],
            'respiratory': ['lung', 'respiratory', 'breathing', 'pulmonary', 'asthma'],
            'genetic_disorders': ['genetic', 'hereditary', 'congenital', 'syndrome', 'mutation'],
            'metabolic': ['diabetes', 'metabolic', 'endocrine', 'hormone'],
            'infectious': ['infection', 'infectious', 'viral', 'bacterial', 'vaccine'],
            'maternal_child': ['pregnancy', 'birth', 'maternal', 'child', 'pediatric']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                categories.append(category.replace('_', ' ').title())
        
        return categories
    
    @classmethod
    def find_gene_related_conditions(cls, gene_name: str) -> List[str]:
        """
        Find known conditions associated with a specific gene
        
        Args:
            gene_name: Gene name to look up
            
        Returns:
            List of associated conditions
        """
        gene_upper = gene_name.upper().strip()
        
        # Direct lookup
        if gene_upper in cls.GENE_CONDITION_MAPPING:
            return cls.GENE_CONDITION_MAPPING[gene_upper].copy()
        
        # Fuzzy matching for common variations
        fuzzy_matches = []
        for gene, conditions in cls.GENE_CONDITION_MAPPING.items():
            if gene_upper in gene or gene in gene_upper:
                fuzzy_matches.extend(conditions)
        
        return fuzzy_matches
    
    @classmethod
    def expand_disease_search_terms(cls, disease_name: str) -> List[str]:
        """
        Expand a disease name with synonyms and related terms
        
        Args:
            disease_name: Original disease name
            
        Returns:
            List of search terms including synonyms
        """
        terms = [disease_name.lower().strip()]
        
        # Add synonyms
        for main_term, synonyms in cls.DISEASE_SYNONYMS.items():
            if main_term in disease_name.lower():
                terms.extend(synonyms)
            elif any(synonym in disease_name.lower() for synonym in synonyms):
                terms.append(main_term)
                terms.extend(synonyms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    @staticmethod
    def filter_datasets_by_relevance(datasets: List[Dict[str, Any]], 
                                   min_relevance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Filter datasets based on genetic relevance score
        
        Args:
            datasets: List of dataset dictionaries
            min_relevance: Minimum relevance score to include
            
        Returns:
            Filtered list of relevant datasets
        """
        relevant_datasets = []
        
        for dataset in datasets:
            relevance = dataset.get('genetic_relevance', 0.0)
            if relevance >= min_relevance:
                relevant_datasets.append(dataset)
        
        # Sort by relevance score (highest first)
        relevant_datasets.sort(key=lambda x: x.get('genetic_relevance', 0), reverse=True)
        
        return relevant_datasets
    
    @staticmethod
    def format_for_frontend(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format dataset information for frontend consumption
        
        Args:
            datasets: Raw dataset information
            
        Returns:
            Formatted datasets optimized for React components and Pydantic models
        """
        from app.models.nhs_scotland import DatasetSearchResult
        
        formatted = []
        
        for dataset in datasets:
            # Create DatasetSearchResult objects that match the expected Pydantic model
            formatted_dataset = DatasetSearchResult(
                name=dataset.get('name', ''),
                title=dataset.get('title', 'Unknown Dataset'),
                notes=dataset.get('notes', '')[:300] + ('...' if len(dataset.get('notes', '')) > 300 else ''),
                score=round(dataset.get('genetic_relevance', 0.0), 2),
                tags=dataset.get('tags', [])[:5] if dataset.get('tags') else [],
                organization=dataset.get('organization', 'NHS Scotland'),
                resources=dataset.get('resources', 0) if isinstance(dataset.get('resources'), int) else 0,
                metadata_modified=dataset.get('metadata_modified_formatted', None)
            )
            
            formatted.append(formatted_dataset)
        
        return formatted
    
    @classmethod
    def create_analysis_suggestions(cls, gene_name: str = None, 
                                  disease_name: str = None) -> Dict[str, Any]:
        """
        Create analysis suggestions based on gene or disease input
        
        Args:
            gene_name: Optional gene name
            disease_name: Optional disease name
            
        Returns:
            Dictionary with analysis suggestions and related information
        """
        suggestions = {
            'search_strategies': [],
            'related_datasets': [],
            'analysis_approaches': [],
            'data_limitations': []
        }
        
        if gene_name:
            # Gene-based suggestions
            conditions = cls.find_gene_related_conditions(gene_name)
            if conditions:
                suggestions['search_strategies'].append(
                    f"Search for datasets related to conditions associated with {gene_name}: {', '.join(conditions[:3])}"
                )
                suggestions['related_datasets'] = conditions
            else:
                suggestions['search_strategies'].append(
                    f"Search for general genetic/genomic datasets that might contain information about {gene_name}"
                )
            
            suggestions['analysis_approaches'].extend([
                "Look for cancer incidence data if this is an oncogene",
                "Check congenital conditions data for developmental genes",
                "Search screening program data for known genetic conditions"
            ])
        
        if disease_name:
            # Disease-based suggestions
            expanded_terms = cls.expand_disease_search_terms(disease_name)
            suggestions['search_strategies'].append(
                f"Search using expanded terms: {', '.join(expanded_terms[:5])}"
            )
            
            suggestions['analysis_approaches'].extend([
                "Look for incidence and prevalence data",
                "Check mortality statistics",
                "Find screening or diagnostic data",
                "Search for geographic distribution patterns"
            ])
        
        # General limitations
        suggestions['data_limitations'] = [
            "NHS Scotland open data may not contain direct genetic sequence information",
            "Individual patient data is not available due to privacy regulations",
            "Some datasets may have geographic limitations (Scotland-specific)",
            "Data completeness varies between different time periods"
        ]
        
        return suggestions


# Global transformer instance
data_transformer = DataTransformer()