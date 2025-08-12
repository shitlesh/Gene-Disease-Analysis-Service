"""
Prompt templates for LLM gene-disease correlation analysis
Designed to minimize hallucination and ensure structured JSON output
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    @abstractmethod
    def format_system_message(self, **kwargs) -> str:
        """Format the system message with provided parameters"""
        pass
    
    @abstractmethod
    def format_user_message(self, **kwargs) -> str:
        """Format the user message with provided parameters"""
        pass


class GeneDiseaseCorrAnalysisTemplate(PromptTemplate):
    """Template for gene-disease correlation analysis"""
    
    SYSTEM_MESSAGE = """You are a precise medical genetics expert. Your task is to analyze gene-disease correlations using established scientific knowledge.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON in the exact format specified below
2. Base your analysis ONLY on well-established scientific evidence
3. If evidence is limited, clearly state this in the uncertainty field
4. Do NOT hallucinate or speculate beyond established research
5. Use "unknown" or "insufficient evidence" when appropriate
6. Be conservative in your confidence assessments

REQUIRED JSON FORMAT:
{
    "summary": "Brief summary of gene-disease relationship (max 200 chars)",
    "associations": [
        {
            "gene_function": "Primary biological function of the gene",
            "mechanism": "How the gene relates to the disease mechanism",
            "evidence_level": "strong|moderate|weak|insufficient",
            "phenotypes": ["list", "of", "associated", "clinical", "phenotypes"],
            "inheritance_pattern": "autosomal dominant|autosomal recessive|X-linked|mitochondrial|complex|unknown"
        }
    ],
    "recommendation": "Clinical or research recommendation based on evidence",
    "uncertainty": "Assessment of limitations, unknowns, and confidence bounds",
    "confidence_score": 0.0
}

EVIDENCE LEVELS:
- strong: Multiple independent studies, functional validation, clinical guidelines
- moderate: Some studies with consistent findings, plausible mechanism
- weak: Limited studies, preliminary evidence, case reports
- insufficient: No clear evidence or conflicting data

Output ONLY the JSON object. No additional text or explanation."""

    USER_MESSAGE = """Analyze the correlation between gene {gene} and disease {disease}.

Gene: {gene}
Disease: {disease}
{context_section}

Provide your analysis in the exact JSON format specified in the system message. Base your response strictly on established scientific evidence."""

    USER_MESSAGE_WITH_REFERENCES = """Analyze the correlation between gene {gene} and disease {disease}.

Gene: {gene}
Disease: {disease}
{context_section}

Include literature references in your response using this extended JSON format:
{
    "summary": "...",
    "associations": [...],
    "recommendation": "...",
    "uncertainty": "...",
    "confidence_score": 0.0,
    "references": ["Author et al. Journal (Year)", "Another reference"]
}

Provide your analysis in the exact JSON format specified. Base your response strictly on established scientific evidence."""

    def format_system_message(self, **kwargs) -> str:
        """Format the system message"""
        return self.SYSTEM_MESSAGE

    def format_user_message(
        self, 
        gene: str, 
        disease: str, 
        context: Optional[str] = None,
        include_references: bool = False,
        **kwargs
    ) -> str:
        """Format the user message with gene, disease, and optional context"""
        context_section = ""
        if context:
            context_section = f"Additional Context: {context}\n"
        
        template = self.USER_MESSAGE_WITH_REFERENCES if include_references else self.USER_MESSAGE
        
        return template.format(
            gene=gene,
            disease=disease,
            context_section=context_section
        )


class GeneFunctionAnalysisTemplate(PromptTemplate):
    """Template for focused gene function analysis"""
    
    SYSTEM_MESSAGE = """You are a molecular biology expert specializing in gene function analysis.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON in the exact format specified
2. Focus on molecular mechanisms and pathways
3. Be precise about gene function and protein products
4. Use established nomenclature and classifications
5. Indicate uncertainty when evidence is limited

REQUIRED JSON FORMAT:
{
    "summary": "Brief description of gene function",
    "associations": [
        {
            "gene_function": "Primary molecular function and pathways",
            "mechanism": "Molecular mechanism of action",
            "evidence_level": "strong|moderate|weak|insufficient",
            "phenotypes": ["molecular", "cellular", "phenotypes"],
            "inheritance_pattern": "pattern or unknown"
        }
    ],
    "recommendation": "Research or functional study recommendations",
    "uncertainty": "Limitations in current understanding",
    "confidence_score": 0.0
}

Output ONLY the JSON object."""

    USER_MESSAGE = """Analyze the function of gene {gene} in relation to {disease}.

Focus on:
- Molecular function and protein product
- Cellular pathways and interactions
- Known disease mechanisms
- Functional consequences of variants

{context_section}

Provide analysis in the specified JSON format."""

    def format_system_message(self, **kwargs) -> str:
        return self.SYSTEM_MESSAGE

    def format_user_message(
        self, 
        gene: str, 
        disease: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        context_section = ""
        if context:
            context_section = f"Additional Context: {context}\n"
            
        return self.USER_MESSAGE.format(
            gene=gene,
            disease=disease,
            context_section=context_section
        )


class DiseaseGeneticsTemplate(PromptTemplate):
    """Template for disease genetics analysis"""
    
    SYSTEM_MESSAGE = """You are a clinical genetics specialist analyzing disease genetics.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON in the exact format specified
2. Focus on clinical genetics and inheritance patterns
3. Include population frequencies when known
4. Be specific about penetrance and expressivity
5. Distinguish between causative and risk genes

REQUIRED JSON FORMAT:
{
    "summary": "Clinical genetics summary of the disease",
    "associations": [
        {
            "gene_function": "Role in disease pathogenesis",
            "mechanism": "Pathogenic mechanism",
            "evidence_level": "strong|moderate|weak|insufficient",
            "phenotypes": ["clinical", "features", "and", "symptoms"],
            "inheritance_pattern": "inheritance pattern with penetrance info"
        }
    ],
    "recommendation": "Clinical genetics recommendations",
    "uncertainty": "Clinical uncertainties and variable expressivity",
    "confidence_score": 0.0
}

Output ONLY the JSON object."""

    USER_MESSAGE = """Analyze the genetics of {disease} with focus on gene {gene}.

Consider:
- Inheritance patterns and penetrance
- Clinical phenotypes and variability
- Population genetics and frequencies
- Genotype-phenotype correlations

{context_section}

Provide analysis in the specified JSON format."""

    def format_system_message(self, **kwargs) -> str:
        return self.SYSTEM_MESSAGE

    def format_user_message(
        self, 
        gene: str, 
        disease: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        context_section = ""
        if context:
            context_section = f"Additional Context: {context}\n"
            
        return self.USER_MESSAGE.format(
            gene=gene,
            disease=disease,
            context_section=context_section
        )


class PromptTemplateRegistry:
    """Registry for managing different prompt templates"""
    
    def __init__(self):
        self._templates = {
            'correlation': GeneDiseaseCorrAnalysisTemplate(),
            'gene_function': GeneFunctionAnalysisTemplate(),
            'disease_genetics': DiseaseGeneticsTemplate()
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a template by name"""
        if template_name not in self._templates:
            raise ValueError(f"Unknown template: {template_name}")
        return self._templates[template_name]
    
    def register_template(self, name: str, template: PromptTemplate):
        """Register a new template"""
        self._templates[name] = template
    
    def list_templates(self) -> list:
        """List available templates"""
        return list(self._templates.keys())


# Global registry instance
template_registry = PromptTemplateRegistry()


def get_analysis_prompt(
    gene: str,
    disease: str,
    template_name: str = 'correlation',
    context: Optional[str] = None,
    include_references: bool = False,
    **kwargs
) -> tuple[str, str]:
    """
    Get formatted system and user messages for analysis
    
    Args:
        gene: Gene symbol
        disease: Disease name
        template_name: Template to use ('correlation', 'gene_function', 'disease_genetics')
        context: Optional additional context
        include_references: Whether to request literature references
        **kwargs: Additional template parameters
    
    Returns:
        Tuple of (system_message, user_message)
    """
    template = template_registry.get_template(template_name)
    
    system_message = template.format_system_message(**kwargs)
    user_message = template.format_user_message(
        gene=gene,
        disease=disease,
        context=context,
        include_references=include_references,
        **kwargs
    )
    
    return system_message, user_message


def validate_json_response_format(response: str) -> bool:
    """
    Validate that response contains the required JSON fields
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        True if response has required structure
    """
    import json
    
    try:
        data = json.loads(response.strip())
        
        required_fields = ['summary', 'associations', 'recommendation', 'uncertainty']
        
        # Check top-level fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Check associations structure
        if not isinstance(data['associations'], list):
            return False
            
        for assoc in data['associations']:
            if not isinstance(assoc, dict):
                return False
            
            required_assoc_fields = ['gene_function', 'mechanism', 'evidence_level', 'phenotypes']
            for field in required_assoc_fields:
                if field not in assoc:
                    return False
        
        return True
        
    except (json.JSONDecodeError, TypeError, KeyError):
        return False