import asyncio
import random
from datetime import datetime
from typing import List, Dict, Callable, Optional
import logging
from ..models.analysis import AnalysisResult, AnalysisStatus, AnalysisProgress
from ..storage.memory_store import storage

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Business logic service for gene-disease analysis processing
    Handles mock analysis with realistic streaming simulation
    """
    
    def __init__(self):
        """Initialize analysis service with mock data configurations"""
        
        # Mock research findings templates for variety
        self._finding_templates = [
            "{gene} expression is significantly altered in {disease} patients ({percentage}% of cases)",
            "Mutations in {gene} are associated with increased {disease} risk (OR: {odds_ratio})",
            "{gene} protein dysfunction contributes to {disease} pathogenesis through {pathway} pathway",
            "Clinical trials show {gene} inhibitors reduce {disease} symptoms by {percentage}%",
            "{gene} variants are found in {percentage}% of familial {disease} cases",
            "Overexpression of {gene} correlates with {disease} severity and progression",
            "{gene} methylation patterns are disrupted in {disease} tissue samples",
            "Novel therapeutic targets identified in {gene}-{disease} interaction pathway"
        ]
        
        # Biological pathway options for realistic findings
        self._pathways = [
            "inflammatory response", "apoptosis regulation", "cell cycle control",
            "DNA repair mechanisms", "metabolic signaling", "protein folding",
            "immune system activation", "oxidative stress response"
        ]
        
        # Progress messages for streaming simulation
        self._progress_stages = [
            "Initializing analysis pipeline...",
            "Querying genetic databases...",
            "Retrieving gene sequence data...",
            "Analyzing protein structure...",
            "Cross-referencing disease associations...",
            "Examining genetic variants...",
            "Processing pathway interactions...",
            "Evaluating therapeutic targets...",
            "Generating comprehensive report...",
            "Finalizing analysis results..."
        ]
    
    async def start_analysis(self, session_id: str, gene: str, disease: str, 
                           progress_callback: Optional[Callable] = None) -> AnalysisResult:
        """
        Initiates mock gene-disease analysis with streaming progress updates
        
        Args:
            session_id: Session identifier for authentication
            gene: Gene name to analyze (e.g., "BRCA1")
            disease: Disease name to analyze (e.g., "breast cancer")
            progress_callback: Optional callback for streaming progress updates
            
        Returns:
            AnalysisResult: Complete analysis with findings and metadata
            
        Raises:
            ValueError: If session doesn't exist or parameters are invalid
        """
        
        # Validate session exists
        if not storage.session_exists(session_id):
            raise ValueError("Invalid session ID")
        
        logger.info(f"Starting analysis: {gene} vs {disease} for session {session_id}")
        
        # Create initial analysis record
        analysis = storage.create_analysis(session_id, gene, disease)
        analysis_id = analysis.analysis_id
        
        try:
            # Update status to processing
            storage.update_analysis(analysis_id, status=AnalysisStatus.PROCESSING)
            
            # Simulate analysis with progress updates
            await self._simulate_analysis_processing(
                analysis_id, gene, disease, progress_callback
            )
            
            # Generate final results
            final_analysis = await self._generate_analysis_results(
                analysis_id, gene, disease
            )
            
            logger.info(f"Analysis completed: {analysis_id}")
            return final_analysis
            
        except Exception as e:
            # Mark analysis as failed
            logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
            storage.update_analysis(
                analysis_id,
                status=AnalysisStatus.FAILED,
                summary=f"Analysis failed: {str(e)}"
            )
            raise
    
    async def _simulate_analysis_processing(self, analysis_id: str, gene: str, disease: str,
                                          progress_callback: Optional[Callable] = None):
        """
        Simulates realistic analysis processing with streaming updates
        
        Args:
            analysis_id: Analysis identifier
            gene: Gene being analyzed
            disease: Disease being analyzed
            progress_callback: Function to call with progress updates
        """
        
        total_stages = len(self._progress_stages)
        
        for i, stage_message in enumerate(self._progress_stages):
            # Simulate variable processing time for realism
            delay = random.uniform(0.8, 1.5)  # 0.8-1.5 seconds per stage
            await asyncio.sleep(delay)
            
            # Calculate progress percentage
            progress_percent = ((i + 1) / total_stages) * 100
            
            # Customize progress message with actual gene/disease names
            formatted_message = stage_message.replace("{gene}", gene).replace("{disease}", disease)
            
            # Update analysis progress in storage
            storage.update_analysis(
                analysis_id,
                summary=formatted_message
            )
            
            # Call progress callback if provided (for streaming endpoints)
            if progress_callback:
                progress_update = AnalysisProgress(
                    analysis_id=analysis_id,
                    status=AnalysisStatus.PROCESSING,
                    progress_message=formatted_message,
                    progress_percentage=progress_percent,
                    timestamp=datetime.utcnow()
                )
                await progress_callback(progress_update)
            
            logger.debug(f"Analysis {analysis_id}: {formatted_message} ({progress_percent:.1f}%)")
    
    async def _generate_analysis_results(self, analysis_id: str, gene: str, disease: str) -> AnalysisResult:
        """
        Generates comprehensive mock analysis results
        
        Args:
            analysis_id: Analysis identifier
            gene: Gene that was analyzed
            disease: Disease that was analyzed
            
        Returns:
            AnalysisResult: Complete analysis with realistic findings
        """
        
        start_time = datetime.utcnow()
        
        # Generate realistic confidence score (85-95% for successful analyses)
        confidence_score = round(random.uniform(85.0, 95.0), 1)
        
        # Generate key findings using templates
        key_findings = self._generate_key_findings(gene, disease)
        
        # Generate pathway analysis
        pathway = random.choice(self._pathways)
        pathway_analysis = f"{gene} primarily affects {disease} through the {pathway} pathway, " \
                          f"with significant downstream effects on cellular function and tissue homeostasis."
        
        # Generate therapeutic targets
        therapeutic_targets = self._generate_therapeutic_targets(gene)
        
        # Create executive summary
        summary = self._generate_executive_summary(gene, disease, confidence_score, len(key_findings))
        
        # Calculate processing time
        analysis_record = storage.get_analysis(analysis_id)
        if analysis_record:
            processing_time = (start_time - analysis_record.created_at).total_seconds()
        else:
            processing_time = 10.0  # Default fallback
        
        # Update analysis with complete results
        completed_analysis = storage.update_analysis(
            analysis_id,
            status=AnalysisStatus.COMPLETED,
            summary=summary,
            confidence_score=confidence_score,
            key_findings=key_findings,
            pathway_analysis=pathway_analysis,
            therapeutic_targets=therapeutic_targets,
            completed_at=start_time,
            processing_time_seconds=processing_time
        )
        
        return completed_analysis
    
    def _generate_key_findings(self, gene: str, disease: str) -> List[str]:
        """
        Generates realistic key findings using templates and random data
        
        Args:
            gene: Gene name
            disease: Disease name
            
        Returns:
            List[str]: 3-5 key research findings
        """
        
        findings = []
        
        # Select 3-5 random finding templates
        selected_templates = random.sample(self._finding_templates, random.randint(3, 5))
        
        for template in selected_templates:
            # Generate realistic percentages and statistics
            percentage = random.randint(23, 78)
            odds_ratio = round(random.uniform(1.2, 4.8), 1)
            pathway = random.choice(self._pathways)
            
            # Fill template with gene, disease, and generated data
            finding = template.format(
                gene=gene,
                disease=disease,
                percentage=percentage,
                odds_ratio=odds_ratio,
                pathway=pathway
            )
            
            findings.append(finding)
        
        return findings
    
    def _generate_therapeutic_targets(self, gene: str) -> List[str]:
        """
        Generates potential therapeutic targets based on the gene
        
        Args:
            gene: Gene name
            
        Returns:
            List[str]: 2-4 therapeutic target suggestions
        """
        
        target_templates = [
            f"{gene} protein inhibitors",
            f"{gene} gene therapy approaches",
            f"Small molecules targeting {gene} pathway",
            f"{gene} expression modulators",
            f"Monoclonal antibodies against {gene}",
            f"{gene}-specific RNA interference"
        ]
        
        # Return 2-4 random therapeutic approaches
        return random.sample(target_templates, random.randint(2, 4))
    
    def _generate_executive_summary(self, gene: str, disease: str, confidence: float, 
                                  finding_count: int) -> str:
        """
        Creates an executive summary of the analysis
        
        Args:
            gene: Gene name
            disease: Disease name
            confidence: Confidence score
            finding_count: Number of key findings
            
        Returns:
            str: Executive summary text
        """
        
        return f"""Comprehensive analysis of {gene} gene in relation to {disease} reveals significant associations 
and potential therapeutic opportunities. Our analysis identified {finding_count} key findings with a confidence 
score of {confidence}%. The results indicate substantial involvement of {gene} in {disease} pathogenesis, 
supported by genetic, molecular, and clinical evidence. These findings suggest promising avenues for 
therapeutic intervention and warrant further clinical investigation."""


# Global service instance
analysis_service = AnalysisService()