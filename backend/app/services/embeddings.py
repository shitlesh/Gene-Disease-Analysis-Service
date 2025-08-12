"""
Embeddings Service
Generates, stores, and searches embeddings for analysis similarity
Supports finding similar analyses and content-based recommendations
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.session import get_async_session
from ..db.models import Analysis, LLMProviderType
from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Base class for embedding providers"""
    
    async def generate_embedding(self, text: str) -> List[float]:
        raise NotImplementedError
    
    def get_embedding_dimension(self) -> int:
        raise NotImplementedError
    
    def get_model_name(self) -> str:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.dimension = 1536  # text-embedding-3-small dimension
        self.max_tokens = 8191
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            # Truncate text if too long
            truncated_text = self._truncate_text(text, self.max_tokens)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": truncated_text,
                        "model": self.model
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
                logger.debug(f"Generated OpenAI embedding with dimension {len(embedding)}")
                return embedding
                
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit (approximate)"""
        # Rough approximation: 4 characters per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Anthropic embedding provider (placeholder - would use actual API when available)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.dimension = 1024  # Placeholder dimension
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Anthropic API (placeholder)"""
        # This is a placeholder - Anthropic doesn't currently offer embedding APIs
        # In practice, you might use a local model or another embedding service
        logger.warning("Anthropic embeddings not implemented, using mock embedding")
        
        # Return a mock embedding for testing
        import random
        random.seed(hash(text) % 2**32)
        return [random.uniform(-1, 1) for _ in range(self.dimension)]
    
    def get_embedding_dimension(self) -> int:
        return self.dimension
    
    def get_model_name(self) -> str:
        return "anthropic/mock-embedding"


class EmbeddingsService:
    """Service for generating and managing embeddings"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
        self.similarity_threshold = 0.8  # Cosine similarity threshold
        self.max_similar_analyses = 10
    
    def _initialize_providers(self):
        """Initialize embedding providers based on configuration"""
        try:
            # Initialize OpenAI provider if API key is available
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                self.providers[LLMProviderType.OPENAI] = OpenAIEmbeddingProvider(
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info("OpenAI embedding provider initialized")
            
            # Initialize Anthropic provider if API key is available
            if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
                self.providers[LLMProviderType.ANTHROPIC] = AnthropicEmbeddingProvider(
                    api_key=settings.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic embedding provider initialized")
            
            if not self.providers:
                logger.warning("No embedding providers initialized - similarity search will be unavailable")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding providers: {e}")
    
    async def generate_analysis_embedding(
        self,
        analysis: Analysis,
        provider_override: Optional[LLMProviderType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate embedding for an analysis
        
        Args:
            analysis: Analysis object
            provider_override: Override the embedding provider
            
        Returns:
            Dict containing embedding data or None if failed
        """
        try:
            # Determine which provider to use
            embedding_provider_type = provider_override or analysis.provider
            
            if embedding_provider_type not in self.providers:
                logger.warning(f"No embedding provider available for {embedding_provider_type}")
                return None
            
            provider = self.providers[embedding_provider_type]
            
            # Create text for embedding
            embedding_text = self._create_embedding_text(analysis)
            
            # Generate embedding
            embedding_vector = await provider.generate_embedding(embedding_text)
            
            # Create embedding data structure
            embedding_data = {
                "vector": embedding_vector,
                "dimension": len(embedding_vector),
                "model": provider.get_model_name(),
                "created_at": datetime.utcnow().isoformat(),
                "text_length": len(embedding_text),
                "metadata": {
                    "gene": analysis.gene,
                    "disease": analysis.disease,
                    "provider": analysis.provider.value,
                    "analysis_model": analysis.model,
                    "status": analysis.status.value
                }
            }
            
            logger.info(f"Generated embedding for analysis {analysis.id} using {provider.get_model_name()}")
            return embedding_data
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for analysis {analysis.id}: {e}")
            return None
    
    def _create_embedding_text(self, analysis: Analysis) -> str:
        """Create text representation of analysis for embedding"""
        text_parts = []
        
        # Add gene and disease information
        text_parts.append(f"Gene: {analysis.gene}")
        text_parts.append(f"Disease: {analysis.disease}")
        
        # Add context if available
        if analysis.context:
            text_parts.append(f"Context: {analysis.context}")
        
        # Add result summary if available
        if analysis.result_summary:
            text_parts.append(f"Results: {analysis.result_summary}")
        
        # Add a portion of the prompt for more context
        if analysis.prompt_text:
            # Take first 500 characters of prompt
            prompt_excerpt = analysis.prompt_text[:500]
            text_parts.append(f"Prompt: {prompt_excerpt}")
        
        return " | ".join(text_parts)
    
    async def store_analysis_embedding(self, analysis_id: int, embedding_data: Dict[str, Any]) -> bool:
        """Store embedding data in the analysis record"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, update
                
                # Update the analysis with embedding data
                query = (
                    update(Analysis)
                    .where(Analysis.id == analysis_id)
                    .values(embeddings=embedding_data, updated_at=datetime.utcnow())
                )
                
                result = await session.execute(query)
                await session.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Stored embedding for analysis {analysis_id}")
                else:
                    logger.warning(f"No analysis found with ID {analysis_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to store embedding for analysis {analysis_id}: {e}")
            return False
    
    async def find_similar_analyses(
        self,
        analysis_id: int,
        session_id: Optional[int] = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Find analyses similar to the given analysis
        
        Args:
            analysis_id: ID of the reference analysis
            session_id: Optional session ID to filter results
            limit: Maximum number of similar analyses to return
            
        Returns:
            List of similar analyses with similarity scores
        """
        try:
            limit = limit or self.max_similar_analyses
            
            async with get_async_session() as session:
                from sqlmodel import select
                
                # Get the reference analysis and its embedding
                ref_query = select(Analysis).where(Analysis.id == analysis_id)
                ref_result = await session.execute(ref_query)
                ref_analysis = ref_result.scalar_one_or_none()
                
                if not ref_analysis or not ref_analysis.embeddings:
                    logger.warning(f"Analysis {analysis_id} not found or has no embedding")
                    return []
                
                ref_embedding = np.array(ref_analysis.embeddings["vector"])
                
                # Get all analyses with embeddings for comparison
                query = select(Analysis).where(
                    Analysis.embeddings.is_not(None),
                    Analysis.id != analysis_id,  # Exclude the reference analysis
                    Analysis.status == "completed"  # Only completed analyses
                )
                
                # Filter by session if specified
                if session_id:
                    query = query.where(Analysis.session_id == session_id)
                
                result = await session.execute(query)
                candidate_analyses = result.scalars().all()
                
                # Calculate similarities
                similarities = []
                for candidate in candidate_analyses:
                    try:
                        candidate_embedding = np.array(candidate.embeddings["vector"])
                        similarity = self._calculate_cosine_similarity(ref_embedding, candidate_embedding)
                        
                        if similarity >= self.similarity_threshold:
                            similarities.append({
                                "analysis_id": candidate.id,
                                "similarity_score": float(similarity),
                                "gene": candidate.gene,
                                "disease": candidate.disease,
                                "created_at": candidate.created_at.isoformat(),
                                "provider": candidate.provider.value,
                                "model": candidate.model,
                                "result_summary": candidate.result_summary[:200] if candidate.result_summary else None
                            })
                            
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity for analysis {candidate.id}: {e}")
                        continue
                
                # Sort by similarity score and limit results
                similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"Failed to find similar analyses for {analysis_id}: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def search_similar_content(
        self,
        query_text: str,
        provider: LLMProviderType,
        session_id: Optional[int] = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for analyses similar to a text query
        
        Args:
            query_text: Text to search for
            provider: Embedding provider to use
            session_id: Optional session ID to filter results
            limit: Maximum number of results
            
        Returns:
            List of similar analyses
        """
        try:
            limit = limit or self.max_similar_analyses
            
            if provider not in self.providers:
                logger.warning(f"No embedding provider available for {provider}")
                return []
            
            # Generate embedding for query text
            embedding_provider = self.providers[provider]
            query_embedding = np.array(await embedding_provider.generate_embedding(query_text))
            
            async with get_async_session() as session:
                from sqlmodel import select
                
                # Get analyses with embeddings
                query = select(Analysis).where(
                    Analysis.embeddings.is_not(None),
                    Analysis.status == "completed"
                )
                
                if session_id:
                    query = query.where(Analysis.session_id == session_id)
                
                result = await session.execute(query)
                analyses = result.scalars().all()
                
                # Calculate similarities
                similarities = []
                for analysis in analyses:
                    try:
                        analysis_embedding = np.array(analysis.embeddings["vector"])
                        similarity = self._calculate_cosine_similarity(query_embedding, analysis_embedding)
                        
                        if similarity >= self.similarity_threshold:
                            similarities.append({
                                "analysis_id": analysis.id,
                                "similarity_score": float(similarity),
                                "gene": analysis.gene,
                                "disease": analysis.disease,
                                "created_at": analysis.created_at.isoformat(),
                                "provider": analysis.provider.value,
                                "model": analysis.model,
                                "result_summary": analysis.result_summary[:200] if analysis.result_summary else None
                            })
                            
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity for analysis {analysis.id}: {e}")
                        continue
                
                # Sort by similarity and limit results
                similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"Failed to search similar content: {e}")
            return []
    
    async def batch_generate_embeddings(
        self,
        session_id: Optional[int] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings for analyses that don't have them
        
        Args:
            session_id: Optional session ID to filter analyses
            force_regenerate: Whether to regenerate existing embeddings
            
        Returns:
            Dict with processing results
        """
        try:
            async with get_async_session() as session:
                from sqlmodel import select
                
                # Query for analyses that need embeddings
                query = select(Analysis).where(Analysis.status == "completed")
                
                if not force_regenerate:
                    query = query.where(Analysis.embeddings.is_(None))
                
                if session_id:
                    query = query.where(Analysis.session_id == session_id)
                
                result = await session.execute(query)
                analyses = result.scalars().all()
                
                processed = 0
                successful = 0
                failed = 0
                
                for analysis in analyses:
                    try:
                        # Generate embedding
                        embedding_data = await self.generate_analysis_embedding(analysis)
                        
                        if embedding_data:
                            # Store embedding
                            stored = await self.store_analysis_embedding(analysis.id, embedding_data)
                            if stored:
                                successful += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                        
                        processed += 1
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Failed to process embedding for analysis {analysis.id}: {e}")
                        failed += 1
                        processed += 1
                
                result_summary = {
                    "total_analyses": len(analyses),
                    "processed": processed,
                    "successful": successful,
                    "failed": failed,
                    "session_id": session_id,
                    "force_regenerate": force_regenerate
                }
                
                logger.info(f"Batch embedding generation completed: {result_summary}")
                return result_summary
                
        except Exception as e:
            logger.error(f"Failed to batch generate embeddings: {e}")
            return {"error": str(e)}
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the system"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, func
                
                # Count analyses with and without embeddings
                total_query = select(func.count(Analysis.id)).where(Analysis.status == "completed")
                total_result = await session.execute(total_query)
                total_completed = total_result.scalar() or 0
                
                embedded_query = select(func.count(Analysis.id)).where(
                    Analysis.status == "completed",
                    Analysis.embeddings.is_not(None)
                )
                embedded_result = await session.execute(embedded_query)
                total_embedded = embedded_result.scalar() or 0
                
                # Get embedding model distribution
                model_query = select(
                    func.json_extract(Analysis.embeddings, "$.model").label("embedding_model"),
                    func.count(Analysis.id).label("count")
                ).where(
                    Analysis.embeddings.is_not(None)
                ).group_by("embedding_model")
                
                model_result = await session.execute(model_query)
                model_distribution = {}
                
                for row in model_result.fetchall():
                    model = row[0]
                    count = row[1] or 0
                    if model:  # Only include non-null models
                        model_distribution[model] = count
                
                return {
                    "total_completed_analyses": total_completed,
                    "total_embedded_analyses": total_embedded,
                    "embedding_coverage": total_embedded / total_completed if total_completed > 0 else 0,
                    "available_providers": list(self.providers.keys()),
                    "embedding_model_distribution": model_distribution,
                    "similarity_threshold": self.similarity_threshold,
                    "max_similar_analyses": self.max_similar_analyses
                }
                
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {"error": str(e)}


# Global service instance
embeddings_service = EmbeddingsService()