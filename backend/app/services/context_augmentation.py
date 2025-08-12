"""
Context Augmentation Service
Automatically fetches NHS Scotland public data and other external sources
to enhance LLM prompts with factual information and reduce hallucinations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.session import get_async_session
from ..db.models import ContextAugmentation, LLMProviderType
from ..services.nhs_scotland import NHSScotlandService

logger = logging.getLogger(__name__)


class ContextAugmentationService:
    """Service for augmenting analysis prompts with external data"""
    
    def __init__(self):
        self.nhs_service = NHSScotlandService()
        self.cache_duration_hours = 24  # Cache external data for 24 hours
        self.max_context_length = 2000  # Maximum characters for context
        
    async def augment_analysis_context(
        self,
        gene: str,
        disease: str,
        existing_context: Optional[str] = None,
        provider: Optional[LLMProviderType] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Augment analysis context with external data
        
        Args:
            gene: Gene symbol
            disease: Disease name
            existing_context: Any existing context
            provider: LLM provider (for provider-specific optimizations)
            
        Returns:
            Tuple of (augmented_context, metadata)
        """
        try:
            # Get cached or fresh external data
            external_data = await self._get_external_data(gene, disease)
            
            # Build augmented context
            context_parts = []
            
            # Add existing context if provided
            if existing_context:
                context_parts.append(f"User Context: {existing_context}")
            
            # Add NHS Scotland data
            if external_data.get("nhs_scotland"):
                nhs_context = await self._format_nhs_context(external_data["nhs_scotland"])
                if nhs_context:
                    context_parts.append(f"NHS Scotland Data: {nhs_context}")
            
            # Add other data sources
            if external_data.get("additional"):
                for source, data in external_data["additional"].items():
                    formatted = await self._format_additional_context(source, data)
                    if formatted:
                        context_parts.append(f"{source.title()} Data: {formatted}")
            
            # Combine and truncate if needed
            full_context = "\n\n".join(context_parts)
            truncated_context = self._truncate_context(full_context, self.max_context_length)
            
            # Create metadata
            metadata = {
                "augmentation_applied": True,
                "data_sources": list(external_data.keys()),
                "context_length": len(truncated_context),
                "original_context_length": len(existing_context) if existing_context else 0,
                "truncated": len(full_context) > self.max_context_length,
                "augmentation_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Context augmented for {gene}-{disease} with {len(external_data)} sources")
            return truncated_context, metadata
            
        except Exception as e:
            logger.error(f"Context augmentation failed for {gene}-{disease}: {e}")
            # Return original context on failure
            fallback_metadata = {
                "augmentation_applied": False,
                "error": str(e),
                "fallback_used": True
            }
            return existing_context or "", fallback_metadata
    
    async def _get_external_data(self, gene: str, disease: str) -> Dict[str, Any]:
        """Get external data from various sources with caching"""
        external_data = {}
        
        # Try to get cached data first
        async with get_async_session() as session:
            cached_data = await self._get_cached_data(session, gene, disease)
            
            if cached_data:
                external_data.update(cached_data)
                logger.debug(f"Using cached data for {gene}-{disease}")
            else:
                # Fetch fresh data
                fresh_data = await self._fetch_fresh_data(gene, disease)
                external_data.update(fresh_data)
                
                # Cache the fresh data
                await self._cache_external_data(session, gene, disease, fresh_data)
                logger.debug(f"Cached fresh data for {gene}-{disease}")
        
        return external_data
    
    async def _get_cached_data(
        self, 
        session: AsyncSession, 
        gene: str, 
        disease: str
    ) -> Dict[str, Any]:
        """Get cached external data if still valid"""
        from sqlmodel import select
        
        # Check NHS Scotland cache
        nhs_query = select(ContextAugmentation).where(
            ContextAugmentation.gene == gene.upper(),
            ContextAugmentation.disease == disease.lower(),
            ContextAugmentation.data_source == "nhs_scotland"
        )
        
        nhs_result = await session.execute(nhs_query)
        nhs_cached = nhs_result.scalar_one_or_none()
        
        cached_data = {}
        current_time = datetime.utcnow()
        
        if nhs_cached and (not nhs_cached.expires_at or nhs_cached.expires_at > current_time):
            cached_data["nhs_scotland"] = nhs_cached.context_data
            
            # Update usage count
            nhs_cached.usage_count += 1
            await session.commit()
        
        return cached_data
    
    async def _fetch_fresh_data(self, gene: str, disease: str) -> Dict[str, Any]:
        """Fetch fresh data from external sources"""
        fresh_data = {}
        
        # Fetch NHS Scotland data
        try:
            nhs_data = await self._fetch_nhs_scotland_data(gene, disease)
            if nhs_data:
                fresh_data["nhs_scotland"] = nhs_data
        except Exception as e:
            logger.warning(f"Failed to fetch NHS Scotland data for {gene}-{disease}: {e}")
        
        # Add other data sources here (PubMed, OMIM, etc.)
        # try:
        #     pubmed_data = await self._fetch_pubmed_data(gene, disease)
        #     if pubmed_data:
        #         fresh_data["pubmed"] = pubmed_data
        # except Exception as e:
        #     logger.warning(f"Failed to fetch PubMed data: {e}")
        
        return fresh_data
    
    async def _fetch_nhs_scotland_data(self, gene: str, disease: str) -> Optional[Dict[str, Any]]:
        """Fetch NHS Scotland data for gene-disease combination"""
        try:
            # Search for gene information
            gene_results = await self.nhs_service.search_gene(gene)
            
            # Search for disease information
            disease_results = await self.nhs_service.search_disease(disease)
            
            # Combine and structure the data
            nhs_data = {
                "gene_info": gene_results[:3] if gene_results else [],  # Top 3 results
                "disease_info": disease_results[:3] if disease_results else [],
                "last_updated": datetime.utcnow().isoformat(),
                "source": "nhs_scotland_api"
            }
            
            # Only return data if we found something useful
            if nhs_data["gene_info"] or nhs_data["disease_info"]:
                return nhs_data
            
            return None
            
        except Exception as e:
            logger.error(f"NHS Scotland data fetch failed: {e}")
            return None
    
    async def _cache_external_data(
        self,
        session: AsyncSession,
        gene: str,
        disease: str,
        data: Dict[str, Any]
    ) -> None:
        """Cache external data for future use"""
        try:
            current_time = datetime.utcnow()
            expires_at = current_time + timedelta(hours=self.cache_duration_hours)
            
            for source, source_data in data.items():
                # Check if cache entry already exists
                from sqlmodel import select
                
                query = select(ContextAugmentation).where(
                    ContextAugmentation.gene == gene.upper(),
                    ContextAugmentation.disease == disease.lower(),
                    ContextAugmentation.data_source == source
                )
                
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing cache
                    existing.context_data = source_data
                    existing.expires_at = expires_at
                    existing.last_validated_at = current_time
                    existing.updated_at = current_time
                else:
                    # Create new cache entry
                    cache_entry = ContextAugmentation(
                        gene=gene.upper(),
                        disease=disease.lower(),
                        data_source=source,
                        context_data=source_data,
                        expires_at=expires_at,
                        last_validated_at=current_time,
                        usage_count=0
                    )
                    session.add(cache_entry)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to cache external data: {e}")
            await session.rollback()
    
    async def _format_nhs_context(self, nhs_data: Dict[str, Any]) -> str:
        """Format NHS Scotland data for prompt context"""
        context_parts = []
        
        # Format gene information
        if nhs_data.get("gene_info"):
            gene_descriptions = []
            for gene_info in nhs_data["gene_info"][:2]:  # Top 2
                if gene_info.get("description"):
                    gene_descriptions.append(gene_info["description"][:200])
            
            if gene_descriptions:
                context_parts.append(f"Gene Information: {' '.join(gene_descriptions)}")
        
        # Format disease information
        if nhs_data.get("disease_info"):
            disease_descriptions = []
            for disease_info in nhs_data["disease_info"][:2]:  # Top 2
                if disease_info.get("description"):
                    disease_descriptions.append(disease_info["description"][:200])
            
            if disease_descriptions:
                context_parts.append(f"Disease Information: {' '.join(disease_descriptions)}")
        
        return " | ".join(context_parts)
    
    async def _format_additional_context(self, source: str, data: Dict[str, Any]) -> str:
        """Format additional data sources for prompt context"""
        # Implement formatting for other data sources as needed
        if isinstance(data, dict):
            return str(data).replace("{", "").replace("}", "")[:300]
        return str(data)[:300]
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context to maximum length while preserving structure"""
        if len(context) <= max_length:
            return context
        
        # Try to truncate at sentence boundaries
        truncated = context[:max_length]
        last_period = truncated.rfind('. ')
        last_newline = truncated.rfind('\n')
        
        # Use the latest boundary found
        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.8:  # If boundary is reasonably close to max
            return context[:boundary + 1] + "..."
        else:
            return context[:max_length - 3] + "..."
    
    async def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about context augmentation usage"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, func
                
                # Total cache entries
                total_query = select(func.count(ContextAugmentation.id))
                total_result = await session.execute(total_query)
                total_entries = total_result.scalar() or 0
                
                # Usage statistics
                usage_query = select(
                    ContextAugmentation.data_source,
                    func.count(ContextAugmentation.id).label("count"),
                    func.sum(ContextAugmentation.usage_count).label("total_usage")
                ).group_by(ContextAugmentation.data_source)
                
                usage_result = await session.execute(usage_query)
                usage_stats = {}
                
                for row in usage_result.fetchall():
                    source = row[0]
                    count = row[1] or 0
                    total_usage = row[2] or 0
                    
                    usage_stats[source] = {
                        "cached_entries": count,
                        "total_usage": total_usage,
                        "avg_usage_per_entry": total_usage / count if count > 0 else 0
                    }
                
                return {
                    "total_cached_entries": total_entries,
                    "cache_duration_hours": self.cache_duration_hours,
                    "max_context_length": self.max_context_length,
                    "sources": usage_stats
                }
                
        except Exception as e:
            logger.error(f"Failed to get augmentation stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        try:
            async with get_async_session() as session:
                from sqlmodel import delete
                
                current_time = datetime.utcnow()
                
                # Delete expired entries
                delete_query = delete(ContextAugmentation).where(
                    ContextAugmentation.expires_at < current_time
                )
                
                result = await session.execute(delete_query)
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired context cache entries")
                
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0


# Global service instance
context_augmentation_service = ContextAugmentationService()