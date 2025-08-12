"""
CRUD operations for LLM analysis persistence
Provides async database operations for UserSession, Analysis, and AnalysisChunk models
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, and_, or_, func, text, case
from sqlalchemy.orm import selectinload
from sqlmodel import select, delete, update

from .models import (
    UserSession, Analysis, AnalysisChunk, AnalysisStatus, 
    LLMProviderType, create_analysis_from_request, create_analysis_chunk
)

logger = logging.getLogger(__name__)


# UserSession CRUD Operations

class UserSessionCRUD:
    """CRUD operations for UserSession model"""
    
    @staticmethod
    async def create_session(
        session: AsyncSession,
        username: str,
        provider: str = "internal",
        external_id: Optional[str] = None,
        session_token: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> UserSession:
        """
        Create a new user session
        
        Args:
            session: Database session
            username: Username
            provider: Authentication provider
            external_id: External provider user ID
            session_token: Session token
            preferences: User preferences
            expires_at: Session expiration time
            
        Returns:
            Created UserSession instance
        """
        user_session = UserSession(
            username=username,
            provider=provider,
            external_id=external_id,
            session_token=session_token,
            preferences=preferences or {},
            expires_at=expires_at,
            is_active=True
        )
        
        session.add(user_session)
        await session.commit()
        await session.refresh(user_session)
        
        logger.info(f"Created user session for {username} (ID: {user_session.id})")
        return user_session
    
    @staticmethod
    async def get_session_by_id(
        session: AsyncSession,
        session_id: int,
        include_analyses: bool = False
    ) -> Optional[UserSession]:
        """
        Get user session by ID
        
        Args:
            session: Database session
            session_id: Session ID
            include_analyses: Whether to include related analyses
            
        Returns:
            UserSession if found, None otherwise
        """
        query = select(UserSession).where(UserSession.id == session_id)
        
        if include_analyses:
            query = query.options(selectinload(UserSession.analyses))
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_session_by_username(
        session: AsyncSession,
        username: str,
        provider: str = "internal"
    ) -> Optional[UserSession]:
        """
        Get user session by username and provider
        
        Args:
            session: Database session
            username: Username
            provider: Authentication provider
            
        Returns:
            UserSession if found, None otherwise
        """
        query = select(UserSession).where(
            and_(
                UserSession.username == username,
                UserSession.provider == provider,
                UserSession.is_active == True
            )
        )
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_session_by_token(
        session: AsyncSession,
        session_token: str
    ) -> Optional[UserSession]:
        """
        Get user session by session token
        
        Args:
            session: Database session
            session_token: Session token
            
        Returns:
            UserSession if found and valid, None otherwise
        """
        query = select(UserSession).where(
            and_(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                or_(
                    UserSession.expires_at.is_(None),
                    UserSession.expires_at > datetime.utcnow()
                )
            )
        )
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_last_active(
        session: AsyncSession,
        session_id: int
    ) -> bool:
        """
        Update last active timestamp for a session
        
        Args:
            session: Database session
            session_id: Session ID
            
        Returns:
            True if updated, False if session not found
        """
        query = (
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(last_active_at=datetime.utcnow())
        )
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount > 0
    
    @staticmethod
    async def increment_usage_stats(
        session: AsyncSession,
        session_id: int,
        tokens_used: int = 0
    ) -> bool:
        """
        Increment usage statistics for a session
        
        Args:
            session: Database session
            session_id: Session ID
            tokens_used: Number of tokens used
            
        Returns:
            True if updated, False if session not found
        """
        query = (
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(
                total_analyses=UserSession.total_analyses + 1,
                total_tokens_used=UserSession.total_tokens_used + tokens_used,
                last_active_at=datetime.utcnow()
            )
        )
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount > 0
    
    @staticmethod
    async def deactivate_session(
        session: AsyncSession,
        session_id: int
    ) -> bool:
        """
        Deactivate a user session
        
        Args:
            session: Database session
            session_id: Session ID
            
        Returns:
            True if deactivated, False if session not found
        """
        query = (
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount > 0
    
    @staticmethod
    async def cleanup_expired_sessions(
        session: AsyncSession,
        older_than_hours: int = 24
    ) -> int:
        """
        Cleanup expired sessions
        
        Args:
            session: Database session
            older_than_hours: Hours after which inactive sessions are cleaned up
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        # Delete sessions that are either explicitly expired or inactive for too long
        query = delete(UserSession).where(
            or_(
                and_(
                    UserSession.expires_at.is_not(None),
                    UserSession.expires_at < datetime.utcnow()
                ),
                UserSession.last_active_at < cutoff_time
            )
        )
        
        result = await session.execute(query)
        await session.commit()
        
        logger.info(f"Cleaned up {result.rowcount} expired user sessions")
        return result.rowcount


# Analysis CRUD Operations

class AnalysisCRUD:
    """CRUD operations for Analysis model"""
    
    @staticmethod
    async def create_analysis(
        session: AsyncSession,
        session_id: int,
        gene: str,
        disease: str,
        provider: LLMProviderType,
        model: str,
        prompt_text: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """
        Create a new analysis entry
        
        Args:
            session: Database session
            session_id: User session ID
            gene: Gene symbol
            disease: Disease name
            provider: LLM provider
            model: Model name
            prompt_text: Generated prompt
            context: Additional context
            metadata: Analysis metadata
            
        Returns:
            Created Analysis instance
        """
        analysis = create_analysis_from_request(
            session_id=session_id,
            gene=gene,
            disease=disease,
            provider=provider,
            model=model,
            prompt_text=prompt_text,
            context=context,
            metadata=metadata
        )
        
        session.add(analysis)
        await session.commit()
        await session.refresh(analysis)
        
        logger.info(f"Created analysis for {gene}-{disease} (ID: {analysis.id})")
        return analysis
    
    @staticmethod
    async def get_analysis_by_id(
        session: AsyncSession,
        analysis_id: int,
        include_chunks: bool = False,
        include_session: bool = False
    ) -> Optional[Analysis]:
        """
        Get analysis by ID
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            include_chunks: Whether to include analysis chunks
            include_session: Whether to include user session
            
        Returns:
            Analysis if found, None otherwise
        """
        query = select(Analysis).where(Analysis.id == analysis_id)
        
        options = []
        if include_chunks:
            options.append(selectinload(Analysis.chunks))
        if include_session:
            options.append(selectinload(Analysis.session))
        
        if options:
            query = query.options(*options)
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def start_analysis(
        session: AsyncSession,
        analysis_id: int
    ) -> bool:
        """
        Mark analysis as started
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            
        Returns:
            True if updated, False if analysis not found
        """
        query = (
            update(Analysis)
            .where(Analysis.id == analysis_id)
            .values(
                status=AnalysisStatus.STREAMING,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount > 0
    
    @staticmethod
    async def complete_analysis(
        session: AsyncSession,
        analysis_id: int,
        result_summary: str,
        confidence_score: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark analysis as completed with results
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            result_summary: Analysis result summary
            confidence_score: Confidence score (0-1)
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            total_tokens: Total tokens used
            estimated_cost: Estimated cost
            metadata: Additional metadata
            
        Returns:
            True if updated, False if analysis not found
        """
        now = datetime.utcnow()
        
        # Get current analysis to calculate processing time
        current_analysis = await AnalysisCRUD.get_analysis_by_id(session, analysis_id)
        if not current_analysis:
            return False
        
        processing_time = None
        if current_analysis.started_at:
            processing_time = (now - current_analysis.started_at).total_seconds()
        
        # Update metadata with existing metadata
        final_metadata = current_analysis.analysis_metadata.copy() if current_analysis.analysis_metadata else {}
        if metadata:
            final_metadata.update(metadata)
        
        query = (
            update(Analysis)
            .where(Analysis.id == analysis_id)
            .values(
                status=AnalysisStatus.COMPLETED,
                completed_at=now,
                processing_time_seconds=processing_time,
                result_summary=result_summary,
                confidence_score=confidence_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                analysis_metadata=final_metadata,
                updated_at=now
            )
        )
        
        result = await session.execute(query)
        await session.commit()
        
        logger.info(f"Completed analysis {analysis_id} in {processing_time:.2f}s" if processing_time else f"Completed analysis {analysis_id}")
        return result.rowcount > 0
    
    @staticmethod
    async def fail_analysis(
        session: AsyncSession,
        analysis_id: int,
        error_message: str,
        retry_count: Optional[int] = None
    ) -> bool:
        """
        Mark analysis as failed
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            error_message: Error message
            retry_count: Current retry count
            
        Returns:
            True if updated, False if analysis not found
        """
        now = datetime.utcnow()
        
        # Get current analysis to calculate processing time
        current_analysis = await AnalysisCRUD.get_analysis_by_id(session, analysis_id)
        if not current_analysis:
            return False
        
        processing_time = None
        if current_analysis.started_at:
            processing_time = (now - current_analysis.started_at).total_seconds()
        
        update_values = {
            "status": AnalysisStatus.FAILED,
            "completed_at": now,
            "error_message": error_message,
            "updated_at": now
        }
        
        if processing_time:
            update_values["processing_time_seconds"] = processing_time
        
        if retry_count is not None:
            update_values["retry_count"] = retry_count
        
        query = update(Analysis).where(Analysis.id == analysis_id).values(**update_values)
        
        result = await session.execute(query)
        await session.commit()
        
        logger.warning(f"Failed analysis {analysis_id}: {error_message}")
        return result.rowcount > 0
    
    @staticmethod
    async def get_user_analysis_history(
        session: AsyncSession,
        session_id: int,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[AnalysisStatus] = None,
        gene_filter: Optional[str] = None,
        disease_filter: Optional[str] = None
    ) -> Tuple[List[Analysis], int]:
        """
        Get analysis history for a user session
        
        Args:
            session: Database session
            session_id: User session ID
            limit: Maximum number of results
            offset: Results offset for pagination
            status_filter: Filter by status
            gene_filter: Filter by gene (partial match)
            disease_filter: Filter by disease (partial match)
            
        Returns:
            Tuple of (analyses list, total count)
        """
        # Build base query
        query = select(Analysis).where(Analysis.session_id == session_id)
        count_query = select(func.count(Analysis.id)).where(Analysis.session_id == session_id)
        
        # Apply filters
        if status_filter:
            query = query.where(Analysis.status == status_filter)
            count_query = count_query.where(Analysis.status == status_filter)
        
        if gene_filter:
            gene_condition = Analysis.gene.ilike(f"%{gene_filter}%")
            query = query.where(gene_condition)
            count_query = count_query.where(gene_condition)
        
        if disease_filter:
            disease_condition = Analysis.disease.ilike(f"%{disease_filter}%")
            query = query.where(disease_condition)
            count_query = count_query.where(disease_condition)
        
        # Order by most recent first
        query = query.order_by(desc(Analysis.created_at))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute queries
        analyses_result = await session.execute(query)
        count_result = await session.execute(count_query)
        
        analyses = analyses_result.scalars().all()
        total_count = count_result.scalar() or 0
        
        return list(analyses), total_count
    
    @staticmethod
    async def search_analyses(
        session: AsyncSession,
        search_term: str,
        session_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Analysis], int]:
        """
        Search analyses by gene, disease, or summary content
        
        Args:
            session: Database session
            search_term: Search term
            session_id: Optional session ID filter
            limit: Maximum number of results
            offset: Results offset for pagination
            
        Returns:
            Tuple of (analyses list, total count)
        """
        search_pattern = f"%{search_term}%"
        
        # Build search condition
        search_condition = or_(
            Analysis.gene.ilike(search_pattern),
            Analysis.disease.ilike(search_pattern),
            Analysis.result_summary.ilike(search_pattern),
            Analysis.context.ilike(search_pattern)
        )
        
        # Base query
        query = select(Analysis).where(search_condition)
        count_query = select(func.count(Analysis.id)).where(search_condition)
        
        # Apply session filter if provided
        if session_id:
            session_condition = Analysis.session_id == session_id
            query = query.where(session_condition)
            count_query = count_query.where(session_condition)
        
        # Order by relevance (exact matches first, then by date)
        query = query.order_by(
            desc(Analysis.gene.ilike(search_term)),  # Exact gene matches first
            desc(Analysis.created_at)
        )
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute queries
        analyses_result = await session.execute(query)
        count_result = await session.execute(count_query)
        
        analyses = analyses_result.scalars().all()
        total_count = count_result.scalar() or 0
        
        return list(analyses), total_count


# AnalysisChunk CRUD Operations

class AnalysisChunkCRUD:
    """CRUD operations for AnalysisChunk model"""
    
    @staticmethod
    async def add_chunk(
        session: AsyncSession,
        analysis_id: int,
        sequence_number: int,
        chunk_text: str,
        chunk_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> AnalysisChunk:
        """
        Add a chunk to an analysis
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            sequence_number: Sequence number for ordering
            chunk_text: Chunk content
            chunk_type: Type of chunk (text, json, metadata)
            metadata: Chunk metadata
            token_count: Number of tokens in chunk
            
        Returns:
            Created AnalysisChunk instance
        """
        chunk = create_analysis_chunk(
            analysis_id=analysis_id,
            sequence_number=sequence_number,
            chunk_text=chunk_text,
            chunk_type=chunk_type,
            metadata=metadata
        )
        
        if token_count:
            chunk.token_count = token_count
        
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        
        logger.debug(f"Added chunk {sequence_number} to analysis {analysis_id}")
        return chunk
    
    @staticmethod
    async def get_analysis_chunks(
        session: AsyncSession,
        analysis_id: int,
        chunk_type: Optional[str] = None
    ) -> List[AnalysisChunk]:
        """
        Get all chunks for an analysis
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            chunk_type: Optional chunk type filter
            
        Returns:
            List of AnalysisChunk instances ordered by sequence
        """
        query = (
            select(AnalysisChunk)
            .where(AnalysisChunk.analysis_id == analysis_id)
            .order_by(AnalysisChunk.sequence_number)
        )
        
        if chunk_type:
            query = query.where(AnalysisChunk.chunk_type == chunk_type)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def mark_chunk_complete(
        session: AsyncSession,
        chunk_id: int
    ) -> bool:
        """
        Mark a chunk as streaming complete
        
        Args:
            session: Database session
            chunk_id: Chunk ID
            
        Returns:
            True if updated, False if chunk not found
        """
        query = (
            update(AnalysisChunk)
            .where(AnalysisChunk.id == chunk_id)
            .values(
                streaming_complete=True,
                updated_at=datetime.utcnow()
            )
        )
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount > 0
    
    @staticmethod
    async def get_full_analysis_text(
        session: AsyncSession,
        analysis_id: int,
        chunk_type: str = "text"
    ) -> str:
        """
        Reconstruct full analysis text from chunks
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            chunk_type: Chunk type to include
            
        Returns:
            Reconstructed full text
        """
        chunks = await AnalysisChunkCRUD.get_analysis_chunks(
            session, analysis_id, chunk_type
        )
        
        return "".join(chunk.chunk_text for chunk in chunks)
    
    @staticmethod
    async def delete_analysis_chunks(
        session: AsyncSession,
        analysis_id: int
    ) -> int:
        """
        Delete all chunks for an analysis
        
        Args:
            session: Database session
            analysis_id: Analysis ID
            
        Returns:
            Number of chunks deleted
        """
        query = delete(AnalysisChunk).where(AnalysisChunk.analysis_id == analysis_id)
        
        result = await session.execute(query)
        await session.commit()
        
        return result.rowcount


# Statistics and Analytics

class AnalyticsQueries:
    """Analytics and statistics queries"""
    
    @staticmethod
    async def get_usage_statistics(
        session: AsyncSession,
        session_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Args:
            session: Database session
            session_id: Optional session ID filter
            days: Number of days to include
            
        Returns:
            Dictionary with usage statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        base_filter = Analysis.created_at >= cutoff_date
        if session_id:
            base_filter = and_(base_filter, Analysis.session_id == session_id)
        
        # Total analyses
        total_query = select(func.count(Analysis.id)).where(base_filter)
        total_result = await session.execute(total_query)
        total_analyses = total_result.scalar() or 0
        
        # Completed analyses
        completed_query = select(func.count(Analysis.id)).where(
            and_(base_filter, Analysis.status == AnalysisStatus.COMPLETED)
        )
        completed_result = await session.execute(completed_query)
        completed_analyses = completed_result.scalar() or 0
        
        # Average processing time
        processing_time_query = select(func.avg(Analysis.processing_time_seconds)).where(
            and_(base_filter, Analysis.processing_time_seconds.is_not(None))
        )
        processing_time_result = await session.execute(processing_time_query)
        avg_processing_time = processing_time_result.scalar()
        
        # Token usage
        token_query = select(func.sum(Analysis.total_tokens)).where(
            and_(base_filter, Analysis.total_tokens.is_not(None))
        )
        token_result = await session.execute(token_query)
        total_tokens = token_result.scalar() or 0
        
        # Popular genes and diseases
        gene_query = select(
            Analysis.gene,
            func.count(Analysis.id).label("count")
        ).where(base_filter).group_by(Analysis.gene).order_by(desc("count")).limit(10)
        
        gene_result = await session.execute(gene_query)
        popular_genes = [{"gene": row[0], "count": row[1]} for row in gene_result.fetchall()]
        
        disease_query = select(
            Analysis.disease,
            func.count(Analysis.id).label("count")
        ).where(base_filter).group_by(Analysis.disease).order_by(desc("count")).limit(10)
        
        disease_result = await session.execute(disease_query)
        popular_diseases = [{"disease": row[0], "count": row[1]} for row in disease_result.fetchall()]
        
        return {
            "period_days": days,
            "total_analyses": total_analyses,
            "completed_analyses": completed_analyses,
            "success_rate": completed_analyses / total_analyses if total_analyses > 0 else 0,
            "average_processing_time": float(avg_processing_time) if avg_processing_time else 0,
            "total_tokens_used": total_tokens,
            "popular_genes": popular_genes,
            "popular_diseases": popular_diseases
        }
    
    @staticmethod
    async def get_provider_statistics(
        session: AsyncSession,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get statistics by LLM provider
        
        Args:
            session: Database session
            days: Number of days to include
            
        Returns:
            Dictionary with provider statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        base_filter = Analysis.created_at >= cutoff_date
        
        provider_query = select(
            Analysis.provider,
            func.count(Analysis.id).label("total"),
            func.sum(
                case(
                    (Analysis.status == AnalysisStatus.COMPLETED, 1),
                    else_=0
                )
            ).label("completed"),
            func.avg(Analysis.processing_time_seconds).label("avg_time"),
            func.sum(Analysis.total_tokens).label("total_tokens")
        ).where(base_filter).group_by(Analysis.provider)
        
        result = await session.execute(provider_query)
        
        provider_stats = {}
        for row in result.fetchall():
            provider = row[0]
            total = row[1] or 0
            completed = row[2] or 0
            avg_time = float(row[3]) if row[3] else 0
            tokens = row[4] or 0
            
            provider_stats[provider] = {
                "total_analyses": total,
                "completed_analyses": completed,
                "success_rate": completed / total if total > 0 else 0,
                "average_processing_time": avg_time,
                "total_tokens": tokens
            }
        
        return provider_stats