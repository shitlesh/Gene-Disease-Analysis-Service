"""
Unit tests for database models and CRUD operations
Uses temporary SQLite database for isolated testing
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, List
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.session import create_test_database, test_session, cleanup_test_database
from app.db.models import UserSession, Analysis, AnalysisChunk, AnalysisStatus, LLMProviderType
from app.db.crud import UserSessionCRUD, AnalysisCRUD, AnalysisChunkCRUD, AnalyticsQueries


@pytest.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database for each test function"""
    engine = await create_test_database()
    yield engine
    await cleanup_test_database(engine)


@pytest.fixture
async def db_session(test_db: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    async with test_session(test_db) as session:
        yield session


class TestUserSessionCRUD:
    """Test UserSession CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_create_session(self, db_session: AsyncSession):
        """Test creating a new user session"""
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser",
            provider="internal",
            session_token="test_token_123",
            preferences={"theme": "dark", "language": "en"}
        )
        
        assert user_session.id is not None
        assert user_session.username == "testuser"
        assert user_session.provider == "internal"
        assert user_session.session_token == "test_token_123"
        assert user_session.preferences == {"theme": "dark", "language": "en"}
        assert user_session.is_active is True
        assert user_session.total_analyses == 0
        assert user_session.total_tokens_used == 0
    
    @pytest.mark.asyncio
    async def test_get_session_by_id(self, session: AsyncSession):
        """Test getting session by ID"""
        # Create session
        created_session = await UserSessionCRUD.create_session(
            session=session,
            username="testuser",
            provider="oauth"
        )
        
        # Retrieve session
        retrieved_session = await UserSessionCRUD.get_session_by_id(
            session=session,
            session_id=created_session.id
        )
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.username == "testuser"
        assert retrieved_session.provider == "oauth"
    
    @pytest.mark.asyncio
    async def test_get_session_by_username(self, session: AsyncSession):
        """Test getting session by username and provider"""
        # Create session
        await UserSessionCRUD.create_session(
            session=session,
            username="testuser",
            provider="github"
        )
        
        # Retrieve by username
        retrieved_session = await UserSessionCRUD.get_session_by_username(
            session=session,
            username="testuser",
            provider="github"
        )
        
        assert retrieved_session is not None
        assert retrieved_session.username == "testuser"
        assert retrieved_session.provider == "github"
        
        # Test with non-existent user
        non_existent = await UserSessionCRUD.get_session_by_username(
            session=session,
            username="nonexistent",
            provider="github"
        )
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_get_session_by_token(self, session: AsyncSession):
        """Test getting session by token"""
        # Create session with token
        created_session = await UserSessionCRUD.create_session(
            session=session,
            username="testuser",
            session_token="unique_token_456"
        )
        
        # Retrieve by token
        retrieved_session = await UserSessionCRUD.get_session_by_token(
            session=session,
            session_token="unique_token_456"
        )
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        
        # Test with invalid token
        invalid_token_session = await UserSessionCRUD.get_session_by_token(
            session=session,
            session_token="invalid_token"
        )
        assert invalid_token_session is None
    
    @pytest.mark.asyncio
    async def test_update_last_active(self, session: AsyncSession):
        """Test updating last active timestamp"""
        # Create session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser"
        )
        
        original_time = user_session.last_active_at
        
        # Wait a bit and update
        await asyncio.sleep(0.01)
        success = await UserSessionCRUD.update_last_active(
            session=session,
            session_id=user_session.id
        )
        
        assert success is True
        
        # Verify update
        updated_session = await UserSessionCRUD.get_session_by_id(
            session=session,
            session_id=user_session.id
        )
        
        assert updated_session.last_active_at > original_time
    
    @pytest.mark.asyncio
    async def test_increment_usage_stats(self, session: AsyncSession):
        """Test incrementing usage statistics"""
        # Create session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser"
        )
        
        # Increment usage
        success = await UserSessionCRUD.increment_usage_stats(
            session=session,
            session_id=user_session.id,
            tokens_used=150
        )
        
        assert success is True
        
        # Verify increment
        updated_session = await UserSessionCRUD.get_session_by_id(
            session=session,
            session_id=user_session.id
        )
        
        assert updated_session.total_analyses == 1
        assert updated_session.total_tokens_used == 150
    
    @pytest.mark.asyncio
    async def test_deactivate_session(self, session: AsyncSession):
        """Test deactivating a session"""
        # Create session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser"
        )
        
        assert user_session.is_active is True
        
        # Deactivate session
        success = await UserSessionCRUD.deactivate_session(
            session=session,
            session_id=user_session.id
        )
        
        assert success is True
        
        # Verify deactivation
        updated_session = await UserSessionCRUD.get_session_by_id(
            session=session,
            session_id=user_session.id
        )
        
        assert updated_session.is_active is False


class TestAnalysisCRUD:
    """Test Analysis CRUD operations"""
    
    @pytest.fixture
    async def sample_user_session(self, session: AsyncSession) -> UserSession:
        """Create a sample user session for testing"""
        return await UserSessionCRUD.create_session(
            session=session,
            username="testuser",
            provider="internal"
        )
    
    @pytest.mark.asyncio
    async def test_create_analysis(self, session: AsyncSession, sample_user_session: UserSession):
        """Test creating a new analysis"""
        analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=sample_user_session.id,
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Test prompt for BRCA1 analysis",
            context="Patient has family history",
            metadata={"test": "metadata"}
        )
        
        assert analysis.id is not None
        assert analysis.session_id == sample_user_session.id
        assert analysis.gene == "BRCA1"
        assert analysis.disease == "breast cancer"
        assert analysis.provider == LLMProviderType.OPENAI
        assert analysis.model == "gpt-4"
        assert analysis.status == AnalysisStatus.PENDING
        assert analysis.metadata == {"test": "metadata"}
    
    @pytest.mark.asyncio
    async def test_get_analysis_by_id(self, session: AsyncSession, sample_user_session: UserSession):
        """Test getting analysis by ID"""
        # Create analysis
        created_analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=sample_user_session.id,
            gene="TP53",
            disease="cancer",
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3",
            prompt_text="Test prompt"
        )
        
        # Retrieve analysis
        retrieved_analysis = await AnalysisCRUD.get_analysis_by_id(
            session=session,
            analysis_id=created_analysis.id
        )
        
        assert retrieved_analysis is not None
        assert retrieved_analysis.id == created_analysis.id
        assert retrieved_analysis.gene == "TP53"
        assert retrieved_analysis.provider == LLMProviderType.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_start_analysis(self, session: AsyncSession, sample_user_session: UserSession):
        """Test starting an analysis"""
        # Create analysis
        analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=sample_user_session.id,
            gene="CFTR",
            disease="cystic fibrosis",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Test prompt"
        )
        
        # Start analysis
        success = await AnalysisCRUD.start_analysis(
            session=session,
            analysis_id=analysis.id
        )
        
        assert success is True
        
        # Verify status change
        updated_analysis = await AnalysisCRUD.get_analysis_by_id(
            session=session,
            analysis_id=analysis.id
        )
        
        assert updated_analysis.status == AnalysisStatus.STREAMING
        assert updated_analysis.started_at is not None
    
    @pytest.mark.asyncio
    async def test_complete_analysis(self, session: AsyncSession, sample_user_session: UserSession):
        """Test completing an analysis"""
        # Create and start analysis
        analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=sample_user_session.id,
            gene="APOE",
            disease="alzheimer's disease",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Test prompt"
        )
        
        await AnalysisCRUD.start_analysis(session=session, analysis_id=analysis.id)
        
        # Wait a bit to simulate processing time
        await asyncio.sleep(0.01)
        
        # Complete analysis
        success = await AnalysisCRUD.complete_analysis(
            session=session,
            analysis_id=analysis.id,
            result_summary="APOE is associated with Alzheimer's disease risk",
            confidence_score=0.92,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=0.006,
            metadata={"completion_metadata": "test"}
        )
        
        assert success is True
        
        # Verify completion
        completed_analysis = await AnalysisCRUD.get_analysis_by_id(
            session=session,
            analysis_id=analysis.id
        )
        
        assert completed_analysis.status == AnalysisStatus.COMPLETED
        assert completed_analysis.completed_at is not None
        assert completed_analysis.processing_time_seconds is not None
        assert completed_analysis.processing_time_seconds > 0
        assert completed_analysis.result_summary == "APOE is associated with Alzheimer's disease risk"
        assert completed_analysis.confidence_score == 0.92
        assert completed_analysis.total_tokens == 300
    
    @pytest.mark.asyncio
    async def test_fail_analysis(self, session: AsyncSession, sample_user_session: UserSession):
        """Test failing an analysis"""
        # Create analysis
        analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=sample_user_session.id,
            gene="TEST",
            disease="test",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Test prompt"
        )
        
        # Fail analysis
        success = await AnalysisCRUD.fail_analysis(
            session=session,
            analysis_id=analysis.id,
            error_message="Test error message",
            retry_count=2
        )
        
        assert success is True
        
        # Verify failure
        failed_analysis = await AnalysisCRUD.get_analysis_by_id(
            session=session,
            analysis_id=analysis.id
        )
        
        assert failed_analysis.status == AnalysisStatus.FAILED
        assert failed_analysis.error_message == "Test error message"
        assert failed_analysis.retry_count == 2
    
    @pytest.mark.asyncio
    async def test_get_user_analysis_history(self, session: AsyncSession, sample_user_session: UserSession):
        """Test getting user analysis history"""
        # Create multiple analyses
        genes = ["BRCA1", "BRCA2", "TP53", "CFTR"]
        diseases = ["breast cancer", "breast cancer", "cancer", "cystic fibrosis"]
        
        created_analyses = []
        for gene, disease in zip(genes, diseases):
            analysis = await AnalysisCRUD.create_analysis(
                session=session,
                session_id=sample_user_session.id,
                gene=gene,
                disease=disease,
                provider=LLMProviderType.OPENAI,
                model="gpt-4",
                prompt_text=f"Test prompt for {gene}"
            )
            created_analyses.append(analysis)
        
        # Complete some analyses
        await AnalysisCRUD.complete_analysis(
            session=session,
            analysis_id=created_analyses[0].id,
            result_summary="Test result 1"
        )
        await AnalysisCRUD.complete_analysis(
            session=session,
            analysis_id=created_analyses[1].id,
            result_summary="Test result 2"
        )
        
        # Get history
        analyses, total_count = await AnalysisCRUD.get_user_analysis_history(
            session=session,
            session_id=sample_user_session.id,
            limit=10
        )
        
        assert total_count == 4
        assert len(analyses) == 4
        
        # Test filtering by status
        completed_analyses, completed_count = await AnalysisCRUD.get_user_analysis_history(
            session=session,
            session_id=sample_user_session.id,
            status_filter=AnalysisStatus.COMPLETED
        )
        
        assert completed_count == 2
        assert len(completed_analyses) == 2
        
        # Test gene filtering
        brca_analyses, brca_count = await AnalysisCRUD.get_user_analysis_history(
            session=session,
            session_id=sample_user_session.id,
            gene_filter="BRCA"
        )
        
        assert brca_count == 2
        assert len(brca_analyses) == 2
    
    @pytest.mark.asyncio
    async def test_search_analyses(self, session: AsyncSession, sample_user_session: UserSession):
        """Test searching analyses"""
        # Create analyses with different content
        test_data = [
            ("BRCA1", "breast cancer", "BRCA1 analysis summary"),
            ("TP53", "lung cancer", "TP53 tumor suppressor analysis"),
            ("CFTR", "cystic fibrosis", "CFTR protein function analysis"),
        ]
        
        for gene, disease, summary in test_data:
            analysis = await AnalysisCRUD.create_analysis(
                session=session,
                session_id=sample_user_session.id,
                gene=gene,
                disease=disease,
                provider=LLMProviderType.OPENAI,
                model="gpt-4",
                prompt_text=f"Test prompt for {gene}"
            )
            await AnalysisCRUD.complete_analysis(
                session=session,
                analysis_id=analysis.id,
                result_summary=summary
            )
        
        # Search by gene
        brca_results, brca_count = await AnalysisCRUD.search_analyses(
            session=session,
            search_term="BRCA1",
            session_id=sample_user_session.id
        )
        
        assert brca_count == 1
        assert len(brca_results) == 1
        assert brca_results[0].gene == "BRCA1"
        
        # Search by disease
        cancer_results, cancer_count = await AnalysisCRUD.search_analyses(
            session=session,
            search_term="cancer",
            session_id=sample_user_session.id
        )
        
        assert cancer_count == 2
        assert len(cancer_results) == 2
        
        # Search by summary content
        protein_results, protein_count = await AnalysisCRUD.search_analyses(
            session=session,
            search_term="protein",
            session_id=sample_user_session.id
        )
        
        assert protein_count == 1
        assert len(protein_results) == 1
        assert protein_results[0].gene == "CFTR"


class TestAnalysisChunkCRUD:
    """Test AnalysisChunk CRUD operations"""
    
    @pytest.fixture
    async def sample_analysis(self, session: AsyncSession) -> Analysis:
        """Create a sample analysis for testing"""
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser"
        )
        
        return await AnalysisCRUD.create_analysis(
            session=session,
            session_id=user_session.id,
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Test prompt"
        )
    
    @pytest.mark.asyncio
    async def test_add_chunk(self, session: AsyncSession, sample_analysis: Analysis):
        """Test adding a chunk to an analysis"""
        chunk = await AnalysisChunkCRUD.add_chunk(
            session=session,
            analysis_id=sample_analysis.id,
            sequence_number=1,
            chunk_text="This is the first chunk of analysis.",
            chunk_type="text",
            metadata={"chunk_info": "test"},
            token_count=25
        )
        
        assert chunk.id is not None
        assert chunk.analysis_id == sample_analysis.id
        assert chunk.sequence_number == 1
        assert chunk.chunk_text == "This is the first chunk of analysis."
        assert chunk.chunk_type == "text"
        assert chunk.token_count == 25
        assert chunk.streaming_complete is False
    
    @pytest.mark.asyncio
    async def test_get_analysis_chunks(self, session: AsyncSession, sample_analysis: Analysis):
        """Test getting all chunks for an analysis"""
        # Add multiple chunks
        chunk_texts = [
            "First chunk of the analysis.",
            "Second chunk with more details.",
            "Final chunk with conclusions."
        ]
        
        for i, text in enumerate(chunk_texts):
            await AnalysisChunkCRUD.add_chunk(
                session=session,
                analysis_id=sample_analysis.id,
                sequence_number=i + 1,
                chunk_text=text,
                chunk_type="text"
            )
        
        # Retrieve all chunks
        chunks = await AnalysisChunkCRUD.get_analysis_chunks(
            session=session,
            analysis_id=sample_analysis.id
        )
        
        assert len(chunks) == 3
        assert chunks[0].sequence_number == 1
        assert chunks[1].sequence_number == 2
        assert chunks[2].sequence_number == 3
        assert chunks[0].chunk_text == "First chunk of the analysis."
        assert chunks[2].chunk_text == "Final chunk with conclusions."
    
    @pytest.mark.asyncio
    async def test_mark_chunk_complete(self, session: AsyncSession, sample_analysis: Analysis):
        """Test marking a chunk as streaming complete"""
        # Add chunk
        chunk = await AnalysisChunkCRUD.add_chunk(
            session=session,
            analysis_id=sample_analysis.id,
            sequence_number=1,
            chunk_text="Test chunk"
        )
        
        assert chunk.streaming_complete is False
        
        # Mark as complete
        success = await AnalysisChunkCRUD.mark_chunk_complete(
            session=session,
            chunk_id=chunk.id
        )
        
        assert success is True
        
        # Verify completion
        chunks = await AnalysisChunkCRUD.get_analysis_chunks(
            session=session,
            analysis_id=sample_analysis.id
        )
        
        assert len(chunks) == 1
        assert chunks[0].streaming_complete is True
    
    @pytest.mark.asyncio
    async def test_get_full_analysis_text(self, session: AsyncSession, sample_analysis: Analysis):
        """Test reconstructing full analysis text from chunks"""
        # Add chunks with sequential text
        chunks_data = [
            (1, "The BRCA1 gene "),
            (2, "is associated with "),
            (3, "hereditary breast cancer risk.")
        ]
        
        for seq, text in chunks_data:
            await AnalysisChunkCRUD.add_chunk(
                session=session,
                analysis_id=sample_analysis.id,
                sequence_number=seq,
                chunk_text=text,
                chunk_type="text"
            )
        
        # Reconstruct full text
        full_text = await AnalysisChunkCRUD.get_full_analysis_text(
            session=session,
            analysis_id=sample_analysis.id
        )
        
        expected_text = "The BRCA1 gene is associated with hereditary breast cancer risk."
        assert full_text == expected_text
    
    @pytest.mark.asyncio
    async def test_delete_analysis_chunks(self, session: AsyncSession, sample_analysis: Analysis):
        """Test deleting all chunks for an analysis"""
        # Add multiple chunks
        for i in range(3):
            await AnalysisChunkCRUD.add_chunk(
                session=session,
                analysis_id=sample_analysis.id,
                sequence_number=i + 1,
                chunk_text=f"Chunk {i + 1}"
            )
        
        # Verify chunks exist
        chunks = await AnalysisChunkCRUD.get_analysis_chunks(
            session=session,
            analysis_id=sample_analysis.id
        )
        assert len(chunks) == 3
        
        # Delete all chunks
        deleted_count = await AnalysisChunkCRUD.delete_analysis_chunks(
            session=session,
            analysis_id=sample_analysis.id
        )
        
        assert deleted_count == 3
        
        # Verify chunks are deleted
        remaining_chunks = await AnalysisChunkCRUD.get_analysis_chunks(
            session=session,
            analysis_id=sample_analysis.id
        )
        assert len(remaining_chunks) == 0


class TestAnalyticsQueries:
    """Test analytics and statistics queries"""
    
    @pytest.fixture
    async def sample_data(self, session: AsyncSession) -> List[Analysis]:
        """Create sample data for analytics testing"""
        # Create user session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="testuser"
        )
        
        # Create multiple analyses with different properties
        analyses_data = [
            ("BRCA1", "breast cancer", LLMProviderType.OPENAI, "completed", 100),
            ("BRCA2", "breast cancer", LLMProviderType.OPENAI, "completed", 150),
            ("TP53", "lung cancer", LLMProviderType.ANTHROPIC, "completed", 200),
            ("CFTR", "cystic fibrosis", LLMProviderType.ANTHROPIC, "failed", 0),
            ("APOE", "alzheimer's", LLMProviderType.OPENAI, "completed", 120),
        ]
        
        created_analyses = []
        for gene, disease, provider, status, tokens in analyses_data:
            analysis = await AnalysisCRUD.create_analysis(
                session=session,
                session_id=user_session.id,
                gene=gene,
                disease=disease,
                provider=provider,
                model="test-model",
                prompt_text="Test prompt"
            )
            
            if status == "completed":
                await AnalysisCRUD.complete_analysis(
                    session=session,
                    analysis_id=analysis.id,
                    result_summary=f"Analysis of {gene}",
                    total_tokens=tokens
                )
            elif status == "failed":
                await AnalysisCRUD.fail_analysis(
                    session=session,
                    analysis_id=analysis.id,
                    error_message="Test error"
                )
            
            created_analyses.append(analysis)
        
        return created_analyses
    
    @pytest.mark.asyncio
    async def test_get_usage_statistics(self, session: AsyncSession, sample_data: List[Analysis]):
        """Test getting usage statistics"""
        stats = await AnalyticsQueries.get_usage_statistics(
            session=session,
            days=30
        )
        
        assert stats["total_analyses"] == 5
        assert stats["completed_analyses"] == 4
        assert stats["success_rate"] == 0.8  # 4/5
        assert stats["total_tokens_used"] == 570  # 100+150+200+120
        assert len(stats["popular_genes"]) <= 5
        assert len(stats["popular_diseases"]) <= 5
        
        # Check popular genes
        gene_names = [item["gene"] for item in stats["popular_genes"]]
        assert "BRCA1" in gene_names
        assert "BRCA2" in gene_names
    
    @pytest.mark.asyncio
    async def test_get_provider_statistics(self, session: AsyncSession, sample_data: List[Analysis]):
        """Test getting provider statistics"""
        provider_stats = await AnalyticsQueries.get_provider_statistics(
            session=session,
            days=30
        )
        
        # Check OpenAI stats
        openai_stats = provider_stats.get("openai")
        assert openai_stats is not None
        assert openai_stats["total_analyses"] == 3  # BRCA1, BRCA2, APOE
        assert openai_stats["completed_analyses"] == 3
        assert openai_stats["success_rate"] == 1.0
        assert openai_stats["total_tokens"] == 370  # 100+150+120
        
        # Check Anthropic stats
        anthropic_stats = provider_stats.get("anthropic")
        assert anthropic_stats is not None
        assert anthropic_stats["total_analyses"] == 2  # TP53, CFTR
        assert anthropic_stats["completed_analyses"] == 1  # Only TP53
        assert anthropic_stats["success_rate"] == 0.5
        assert anthropic_stats["total_tokens"] == 200  # Only TP53


class TestDatabaseIntegration:
    """Test database integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_lifecycle(self, session: AsyncSession):
        """Test complete analysis lifecycle from creation to completion"""
        # Create user session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="integration_test_user",
            session_token="integration_token"
        )
        
        # Create analysis
        analysis = await AnalysisCRUD.create_analysis(
            session=session,
            session_id=user_session.id,
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            prompt_text="Analyze BRCA1 and breast cancer relationship",
            context="Patient has family history",
            metadata={"request_id": "test_123"}
        )
        
        # Start analysis
        await AnalysisCRUD.start_analysis(session=session, analysis_id=analysis.id)
        
        # Add streaming chunks
        chunks_data = [
            "BRCA1 is a tumor suppressor gene ",
            "that plays a critical role in DNA repair. ",
            "Mutations in BRCA1 significantly increase ",
            "the risk of developing breast and ovarian cancer."
        ]
        
        for i, chunk_text in enumerate(chunks_data):
            chunk = await AnalysisChunkCRUD.add_chunk(
                session=session,
                analysis_id=analysis.id,
                sequence_number=i + 1,
                chunk_text=chunk_text,
                chunk_type="text",
                token_count=len(chunk_text.split())
            )
            
            await AnalysisChunkCRUD.mark_chunk_complete(
                session=session,
                chunk_id=chunk.id
            )
        
        # Complete analysis
        full_summary = await AnalysisChunkCRUD.get_full_analysis_text(
            session=session,
            analysis_id=analysis.id
        )
        
        await AnalysisCRUD.complete_analysis(
            session=session,
            analysis_id=analysis.id,
            result_summary=full_summary.strip(),
            confidence_score=0.95,
            input_tokens=50,
            output_tokens=200,
            total_tokens=250,
            estimated_cost=0.01,
            metadata={"completion_time": "2023-01-01T12:00:00"}
        )
        
        # Update user session stats
        await UserSessionCRUD.increment_usage_stats(
            session=session,
            session_id=user_session.id,
            tokens_used=250
        )
        
        # Verify final state
        final_analysis = await AnalysisCRUD.get_analysis_by_id(
            session=session,
            analysis_id=analysis.id,
            include_chunks=True,
            include_session=True
        )
        
        assert final_analysis.status == AnalysisStatus.COMPLETED
        assert final_analysis.result_summary.startswith("BRCA1 is a tumor suppressor gene")
        assert final_analysis.confidence_score == 0.95
        assert final_analysis.total_tokens == 250
        assert len(final_analysis.chunks) == 4
        assert final_analysis.session.total_analyses == 1
        assert final_analysis.session.total_tokens_used == 250
        
        # Test analysis history
        history, count = await AnalysisCRUD.get_user_analysis_history(
            session=session,
            session_id=user_session.id
        )
        
        assert count == 1
        assert len(history) == 1
        assert history[0].id == analysis.id
        
        # Test search functionality
        search_results, search_count = await AnalysisCRUD.search_analyses(
            session=session,
            search_term="BRCA1",
            session_id=user_session.id
        )
        
        assert search_count == 1
        assert search_results[0].id == analysis.id
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, session: AsyncSession):
        """Test concurrent database operations"""
        # Create user session
        user_session = await UserSessionCRUD.create_session(
            session=db_session,
            username="concurrent_test_user"
        )
        
        # Create multiple analyses concurrently
        async def create_analysis_with_chunks(gene: str, disease: str):
            analysis = await AnalysisCRUD.create_analysis(
                session=session,
                session_id=user_session.id,
                gene=gene,
                disease=disease,
                provider=LLMProviderType.OPENAI,
                model="gpt-4",
                prompt_text=f"Test prompt for {gene}"
            )
            
            # Add chunks
            for i in range(3):
                await AnalysisChunkCRUD.add_chunk(
                    session=session,
                    analysis_id=analysis.id,
                    sequence_number=i + 1,
                    chunk_text=f"Chunk {i + 1} for {gene} analysis"
                )
            
            return analysis
        
        # Run concurrent operations
        genes_diseases = [
            ("BRCA1", "breast cancer"),
            ("TP53", "lung cancer"),
            ("CFTR", "cystic fibrosis"),
            ("APOE", "alzheimer's disease")
        ]
        
        tasks = [
            create_analysis_with_chunks(gene, disease)
            for gene, disease in genes_diseases
        ]
        
        analyses = await asyncio.gather(*tasks)
        
        # Verify all analyses were created
        assert len(analyses) == 4
        
        # Verify all chunks were created
        total_chunks = 0
        for analysis in analyses:
            chunks = await AnalysisChunkCRUD.get_analysis_chunks(
                session=session,
                analysis_id=analysis.id
            )
            total_chunks += len(chunks)
        
        assert total_chunks == 12  # 4 analyses * 3 chunks each
        
        # Verify user session history
        history, count = await AnalysisCRUD.get_user_analysis_history(
            session=session,
            session_id=user_session.id
        )
        
        assert count == 4
        assert len(history) == 4


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])