#!/usr/bin/env python3
"""
Simple test for database functionality
"""

import asyncio
from app.db.session import create_test_database, test_session, cleanup_test_database
from app.db.crud import UserSessionCRUD, AnalysisCRUD, AnalysisChunkCRUD
from app.db.models import LLMProviderType


async def test_database_operations():
    """Test basic database operations"""
    print("🔬 Testing database operations...")
    
    # Create test database
    engine = await create_test_database()
    
    async with test_session(engine) as session:
        try:
            # Test 1: Create user session
            print("📝 Creating user session...")
            user_session = await UserSessionCRUD.create_session(
                session=session,
                username="testuser",
                provider="internal",
                session_token="test_token_123",
                preferences={"theme": "dark", "language": "en"}
            )
            print(f"✅ Created user session: ID={user_session.id}, username={user_session.username}")
            
            # Test 2: Create analysis
            print("📊 Creating analysis...")
            analysis = await AnalysisCRUD.create_analysis(
                session=session,
                session_id=user_session.id,
                gene="BRCA1",
                disease="breast cancer",
                provider=LLMProviderType.OPENAI,
                model="gpt-4",
                prompt_text="Test prompt for BRCA1 analysis",
                context="Patient has family history",
                metadata={"test": "metadata"}
            )
            print(f"✅ Created analysis: ID={analysis.id}, gene={analysis.gene}, status={analysis.status}")
            
            # Test 3: Start analysis
            print("🚀 Starting analysis...")
            await AnalysisCRUD.start_analysis(session=session, analysis_id=analysis.id)
            
            # Test 4: Add chunks
            print("📦 Adding analysis chunks...")
            chunks_data = [
                "BRCA1 is a tumor suppressor gene ",
                "that plays a critical role in DNA repair. ",
                "Mutations significantly increase cancer risk."
            ]
            
            for i, chunk_text in enumerate(chunks_data):
                chunk = await AnalysisChunkCRUD.add_chunk(
                    session=session,
                    analysis_id=analysis.id,
                    sequence_number=i + 1,
                    chunk_text=chunk_text,
                    chunk_type="text"
                )
                print(f"   Added chunk {i + 1}: {len(chunk_text)} characters")
            
            # Test 5: Complete analysis
            print("✅ Completing analysis...")
            full_text = await AnalysisChunkCRUD.get_full_analysis_text(
                session=session,
                analysis_id=analysis.id
            )
            
            await AnalysisCRUD.complete_analysis(
                session=session,
                analysis_id=analysis.id,
                result_summary=full_text.strip(),
                confidence_score=0.92,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=0.006
            )
            
            # Test 6: Get analysis history
            print("📚 Getting analysis history...")
            history, count = await AnalysisCRUD.get_user_analysis_history(
                session=session,
                session_id=user_session.id
            )
            print(f"✅ Found {count} analyses in history")
            
            # Test 7: Search analyses
            print("🔍 Searching analyses...")
            search_results, search_count = await AnalysisCRUD.search_analyses(
                session=session,
                search_term="BRCA1",
                session_id=user_session.id
            )
            print(f"✅ Found {search_count} analyses matching 'BRCA1'")
            
            print("\n🎉 All database operations completed successfully!")
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            import traceback
            traceback.print_exc()
        
    # Cleanup
    await cleanup_test_database(engine)
    print("🧹 Database cleanup completed")


async def test_analytics():
    """Test analytics queries"""
    print("\n📈 Testing analytics queries...")
    
    engine = await create_test_database()
    
    async with test_session(engine) as session:
        try:
            # Create sample data
            user_session = await UserSessionCRUD.create_session(
                session=session,
                username="analytics_user"
            )
            
            # Create multiple analyses
            genes = ["BRCA1", "BRCA2", "TP53", "CFTR"]
            diseases = ["breast cancer", "breast cancer", "lung cancer", "cystic fibrosis"]
            
            for gene, disease in zip(genes, diseases):
                analysis = await AnalysisCRUD.create_analysis(
                    session=session,
                    session_id=user_session.id,
                    gene=gene,
                    disease=disease,
                    provider=LLMProviderType.OPENAI,
                    model="gpt-4",
                    prompt_text=f"Test prompt for {gene}"
                )
                
                # Complete some analyses
                if gene != "CFTR":  # Leave one pending
                    await AnalysisCRUD.complete_analysis(
                        session=session,
                        analysis_id=analysis.id,
                        result_summary=f"Analysis of {gene} and {disease}",
                        total_tokens=150
                    )
            
            # Test analytics
            from app.db.crud import AnalyticsQueries
            
            stats = await AnalyticsQueries.get_usage_statistics(
                session=session,
                session_id=user_session.id,
                days=30
            )
            
            print(f"✅ Usage statistics:")
            print(f"   Total analyses: {stats['total_analyses']}")
            print(f"   Completed: {stats['completed_analyses']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Total tokens: {stats['total_tokens_used']}")
            
            provider_stats = await AnalyticsQueries.get_provider_statistics(
                session=session,
                days=30
            )
            
            print(f"✅ Provider statistics:")
            for provider, data in provider_stats.items():
                print(f"   {provider}: {data['total_analyses']} analyses, {data['success_rate']:.1%} success")
            
        except Exception as e:
            print(f"❌ Analytics test failed: {e}")
            import traceback
            traceback.print_exc()
    
    await cleanup_test_database(engine)


if __name__ == "__main__":
    async def main():
        await test_database_operations()
        await test_analytics()
    
    asyncio.run(main())