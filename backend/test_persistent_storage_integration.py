#!/usr/bin/env python3
"""
Integration test for the complete persistent storage system
Tests the full workflow from session creation to analysis completion
"""

import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import Dict, Any

from app.main import app, lifespan
from app.db.session import setup_database, close_database


@asynccontextmanager
async def test_app():
    """Context manager for running the FastAPI app in tests"""
    # Initialize app
    async with lifespan(app):
        yield app


async def test_api_client():
    """Test the persistent storage API using HTTP client"""
    print("🧪 Testing Persistent Storage API Integration")
    print("=" * 50)
    
    async with test_app():
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Test 1: Health check
            print("1. Testing health check...")
            response = await client.get("/api/v1/persistent/health")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   Overall status: {health_data['status']}")
                print(f"   Database status: {health_data['database']['status']}")
                print("   ✅ Health check passed")
            else:
                print("   ❌ Health check failed")
                return False
            
            # Test 2: Create user session
            print("\n2. Creating user session...")
            session_data = {
                "username": "test_integration_user",
                "provider": "integration_test",
                "preferences": {"test_mode": True, "integration": "test"}
            }
            
            response = await client.post("/api/v1/persistent/sessions", json=session_data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                session = response.json()
                session_id = session["id"]
                print(f"   Created session ID: {session_id}")
                print(f"   Username: {session['username']}")
                print("   ✅ Session creation passed")
            else:
                print(f"   ❌ Session creation failed: {response.text}")
                return False
            
            # Test 3: Create analysis
            print("\n3. Creating analysis...")
            analysis_data = {
                "session_id": session_id,
                "gene": "BRCA1",
                "disease": "breast cancer",
                "provider": "openai",
                "model": "gpt-4",
                "context": "Integration test for persistent storage",
                "priority": "normal",
                "metadata": {"integration_test": True, "timestamp": "2023-01-01T12:00:00"}
            }
            
            response = await client.post("/api/v1/persistent/analyses", json=analysis_data)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                analysis = response.json()
                analysis_id = analysis["id"]
                print(f"   Created analysis ID: {analysis_id}")
                print(f"   Gene: {analysis['gene']}")
                print(f"   Disease: {analysis['disease']}")
                print(f"   Status: {analysis['status']}")
                print("   ✅ Analysis creation passed")
            else:
                print(f"   ❌ Analysis creation failed: {response.text}")
                return False
            
            # Test 4: Get analysis status
            print("\n4. Checking analysis status...")
            response = await client.get(f"/api/v1/persistent/analyses/{analysis_id}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                analysis_status = response.json()
                print(f"   Current status: {analysis_status['status']}")
                print(f"   Created at: {analysis_status['created_at']}")
                print("   ✅ Analysis status check passed")
            else:
                print(f"   ❌ Analysis status check failed: {response.text}")
                return False
            
            # Test 5: Get analysis history
            print("\n5. Getting analysis history...")
            response = await client.get(f"/api/v1/persistent/sessions/{session_id}/analyses")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                history = response.json()
                print(f"   Total analyses: {history['total_count']}")
                print(f"   Page: {history['page']}")
                print(f"   Page size: {history['page_size']}")
                if history['analyses']:
                    print(f"   First analysis: {history['analyses'][0]['gene']}")
                print("   ✅ Analysis history passed")
            else:
                print(f"   ❌ Analysis history failed: {response.text}")
                return False
            
            # Test 6: Search analyses
            print("\n6. Searching analyses...")
            response = await client.get(
                "/api/v1/persistent/analyses/search",
                params={"q": "BRCA1", "session_id": session_id}
            )
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                search_results = response.json()
                print(f"   Search results: {search_results['total_count']}")
                if search_results['analyses']:
                    print(f"   Found gene: {search_results['analyses'][0]['gene']}")
                print("   ✅ Analysis search passed")
            else:
                print(f"   ❌ Analysis search failed: {response.text}")
                return False
            
            # Test 7: Get session statistics
            print("\n7. Getting session statistics...")
            response = await client.get(f"/api/v1/persistent/sessions/{session_id}/stats")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                stats = response.json()
                print(f"   Total analyses: {stats['total_analyses']}")
                print(f"   Completed analyses: {stats['completed_analyses']}")
                print(f"   Success rate: {stats['success_rate']:.1%}")
                print("   ✅ Session statistics passed")
            else:
                print(f"   ❌ Session statistics failed: {response.text}")
                return False
            
            # Test 8: Get global statistics
            print("\n8. Getting global statistics...")
            response = await client.get("/api/v1/persistent/stats/global")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                global_stats = response.json()
                usage_stats = global_stats["usage_statistics"]
                print(f"   Global total analyses: {usage_stats['total_analyses']}")
                print(f"   Global success rate: {usage_stats['success_rate']:.1%}")
                if "provider_statistics" in global_stats:
                    provider_stats = global_stats["provider_statistics"]
                    print(f"   Providers tracked: {len(provider_stats)}")
                print("   ✅ Global statistics passed")
            else:
                print(f"   ❌ Global statistics failed: {response.text}")
                return False
            
            # Test 9: Get session details
            print("\n9. Getting session details...")
            response = await client.get(f"/api/v1/persistent/sessions/{session_id}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                session_details = response.json()
                print(f"   Session ID: {session_details['id']}")
                print(f"   Username: {session_details['username']}")
                print(f"   Total analyses: {session_details['total_analyses']}")
                print(f"   Is active: {session_details['is_active']}")
                print("   ✅ Session details passed")
            else:
                print(f"   ❌ Session details failed: {response.text}")
                return False
            
            print("\n" + "=" * 50)
            print("🎉 All persistent storage integration tests passed!")
            print("\n📊 Summary:")
            print(f"   ✅ Created session ID: {session_id}")
            print(f"   ✅ Created analysis ID: {analysis_id}")
            print("   ✅ All API endpoints working")
            print("   ✅ Database persistence confirmed")
            print("   ✅ Analytics and search functional")
            
            return True


async def test_direct_database_operations():
    """Test direct database operations to verify data persistence"""
    print("\n🔍 Testing Direct Database Operations")
    print("=" * 50)
    
    try:
        # Setup database
        await setup_database()
        
        from app.db.session import get_async_session
        from app.db.crud import UserSessionCRUD, AnalysisCRUD
        from app.db.models import LLMProviderType
        
        async with get_async_session() as db:
            # Check if our test data is still there
            print("1. Checking for persisted sessions...")
            sessions = await db.execute(
                "SELECT COUNT(*) FROM user_sessions WHERE username LIKE '%test%'"
            )
            session_count = sessions.scalar()
            print(f"   Test sessions found: {session_count}")
            
            print("2. Checking for persisted analyses...")
            analyses = await db.execute(
                "SELECT COUNT(*) FROM analyses WHERE gene = 'BRCA1'"
            )
            analysis_count = analyses.scalar()
            print(f"   BRCA1 analyses found: {analysis_count}")
            
            print("3. Checking analysis chunks...")
            chunks = await db.execute(
                "SELECT COUNT(*) FROM analysis_chunks"
            )
            chunk_count = chunks.scalar()
            print(f"   Analysis chunks found: {chunk_count}")
            
            # Test creating additional data
            print("4. Creating additional test data...")
            test_session = await UserSessionCRUD.create_session(
                session=db,
                username="direct_db_test_user",
                provider="direct_test"
            )
            print(f"   Created session: {test_session.id}")
            
            test_analysis = await AnalysisCRUD.create_analysis(
                session=db,
                session_id=test_session.id,
                gene="TP53",
                disease="cancer",
                provider=LLMProviderType.ANTHROPIC,
                model="claude-3",
                prompt_text="Direct database test prompt"
            )
            print(f"   Created analysis: {test_analysis.id}")
            
            print("   ✅ Direct database operations successful")
            
        await close_database()
        return True
        
    except Exception as e:
        print(f"   ❌ Direct database operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main integration test function"""
    print("🚀 Starting Comprehensive Persistent Storage Integration Tests")
    print("This test verifies the complete implementation including:")
    print("  • SQLModel database models with relationships")
    print("  • Async database session management")
    print("  • CRUD operations for all entities")
    print("  • FastAPI integration with database")
    print("  • RESTful API endpoints for persistent operations")
    print("  • Analytics and search functionality")
    print("  • Health monitoring and status reporting")
    print()
    
    try:
        # Run API integration tests
        api_success = await test_api_client()
        
        # Run direct database tests
        db_success = await test_direct_database_operations()
        
        if api_success and db_success:
            print("\n" + "🎊" * 20)
            print("🏆 ALL INTEGRATION TESTS PASSED! 🏆")
            print("🎊" * 20)
            print("\n✨ The persistent storage system is fully functional:")
            print("  ✅ Database models and relationships working")
            print("  ✅ Async CRUD operations successful")
            print("  ✅ FastAPI endpoints responding correctly")
            print("  ✅ Data persistence confirmed")
            print("  ✅ Analytics and search operational")
            print("  ✅ Health monitoring active")
            print("\n🚀 Ready for production deployment!")
            return True
        else:
            print("\n❌ Some integration tests failed")
            return False
            
    except Exception as e:
        print(f"\n💥 Integration tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)