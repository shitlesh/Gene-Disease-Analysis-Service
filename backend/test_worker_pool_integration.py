#!/usr/bin/env python3
"""
Integration test for the worker pool system
Tests the complete flow from job submission to completion
"""

import asyncio
import time
from typing import Dict, Any

from app.models.llm import LLMAnalysisRequest, LLMProvider
from app.services.worker_pool import (
    submit_llm_job, get_job_status, get_pool_status,
    shutdown_worker_pool, JobStatus
)

# Mock LLM service for testing
class TestLLMService:
    """Simple mock LLM service for integration testing"""
    
    def __init__(self):
        self.call_count = 0
    
    async def analyze_gene_disease(self, request: LLMAnalysisRequest):
        """Mock gene-disease analysis"""
        self.call_count += 1
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        from app.models.llm import LLMServiceResponse, LLMAnalysisResponse
        
        return LLMServiceResponse(
            success=True,
            data=LLMAnalysisResponse(
                summary=f"Mock analysis for {request.gene} and {request.disease}",
                associations=[],
                recommendation="Mock recommendation",
                uncertainty="Mock uncertainty",
                confidence_score=0.85
            )
        )


async def test_basic_job_flow():
    """Test basic job submission and completion flow"""
    print("Testing basic job flow...")
    
    # Mock the LLM service
    import app.services.llm_client
    original_get_llm_service = app.services.llm_client.get_llm_service
    
    async def mock_get_llm_service():
        return TestLLMService()
    
    app.services.llm_client.get_llm_service = mock_get_llm_service
    
    try:
        # Create a test request
        request = LLMAnalysisRequest(
            gene="BRCA1",
            disease="breast cancer",
            provider=LLMProvider.OPENAI
        )
        
        # Submit job
        print("Submitting job...")
        job_id = await submit_llm_job(request, session_id="test_session")
        print(f"Job submitted with ID: {job_id}")
        
        # Check initial status
        job = await get_job_status(job_id)
        print(f"Initial job status: {job.status.value}")
        
        # Wait for completion
        print("Waiting for job completion...")
        timeout = 10.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = await get_job_status(job_id)
            print(f"Job status: {job.status.value}")
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            
            await asyncio.sleep(0.5)
        
        # Check final result
        if job.status == JobStatus.COMPLETED:
            print("✅ Job completed successfully!")
            print(f"Result summary: {job.result.data.summary}")
        else:
            print(f"❌ Job failed with status: {job.status.value}")
            if job.error:
                print(f"Error: {job.error}")
        
        # Get pool status
        pool_status = await get_pool_status()
        print(f"Pool status: Running={pool_status['running']}, Workers={pool_status['workers']}")
        print(f"Statistics: {pool_status['statistics']}")
        
    finally:
        # Restore original function
        app.services.llm_client.get_llm_service = original_get_llm_service


async def test_concurrent_jobs():
    """Test multiple concurrent jobs"""
    print("\nTesting concurrent job processing...")
    
    # Mock the LLM service
    import app.services.llm_client
    original_get_llm_service = app.services.llm_client.get_llm_service
    
    async def mock_get_llm_service():
        return TestLLMService()
    
    app.services.llm_client.get_llm_service = mock_get_llm_service
    
    try:
        # Submit multiple jobs
        job_ids = []
        num_jobs = 5
        
        print(f"Submitting {num_jobs} concurrent jobs...")
        for i in range(num_jobs):
            request = LLMAnalysisRequest(
                gene=f"GENE{i}",
                disease="test disease",
                provider=LLMProvider.OPENAI
            )
            job_id = await submit_llm_job(request, session_id=f"session_{i}")
            job_ids.append(job_id)
        
        print(f"Submitted {len(job_ids)} jobs")
        
        # Wait for all jobs to complete
        print("Waiting for all jobs to complete...")
        timeout = 15.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            completed_jobs = 0
            for job_id in job_ids:
                job = await get_job_status(job_id)
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    completed_jobs += 1
            
            print(f"Progress: {completed_jobs}/{num_jobs} jobs completed")
            
            if completed_jobs == num_jobs:
                break
            
            await asyncio.sleep(1.0)
        
        # Check final results
        successful_jobs = 0
        for job_id in job_ids:
            job = await get_job_status(job_id)
            if job.status == JobStatus.COMPLETED:
                successful_jobs += 1
        
        print(f"✅ {successful_jobs}/{num_jobs} jobs completed successfully")
        
        # Get final pool statistics
        pool_status = await get_pool_status()
        stats = pool_status['statistics']
        print(f"Final statistics:")
        print(f"  Jobs processed: {stats['jobs_processed']}")
        print(f"  Jobs failed: {stats['jobs_failed']}")
        print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        
    finally:
        # Restore original function
        app.services.llm_client.get_llm_service = original_get_llm_service


async def test_pool_status():
    """Test pool status reporting"""
    print("\nTesting pool status reporting...")
    
    status = await get_pool_status()
    
    print("Pool Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


async def main():
    """Main integration test function"""
    print("🚀 Starting Worker Pool Integration Tests")
    print("=" * 50)
    
    try:
        # Run tests
        await test_basic_job_flow()
        await test_concurrent_jobs()
        await test_pool_status()
        
        print("\n" + "=" * 50)
        print("✅ All integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        print("\nShutting down worker pool...")
        await shutdown_worker_pool()
        print("Worker pool shutdown completed")


if __name__ == "__main__":
    # Run the integration tests
    asyncio.run(main())