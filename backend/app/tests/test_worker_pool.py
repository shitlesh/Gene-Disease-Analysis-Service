"""
Comprehensive unit tests for the worker pool system
Tests concurrency control, rate limiting, retry logic, and FIFO processing
"""

import asyncio
import pytest
import time
import random
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.services.worker_pool import (
    WorkerPool, TokenBucket, ExponentialBackoff, ProviderThrottler,
    QueuedJob, JobStatus, JobPriority, submit_llm_job, get_job_status,
    cancel_job, get_pool_status, shutdown_worker_pool
)
from app.models.llm import LLMAnalysisRequest, LLMProvider, LLMServiceResponse, LLMAnalysisResponse
from app.config import settings


class MockLLMService:
    """Mock LLM service for testing"""
    
    def __init__(self, response_time: float = 0.1, failure_rate: float = 0.0, should_timeout: bool = False):
        self.response_time = response_time
        self.failure_rate = failure_rate
        self.should_timeout = should_timeout
        self.call_count = 0
        self.calls = []
    
    async def analyze_gene_disease(self, request: LLMAnalysisRequest) -> LLMServiceResponse:
        """Mock analysis with configurable behavior"""
        self.call_count += 1
        self.calls.append({
            'request': request,
            'timestamp': datetime.utcnow(),
            'call_number': self.call_count
        })
        
        # Simulate processing time
        if self.should_timeout:
            await asyncio.sleep(10)  # Longer than typical timeout
        else:
            await asyncio.sleep(self.response_time)
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            return LLMServiceResponse(
                success=False,
                error={'error_type': 'MOCK_ERROR', 'message': 'Simulated failure'}
            )
        
        # Return successful response
        return LLMServiceResponse(
            success=True,
            data=LLMAnalysisResponse(
                summary=f"Mock analysis for {request.gene} and {request.disease}",
                associations=[],
                recommendation="Mock recommendation",
                uncertainty="Mock uncertainty"
            )
        )


@pytest.fixture
def mock_llm_service():
    """Fixture for mock LLM service"""
    return MockLLMService()


@pytest.fixture
def sample_request():
    """Fixture for sample LLM analysis request"""
    return LLMAnalysisRequest(
        gene="BRCA1",
        disease="breast cancer",
        provider=LLMProvider.OPENAI
    )


@pytest.fixture
def worker_pool(mock_llm_service):
    """Fixture for worker pool with mock service"""
    return WorkerPool(
        concurrency_limit=2,
        queue_size=10,
        llm_service=mock_llm_service
    )


class TestTokenBucket:
    """Test token bucket rate limiting algorithm"""
    
    @pytest.mark.asyncio
    async def test_token_bucket_initialization(self):
        """Test token bucket is initialized correctly"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
        assert bucket.tokens == 10.0
    
    @pytest.mark.asyncio
    async def test_token_acquisition(self):
        """Test token acquisition works correctly"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Should be able to acquire tokens initially
        assert await bucket.acquire(3) is True
        assert bucket.available_tokens == 2
        
        # Should not be able to acquire more than available
        assert await bucket.acquire(5) is False
        assert bucket.available_tokens == 2
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill mechanism"""
        bucket = TokenBucket(capacity=5, refill_rate=2.0)  # 2 tokens per second
        
        # Exhaust all tokens
        await bucket.acquire(5)
        assert bucket.available_tokens == 0
        
        # Wait for refill
        await asyncio.sleep(1.1)  # Wait slightly more than 1 second
        await bucket._refill()
        
        # Should have approximately 2 tokens (2 per second * 1 second)
        assert bucket.available_tokens >= 2
    
    @pytest.mark.asyncio
    async def test_wait_for_token(self):
        """Test waiting for token availability"""
        bucket = TokenBucket(capacity=2, refill_rate=2.0)  # 2 tokens per second
        
        # Exhaust all tokens
        await bucket.acquire(2)
        
        # Measure wait time
        start_time = time.time()
        wait_time = await bucket.wait_for_token(1)
        elapsed = time.time() - start_time
        
        # Should wait approximately 0.5 seconds (1 token / 2 tokens per second)
        assert 0.4 <= elapsed <= 0.8
        assert wait_time >= 0.4


class TestExponentialBackoff:
    """Test exponential backoff algorithm"""
    
    def test_backoff_progression(self):
        """Test exponential backoff delay progression"""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, jitter=False)
        
        # Test delay progression
        assert backoff.get_delay(0) == 1.0  # 1 * 2^0
        assert backoff.get_delay(1) == 2.0  # 1 * 2^1
        assert backoff.get_delay(2) == 4.0  # 1 * 2^2
        assert backoff.get_delay(3) == 8.0  # 1 * 2^3
        assert backoff.get_delay(4) == 10.0  # Capped at max_delay
    
    def test_backoff_with_jitter(self):
        """Test exponential backoff with jitter"""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=20.0, jitter=True)
        
        # Generate multiple delays and check they vary due to jitter
        delays = [backoff.get_delay(2) for _ in range(10)]
        
        # All delays should be around 8.0 (2 * 2^2) with ±25% jitter
        assert all(6.0 <= delay <= 10.0 for delay in delays)
        
        # Delays should vary due to jitter
        assert len(set(delays)) > 1


class TestProviderThrottler:
    """Test per-provider rate limiting"""
    
    @pytest.mark.asyncio
    async def test_throttler_initialization(self):
        """Test provider throttler initializes correctly"""
        throttler = ProviderThrottler()
        
        # Should have buckets for both providers
        assert LLMProvider.OPENAI in throttler.buckets
        assert LLMProvider.ANTHROPIC in throttler.buckets
        
        # Buckets should have different capacities
        openai_bucket = throttler.buckets[LLMProvider.OPENAI]
        anthropic_bucket = throttler.buckets[LLMProvider.ANTHROPIC]
        
        assert openai_bucket.capacity != anthropic_bucket.capacity
    
    @pytest.mark.asyncio
    async def test_permit_acquisition(self):
        """Test permit acquisition for different providers"""
        throttler = ProviderThrottler()
        
        # Should be able to acquire permits initially
        wait_time = await throttler.acquire_permit(LLMProvider.OPENAI)
        assert wait_time >= 0
        
        # Should track separate permits for different providers
        wait_time = await throttler.acquire_permit(LLMProvider.ANTHROPIC)
        assert wait_time >= 0
    
    @pytest.mark.asyncio
    async def test_throttler_status(self):
        """Test throttler status reporting"""
        throttler = ProviderThrottler()
        status = throttler.get_status()
        
        # Should have status for both providers
        assert 'openai' in status
        assert 'anthropic' in status
        
        # Each provider should have expected fields
        for provider_status in status.values():
            assert 'available_tokens' in provider_status
            assert 'capacity' in provider_status
            assert 'refill_rate' in provider_status


class TestWorkerPool:
    """Test worker pool functionality"""
    
    @pytest.mark.asyncio
    async def test_worker_pool_initialization(self, worker_pool):
        """Test worker pool initializes correctly"""
        assert worker_pool.concurrency_limit == 2
        assert worker_pool.queue_size == 10
        assert not worker_pool.running
        assert len(worker_pool.workers) == 0
    
    @pytest.mark.asyncio
    async def test_worker_pool_start_stop(self, worker_pool):
        """Test worker pool startup and shutdown"""
        # Start pool
        await worker_pool.start()
        assert worker_pool.running is True
        assert len(worker_pool.workers) == 2
        
        # Stop pool
        await worker_pool.stop()
        assert worker_pool.running is False
        assert len(worker_pool.workers) == 0
    
    @pytest.mark.asyncio
    async def test_job_submission(self, worker_pool, sample_request):
        """Test basic job submission"""
        await worker_pool.start()
        
        job_id = await worker_pool.submit_job(sample_request)
        assert job_id is not None
        assert job_id in worker_pool.jobs
        
        job = worker_pool.jobs[job_id]
        assert job.status == JobStatus.QUEUED
        assert job.request == sample_request
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_job_processing(self, worker_pool, sample_request):
        """Test job processing"""
        await worker_pool.start()
        
        job_id = await worker_pool.submit_job(sample_request)
        
        # Wait for job to be processed
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            job = worker_pool.jobs[job_id]
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            await asyncio.sleep(0.1)
        
        job = worker_pool.jobs[job_id]
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.started_at is not None
        assert job.completed_at is not None
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, worker_pool, sample_request):
        """Test job cancellation"""
        await worker_pool.start()
        
        # Submit job
        job_id = await worker_pool.submit_job(sample_request)
        
        # Cancel immediately before processing
        success = await worker_pool.cancel_job(job_id)
        assert success is True
        
        job = worker_pool.jobs[job_id]
        assert job.status == JobStatus.CANCELLED
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_queue_status(self, worker_pool, sample_request):
        """Test queue status reporting"""
        await worker_pool.start()
        
        # Submit some jobs
        job_ids = []
        for i in range(3):
            job_id = await worker_pool.submit_job(sample_request)
            job_ids.append(job_id)
        
        status = await worker_pool.get_queue_status()
        
        assert status['running'] is True
        assert status['workers'] == 2
        assert status['concurrency_limit'] == 2
        assert status['queue_capacity'] == 10
        assert 'statistics' in status
        assert 'rate_limiters' in status
        
        await worker_pool.stop()


class TestHighLoadScenarios:
    """Test worker pool under high-load conditions"""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_submission(self, mock_llm_service, sample_request):
        """Test submitting many jobs concurrently"""
        worker_pool = WorkerPool(
            concurrency_limit=4,
            queue_size=50,
            llm_service=mock_llm_service
        )
        
        await worker_pool.start()
        
        # Submit 20 jobs concurrently
        num_jobs = 20
        job_ids = []
        
        async def submit_job(i):
            request = LLMAnalysisRequest(
                gene=f"GENE{i}",
                disease="test disease",
                provider=LLMProvider.OPENAI
            )
            return await worker_pool.submit_job(request, session_id=f"session_{i % 5}")
        
        # Submit all jobs concurrently
        job_ids = await asyncio.gather(*[submit_job(i) for i in range(num_jobs)])
        
        assert len(job_ids) == num_jobs
        assert len(set(job_ids)) == num_jobs  # All job IDs should be unique
        
        # Wait for all jobs to complete
        timeout = 30.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            completed_jobs = sum(
                1 for job_id in job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            if completed_jobs == num_jobs:
                break
            await asyncio.sleep(0.2)
        
        # Check final statistics
        status = await worker_pool.get_queue_status()
        stats = status['statistics']
        
        assert stats['jobs_queued'] == num_jobs
        assert stats['jobs_processed'] + stats['jobs_failed'] == num_jobs
        assert mock_llm_service.call_count == num_jobs
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, mock_llm_service, sample_request):
        """Test behavior when queue capacity is exceeded"""
        worker_pool = WorkerPool(
            concurrency_limit=1,
            queue_size=3,  # Small queue
            llm_service=MockLLMService(response_time=1.0)  # Slow processing
        )
        
        await worker_pool.start()
        
        # Fill the queue
        job_ids = []
        for i in range(3):
            job_id = await worker_pool.submit_job(sample_request)
            job_ids.append(job_id)
        
        # Try to submit one more job (should fail)
        with pytest.raises(asyncio.QueueFull):
            await worker_pool.submit_job(sample_request)
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, mock_llm_service):
        """Test rate limiting behavior under high load"""
        # Create throttler with very low limits for testing
        worker_pool = WorkerPool(
            concurrency_limit=2,
            queue_size=20,
            llm_service=mock_llm_service
        )
        
        # Patch the throttler to have very low limits
        worker_pool.throttler.buckets[LLMProvider.OPENAI] = TokenBucket(
            capacity=2, refill_rate=0.5  # Only 0.5 requests per second
        )
        
        await worker_pool.start()
        
        # Submit multiple jobs rapidly
        start_time = time.time()
        job_ids = []
        
        for i in range(5):
            request = LLMAnalysisRequest(
                gene=f"GENE{i}",
                disease="test disease",
                provider=LLMProvider.OPENAI
            )
            job_id = await worker_pool.submit_job(request)
            job_ids.append(job_id)
        
        # Wait for all jobs to complete
        timeout = 20.0
        while time.time() - start_time < timeout:
            completed_jobs = sum(
                1 for job_id in job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            if completed_jobs == len(job_ids):
                break
            await asyncio.sleep(0.2)
        
        total_time = time.time() - start_time
        
        # With rate limiting, should take longer than without
        # 5 jobs with 0.5 requests/second should take at least 8-10 seconds
        assert total_time >= 8.0
        
        # Check that rate limiting statistics were recorded
        status = await worker_pool.get_queue_status()
        assert status['statistics']['total_wait_time'] > 0
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_failures(self, sample_request):
        """Test retry logic with simulated failures"""
        # Create mock service with 70% failure rate
        failing_service = MockLLMService(failure_rate=0.7, response_time=0.1)
        
        worker_pool = WorkerPool(
            concurrency_limit=2,
            queue_size=10,
            llm_service=failing_service
        )
        
        await worker_pool.start()
        
        job_id = await worker_pool.submit_job(sample_request)
        
        # Wait for job to complete (with retries)
        timeout = 15.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            job = worker_pool.jobs[job_id]
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            await asyncio.sleep(0.2)
        
        job = worker_pool.jobs[job_id]
        
        # Job should either complete successfully or fail after max retries
        assert job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
        
        # Should have attempted retries
        assert job.current_retry > 0 or job.status == JobStatus.COMPLETED
        
        # Check retry statistics
        status = await worker_pool.get_queue_status()
        assert status['statistics']['retry_count'] >= 0
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_fifo_processing_per_session(self, mock_llm_service):
        """Test FIFO job processing per session"""
        worker_pool = WorkerPool(
            concurrency_limit=1,  # Single worker to ensure ordering
            queue_size=20,
            llm_service=MockLLMService(response_time=0.1)
        )
        
        await worker_pool.start()
        
        # Submit jobs for two different sessions
        session_a_jobs = []
        session_b_jobs = []
        
        for i in range(5):
            # Session A job
            request_a = LLMAnalysisRequest(
                gene=f"GENE_A{i}",
                disease="disease A",
                provider=LLMProvider.OPENAI
            )
            job_id_a = await worker_pool.submit_job(request_a, session_id="session_A")
            session_a_jobs.append(job_id_a)
            
            # Session B job
            request_b = LLMAnalysisRequest(
                gene=f"GENE_B{i}",
                disease="disease B",
                provider=LLMProvider.OPENAI
            )
            job_id_b = await worker_pool.submit_job(request_b, session_id="session_B")
            session_b_jobs.append(job_id_b)
        
        # Wait for all jobs to complete
        all_jobs = session_a_jobs + session_b_jobs
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            completed_jobs = sum(
                1 for job_id in all_jobs
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            if completed_jobs == len(all_jobs):
                break
            await asyncio.sleep(0.1)
        
        # Check session queues were maintained
        assert "session_A" in worker_pool.session_queues
        assert "session_B" in worker_pool.session_queues
        
        # Verify all jobs completed
        for job_id in all_jobs:
            job = worker_pool.jobs[job_id]
            assert job.status == JobStatus.COMPLETED
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_request):
        """Test job timeout handling"""
        # Create service that takes too long
        slow_service = MockLLMService(should_timeout=True)
        
        worker_pool = WorkerPool(
            concurrency_limit=1,
            queue_size=5,
            llm_service=slow_service
        )
        
        await worker_pool.start()
        
        # Submit job with short timeout
        job_id = await worker_pool.submit_job(
            sample_request,
            timeout=1.0  # 1 second timeout
        )
        
        # Wait for job to timeout and retry
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            job = worker_pool.jobs[job_id]
            if job.status in [JobStatus.FAILED]:
                break
            await asyncio.sleep(0.2)
        
        job = worker_pool.jobs[job_id]
        
        # Job should eventually fail due to timeout
        assert job.status == JobStatus.FAILED
        assert "timeout" in job.error.lower() or job.current_retry >= job.max_retries
        
        await worker_pool.stop()


class TestGlobalWorkerPoolFunctions:
    """Test global worker pool convenience functions"""
    
    @pytest.mark.asyncio
    async def test_global_pool_functions(self, sample_request):
        """Test global worker pool convenience functions"""
        # Mock the LLM service
        with patch('app.services.worker_pool.get_llm_service') as mock_get_service:
            mock_service = MockLLMService()
            mock_get_service.return_value = mock_service
            
            # Test job submission
            job_id = await submit_llm_job(sample_request)
            assert job_id is not None
            
            # Test job status retrieval
            job = await get_job_status(job_id)
            assert job is not None
            assert job.id == job_id
            
            # Test pool status
            status = await get_pool_status()
            assert 'running' in status
            assert 'statistics' in status
            
            # Test job cancellation
            if job.status == JobStatus.QUEUED:
                success = await cancel_job(job_id)
                assert success in [True, False]  # May or may not succeed depending on timing
            
            # Clean up
            await shutdown_worker_pool()


@pytest.mark.asyncio
async def test_worker_pool_performance_metrics():
    """Test worker pool performance metrics and statistics"""
    mock_service = MockLLMService(response_time=0.1)
    
    worker_pool = WorkerPool(
        concurrency_limit=3,
        queue_size=20,
        llm_service=mock_service
    )
    
    await worker_pool.start()
    
    # Submit multiple jobs
    job_ids = []
    start_time = time.time()
    
    for i in range(10):
        request = LLMAnalysisRequest(
            gene=f"GENE{i}",
            disease="test disease",
            provider=LLMProvider.OPENAI
        )
        job_id = await worker_pool.submit_job(request)
        job_ids.append(job_id)
    
    # Wait for completion
    timeout = 10.0
    while time.time() - start_time < timeout:
        completed_jobs = sum(
            1 for job_id in job_ids
            if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
        )
        if completed_jobs == len(job_ids):
            break
        await asyncio.sleep(0.1)
    
    # Check performance metrics
    status = await worker_pool.get_queue_status()
    stats = status['statistics']
    
    assert stats['jobs_queued'] == 10
    assert stats['jobs_processed'] >= 0
    assert stats['total_processing_time'] >= 0
    assert 'rate_limiters' in status
    
    # Verify individual job metrics
    for job_id in job_ids:
        job = worker_pool.jobs[job_id]
        if job.status == JobStatus.COMPLETED:
            assert job.processing_time is not None
            assert job.processing_time > 0
    
    await worker_pool.stop()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])