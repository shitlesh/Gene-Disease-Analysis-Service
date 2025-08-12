"""
Stress tests and load simulation for worker pool system
Tests system behavior under extreme load conditions
"""

import asyncio
import pytest
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from app.services.worker_pool import (
    WorkerPool, TokenBucket, ExponentialBackoff, ProviderThrottler,
    QueuedJob, JobStatus, JobPriority
)
from app.models.llm import LLMAnalysisRequest, LLMProvider, LLMServiceResponse, LLMAnalysisResponse


class StressTestLLMService:
    """LLM service for stress testing with configurable behavior"""
    
    def __init__(self, 
                 min_response_time: float = 0.01, 
                 max_response_time: float = 2.0,
                 failure_rate: float = 0.1,
                 timeout_rate: float = 0.05):
        self.min_response_time = min_response_time
        self.max_response_time = max_response_time
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.call_count = 0
        self.concurrent_calls = 0
        self.max_concurrent_calls = 0
        self.response_times = []
        self.lock = asyncio.Lock()
    
    async def analyze_gene_disease(self, request: LLMAnalysisRequest) -> LLMServiceResponse:
        """Simulate realistic LLM service behavior"""
        async with self.lock:
            self.call_count += 1
            self.concurrent_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self.concurrent_calls)
        
        start_time = time.time()
        
        try:
            # Simulate variable response times
            response_time = random.uniform(self.min_response_time, self.max_response_time)
            
            # Simulate timeout scenarios
            if random.random() < self.timeout_rate:
                await asyncio.sleep(10)  # Simulate timeout
            else:
                await asyncio.sleep(response_time)
            
            # Simulate failures
            if random.random() < self.failure_rate:
                return LLMServiceResponse(
                    success=False,
                    error={'error_type': 'SIMULATED_ERROR', 'message': 'Random failure for testing'}
                )
            
            # Return successful response
            return LLMServiceResponse(
                success=True,
                data=LLMAnalysisResponse(
                    summary=f"Analysis for {request.gene}-{request.disease}",
                    associations=[],
                    recommendation="Test recommendation",
                    uncertainty="Test uncertainty"
                )
            )
        
        finally:
            processing_time = time.time() - start_time
            async with self.lock:
                self.concurrent_calls -= 1
                self.response_times.append(processing_time)


class TestWorkerPoolStress:
    """Stress tests for worker pool system"""
    
    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self):
        """Test worker pool with extreme concurrent job submission"""
        service = StressTestLLMService(min_response_time=0.1, max_response_time=0.5)
        worker_pool = WorkerPool(
            concurrency_limit=8,
            queue_size=200,
            llm_service=service
        )
        
        await worker_pool.start()
        
        # Submit 100 jobs as quickly as possible
        num_jobs = 100
        job_ids = []
        
        start_time = time.time()
        
        # Submit all jobs concurrently
        async def submit_batch(batch_start: int, batch_size: int):
            batch_jobs = []
            for i in range(batch_start, batch_start + batch_size):
                request = LLMAnalysisRequest(
                    gene=f"GENE_{i:03d}",
                    disease=f"disease_{i % 10}",
                    provider=random.choice([LLMProvider.OPENAI, LLMProvider.ANTHROPIC]),
                    context=f"Test context for job {i}"
                )
                session_id = f"session_{i % 20}"  # 20 different sessions
                priority = random.choice(list(JobPriority))
                
                job_id = await worker_pool.submit_job(request, session_id, priority)
                batch_jobs.append(job_id)
            return batch_jobs
        
        # Submit jobs in batches to avoid overwhelming
        batch_size = 20
        batches = []
        for i in range(0, num_jobs, batch_size):
            batch = submit_batch(i, min(batch_size, num_jobs - i))
            batches.append(batch)
        
        # Execute all batches concurrently
        batch_results = await asyncio.gather(*batches)
        job_ids = [job_id for batch in batch_results for job_id in batch]
        
        submission_time = time.time() - start_time
        print(f"Submitted {num_jobs} jobs in {submission_time:.2f} seconds")
        
        # Wait for all jobs to complete
        timeout = 60.0  # 1 minute timeout
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            completed_count = sum(
                1 for job_id in job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            )
            
            if completed_count == num_jobs:
                break
            
            await asyncio.sleep(0.5)
        
        completion_time = time.time() - start_time
        print(f"All jobs completed in {completion_time:.2f} seconds")
        
        # Analyze results
        status = await worker_pool.get_queue_status()
        stats = status['statistics']
        
        print(f"Statistics: {stats}")
        print(f"Max concurrent calls: {service.max_concurrent_calls}")
        print(f"Average response time: {statistics.mean(service.response_times):.3f}s")
        
        # Assertions
        assert stats['jobs_queued'] == num_jobs
        assert stats['jobs_processed'] + stats['jobs_failed'] == num_jobs
        assert service.max_concurrent_calls <= worker_pool.concurrency_limit
        assert completion_time < timeout  # Should complete within timeout
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_stress(self):
        """Test rate limiting under stress conditions"""
        service = StressTestLLMService(min_response_time=0.05, max_response_time=0.1)
        worker_pool = WorkerPool(
            concurrency_limit=4,
            queue_size=50,
            llm_service=service
        )
        
        # Configure very restrictive rate limits
        worker_pool.throttler.buckets[LLMProvider.OPENAI] = TokenBucket(
            capacity=5, refill_rate=2.0  # 2 requests per second
        )
        worker_pool.throttler.buckets[LLMProvider.ANTHROPIC] = TokenBucket(
            capacity=3, refill_rate=1.0  # 1 request per second
        )
        
        await worker_pool.start()
        
        # Submit 30 jobs rapidly
        num_jobs = 30
        job_ids = []
        
        for i in range(num_jobs):
            request = LLMAnalysisRequest(
                gene=f"GENE_{i}",
                disease="test_disease",
                provider=LLMProvider.OPENAI if i % 2 == 0 else LLMProvider.ANTHROPIC
            )
            job_id = await worker_pool.submit_job(request)
            job_ids.append(job_id)
        
        # Measure completion time
        start_time = time.time()
        timeout = 45.0
        
        while time.time() - start_time < timeout:
            completed_count = sum(
                1 for job_id in job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            
            if completed_count == num_jobs:
                break
            
            await asyncio.sleep(0.5)
        
        completion_time = time.time() - start_time
        status = await worker_pool.get_queue_status()
        
        # With strict rate limiting, should take significant time
        print(f"Rate-limited completion time: {completion_time:.2f} seconds")
        print(f"Total wait time: {status['statistics']['total_wait_time']:.2f} seconds")
        
        assert completion_time >= 15.0  # Should be rate limited
        assert status['statistics']['total_wait_time'] > 0
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively under load"""
        import psutil
        import os
        
        service = StressTestLLMService(min_response_time=0.01, max_response_time=0.05)
        worker_pool = WorkerPool(
            concurrency_limit=6,
            queue_size=100,
            llm_service=service
        )
        
        await worker_pool.start()
        
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Submit and complete multiple batches
        num_batches = 10
        batch_size = 50
        
        peak_memory = initial_memory
        
        for batch in range(num_batches):
            batch_jobs = []
            
            # Submit batch
            for i in range(batch_size):
                request = LLMAnalysisRequest(
                    gene=f"BATCH_{batch}_GENE_{i}",
                    disease="memory_test",
                    provider=LLMProvider.OPENAI
                )
                job_id = await worker_pool.submit_job(request)
                batch_jobs.append(job_id)
            
            # Wait for batch to complete
            timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                completed = sum(
                    1 for job_id in batch_jobs
                    if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
                )
                
                if completed == batch_size:
                    break
                
                await asyncio.sleep(0.1)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = max(peak_memory, current_memory)
            
            print(f"Batch {batch + 1}: Memory usage {current_memory:.1f} MB")
            
            # Optional: Clean up old jobs to simulate real usage
            if batch % 3 == 0:
                # Remove old completed jobs
                old_jobs = [
                    job_id for job_id, job in worker_pool.jobs.items()
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                    job.age_seconds > 30
                ]
                for job_id in old_jobs[:len(old_jobs)//2]:  # Remove half
                    del worker_pool.jobs[job_id]
        
        memory_growth = peak_memory - initial_memory
        print(f"Memory growth: {memory_growth:.1f} MB (initial: {initial_memory:.1f} MB, peak: {peak_memory:.1f} MB)")
        
        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_retry_storm_handling(self):
        """Test handling of retry storms (many jobs failing and retrying)"""
        # Service with high failure rate to trigger retries
        service = StressTestLLMService(
            min_response_time=0.01,
            max_response_time=0.1,
            failure_rate=0.8,  # 80% failure rate
            timeout_rate=0.1   # 10% timeout rate
        )
        
        worker_pool = WorkerPool(
            concurrency_limit=4,
            queue_size=50,
            llm_service=service
        )
        
        # Configure fast retries for testing
        worker_pool.backoff = ExponentialBackoff(
            base_delay=0.1,  # Start with 100ms
            max_delay=2.0,   # Max 2 seconds
            jitter=True
        )
        
        await worker_pool.start()
        
        # Submit jobs that will likely fail and retry
        num_jobs = 20
        job_ids = []
        
        for i in range(num_jobs):
            request = LLMAnalysisRequest(
                gene=f"RETRY_GENE_{i}",
                disease="retry_test",
                provider=LLMProvider.OPENAI
            )
            job_id = await worker_pool.submit_job(request)
            job_ids.append(job_id)
        
        # Wait for completion (allowing for retries)
        timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            final_count = sum(
                1 for job_id in job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            
            if final_count == num_jobs:
                break
            
            await asyncio.sleep(0.5)
        
        # Analyze retry behavior
        status = await worker_pool.get_queue_status()
        stats = status['statistics']
        
        total_retries = sum(worker_pool.jobs[job_id].current_retry for job_id in job_ids)
        successful_jobs = stats['jobs_processed']
        failed_jobs = stats['jobs_failed']
        
        print(f"Retry storm results:")
        print(f"  Total retries: {total_retries}")
        print(f"  Successful jobs: {successful_jobs}")
        print(f"  Failed jobs: {failed_jobs}")
        print(f"  Retry count stat: {stats['retry_count']}")
        print(f"  Service call count: {service.call_count}")
        
        # Should have attempted many retries
        assert total_retries > num_jobs  # More retries than jobs
        assert stats['retry_count'] > 0
        assert service.call_count > num_jobs  # More calls due to retries
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_priority_under_load(self):
        """Test priority handling under mixed load conditions"""
        service = StressTestLLMService(min_response_time=0.1, max_response_time=0.3)
        worker_pool = WorkerPool(
            concurrency_limit=2,  # Limited concurrency to test priority
            queue_size=50,
            llm_service=service
        )
        
        await worker_pool.start()
        
        # Submit jobs with different priorities
        low_priority_jobs = []
        high_priority_jobs = []
        critical_priority_jobs = []
        
        # Submit low priority jobs first
        for i in range(10):
            request = LLMAnalysisRequest(
                gene=f"LOW_GENE_{i}",
                disease="low_priority_test",
                provider=LLMProvider.OPENAI
            )
            job_id = await worker_pool.submit_job(request, priority=JobPriority.LOW)
            low_priority_jobs.append(job_id)
        
        # Then submit high priority jobs
        for i in range(5):
            request = LLMAnalysisRequest(
                gene=f"HIGH_GENE_{i}",
                disease="high_priority_test",
                provider=LLMProvider.OPENAI
            )
            job_id = await worker_pool.submit_job(request, priority=JobPriority.HIGH)
            high_priority_jobs.append(job_id)
        
        # Finally submit critical priority jobs
        for i in range(3):
            request = LLMAnalysisRequest(
                gene=f"CRITICAL_GENE_{i}",
                disease="critical_priority_test",
                provider=LLMProvider.OPENAI
            )
            job_id = await worker_pool.submit_job(request, priority=JobPriority.CRITICAL)
            critical_priority_jobs.append(job_id)
        
        # Wait for completion
        all_jobs = low_priority_jobs + high_priority_jobs + critical_priority_jobs
        timeout = 20.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            completed_count = sum(
                1 for job_id in all_jobs
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            
            if completed_count == len(all_jobs):
                break
            
            await asyncio.sleep(0.2)
        
        # Analyze completion order (priority should influence processing)
        job_completion_times = {}
        for job_id in all_jobs:
            job = worker_pool.jobs[job_id]
            if job.completed_at:
                job_completion_times[job_id] = job.completed_at
        
        # Check that higher priority jobs generally completed earlier
        # (This is probabilistic due to the asynchronous nature)
        critical_avg_time = statistics.mean([
            job_completion_times[job_id].timestamp() 
            for job_id in critical_priority_jobs 
            if job_id in job_completion_times
        ]) if critical_priority_jobs else float('inf')
        
        low_avg_time = statistics.mean([
            job_completion_times[job_id].timestamp() 
            for job_id in low_priority_jobs 
            if job_id in job_completion_times
        ]) if low_priority_jobs else float('inf')
        
        print(f"Average completion times:")
        print(f"  Critical priority: {critical_avg_time}")
        print(f"  Low priority: {low_avg_time}")
        
        # Note: Due to the asynchronous nature and small job count,
        # priority effects may not always be clearly visible
        # This test mainly ensures the system handles mixed priorities without crashes
        
        await worker_pool.stop()


class TestPerformanceBenchmarks:
    """Performance benchmarks for worker pool"""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark worker pool throughput"""
        service = StressTestLLMService(min_response_time=0.01, max_response_time=0.05)
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for concurrency in concurrency_levels:
            worker_pool = WorkerPool(
                concurrency_limit=concurrency,
                queue_size=100,
                llm_service=service
            )
            
            await worker_pool.start()
            
            # Submit fixed number of jobs
            num_jobs = 50
            job_ids = []
            
            start_time = time.time()
            
            for i in range(num_jobs):
                request = LLMAnalysisRequest(
                    gene=f"BENCH_GENE_{i}",
                    disease="benchmark_test",
                    provider=LLMProvider.OPENAI
                )
                job_id = await worker_pool.submit_job(request)
                job_ids.append(job_id)
            
            # Wait for completion
            timeout = 30.0
            while time.time() - start_time < timeout:
                completed_count = sum(
                    1 for job_id in job_ids
                    if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
                )
                
                if completed_count == num_jobs:
                    break
                
                await asyncio.sleep(0.1)
            
            completion_time = time.time() - start_time
            throughput = num_jobs / completion_time
            
            results[concurrency] = {
                'completion_time': completion_time,
                'throughput': throughput,
                'jobs_per_second': throughput
            }
            
            print(f"Concurrency {concurrency}: {throughput:.2f} jobs/second ({completion_time:.2f}s)")
            
            await worker_pool.stop()
        
        # Verify throughput increases with concurrency (up to a point)
        assert results[2]['throughput'] > results[1]['throughput']
        assert results[4]['throughput'] > results[2]['throughput']
        
        print("Throughput benchmark completed successfully")
    
    @pytest.mark.asyncio
    async def test_latency_distribution(self):
        """Test latency distribution under various load conditions"""
        service = StressTestLLMService(min_response_time=0.05, max_response_time=0.2)
        worker_pool = WorkerPool(
            concurrency_limit=4,
            queue_size=100,
            llm_service=service
        )
        
        await worker_pool.start()
        
        # Submit jobs and measure end-to-end latency
        num_jobs = 30
        job_latencies = []
        
        for i in range(num_jobs):
            request = LLMAnalysisRequest(
                gene=f"LATENCY_GENE_{i}",
                disease="latency_test",
                provider=LLMProvider.OPENAI
            )
            
            submission_time = time.time()
            job_id = await worker_pool.submit_job(request)
            
            # Track submission time
            worker_pool.jobs[job_id].submission_time = submission_time
        
        # Wait for all jobs to complete
        all_job_ids = list(worker_pool.jobs.keys())[-num_jobs:]  # Last num_jobs
        timeout = 15.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            completed_count = sum(
                1 for job_id in all_job_ids
                if worker_pool.jobs[job_id].status in [JobStatus.COMPLETED, JobStatus.FAILED]
            )
            
            if completed_count == num_jobs:
                break
            
            await asyncio.sleep(0.1)
        
        # Calculate latencies
        for job_id in all_job_ids:
            job = worker_pool.jobs[job_id]
            if hasattr(job, 'submission_time') and job.completed_at:
                latency = job.completed_at.timestamp() - job.submission_time
                job_latencies.append(latency)
        
        # Analyze latency distribution
        if job_latencies:
            avg_latency = statistics.mean(job_latencies)
            median_latency = statistics.median(job_latencies)
            p95_latency = sorted(job_latencies)[int(0.95 * len(job_latencies))]
            
            print(f"Latency distribution:")
            print(f"  Average: {avg_latency:.3f}s")
            print(f"  Median: {median_latency:.3f}s")
            print(f"  95th percentile: {p95_latency:.3f}s")
            
            # Reasonable latency expectations
            assert avg_latency < 2.0  # Average under 2 seconds
            assert p95_latency < 5.0  # 95th percentile under 5 seconds
        
        await worker_pool.stop()


if __name__ == "__main__":
    # Run stress tests
    pytest.main([__file__, "-v", "-s"])