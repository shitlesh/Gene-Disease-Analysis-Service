"""
Asynchronous worker pool for LLM requests with concurrency control and rate limiting
Implements token bucket algorithm for per-provider throttling and FIFO queue management
"""

import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ..models.llm import LLMAnalysisRequest, LLMServiceResponse, LLMProvider
from ..config import settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the worker pool"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class QueuedJob:
    """Represents a job in the worker pool queue"""
    id: str
    request: LLMAnalysisRequest
    session_id: Optional[str]
    priority: JobPriority
    created_at: datetime
    callback: Optional[Callable[[LLMServiceResponse], Awaitable[None]]] = None
    timeout: float = 300.0  # 5 minutes default
    max_retries: int = 3
    current_retry: int = 0
    status: JobStatus = JobStatus.QUEUED
    result: Optional[LLMServiceResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def age_seconds(self) -> float:
        """Get age of job in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds if job has started"""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed: int = 1) -> bool:
        """
        Try to acquire tokens from bucket
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            await self._refill()
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False
    
    async def wait_for_token(self, tokens_needed: int = 1) -> float:
        """
        Wait until tokens are available and acquire them
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            Time waited in seconds
        """
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens_needed):
                return time.time() - start_time
            
            # Calculate wait time until enough tokens are available
            wait_time = min(1.0, tokens_needed / self.refill_rate)
            await asyncio.sleep(wait_time)
    
    async def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
    
    @property
    def available_tokens(self) -> int:
        """Get current number of available tokens"""
        return int(self.tokens)


class ExponentialBackoff:
    """Exponential backoff with jitter for retry logic"""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)
        
        return delay


class ProviderThrottler:
    """Per-provider rate limiting using token bucket algorithm"""
    
    def __init__(self):
        self.buckets: Dict[LLMProvider, TokenBucket] = {}
        self._initialize_buckets()
    
    def _initialize_buckets(self):
        """Initialize token buckets for each provider"""
        # OpenAI rate limits (conservative estimates)
        self.buckets[LLMProvider.OPENAI] = TokenBucket(
            capacity=getattr(settings, 'OPENAI_RATE_LIMIT_CAPACITY', 60),
            refill_rate=getattr(settings, 'OPENAI_RATE_LIMIT_REFILL', 1.0)  # 1 request/second
        )
        
        # Anthropic rate limits (conservative estimates)
        self.buckets[LLMProvider.ANTHROPIC] = TokenBucket(
            capacity=getattr(settings, 'ANTHROPIC_RATE_LIMIT_CAPACITY', 50),
            refill_rate=getattr(settings, 'ANTHROPIC_RATE_LIMIT_REFILL', 0.8)  # 0.8 requests/second
        )
    
    async def acquire_permit(self, provider: LLMProvider, wait: bool = True) -> float:
        """
        Acquire permit for provider request
        
        Args:
            provider: LLM provider
            wait: Whether to wait for permit if not immediately available
            
        Returns:
            Time waited in seconds
        """
        if provider not in self.buckets:
            logger.warning(f"No rate limiter configured for provider {provider}")
            return 0.0
        
        bucket = self.buckets[provider]
        
        if wait:
            return await bucket.wait_for_token()
        else:
            acquired = await bucket.acquire()
            return 0.0 if acquired else -1.0  # -1 indicates no permit available
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all rate limiters"""
        status = {}
        for provider, bucket in self.buckets.items():
            status[provider.value] = {
                "available_tokens": bucket.available_tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate
            }
        return status


class WorkerPool:
    """Async worker pool for processing LLM requests with concurrency control"""
    
    def __init__(
        self,
        concurrency_limit: int = None,
        queue_size: int = None,
        llm_service = None
    ):
        self.concurrency_limit = concurrency_limit or getattr(settings, 'LLM_CONCURRENCY', 4)
        self.queue_size = queue_size or getattr(settings, 'LLM_QUEUE_SIZE', 100)
        self.llm_service = llm_service
        
        # Queue management
        self.job_queue: asyncio.Queue[QueuedJob] = asyncio.Queue(maxsize=self.queue_size)
        self.jobs: Dict[str, QueuedJob] = {}
        self.session_queues: Dict[str, List[str]] = {}  # FIFO per session
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.throttler = ProviderThrottler()
        self.backoff = ExponentialBackoff(
            base_delay=getattr(settings, 'LLM_RETRY_BASE_DELAY', 1.0),
            max_delay=getattr(settings, 'LLM_RETRY_MAX_DELAY', 60.0)
        )
        
        # Statistics
        self.stats = {
            'jobs_queued': 0,
            'jobs_processed': 0,
            'jobs_failed': 0,
            'jobs_cancelled': 0,
            'total_wait_time': 0.0,
            'total_processing_time': 0.0,
            'retry_count': 0
        }
        
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the worker pool"""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting worker pool with {self.concurrency_limit} workers")
        
        # Start worker tasks
        for i in range(self.concurrency_limit):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info("Worker pool started")
    
    async def stop(self, timeout: float = 30.0):
        """Stop the worker pool gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping worker pool...")
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish with timeout
        if self.workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Worker pool shutdown timed out")
        
        self.workers.clear()
        logger.info("Worker pool stopped")
    
    async def submit_job(
        self,
        request: LLMAnalysisRequest,
        session_id: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL,
        timeout: float = None,
        callback: Optional[Callable[[LLMServiceResponse], Awaitable[None]]] = None
    ) -> str:
        """
        Submit a job to the worker pool
        
        Args:
            request: LLM analysis request
            session_id: Optional session ID for FIFO processing
            priority: Job priority
            timeout: Job timeout in seconds
            callback: Optional callback for job completion
            
        Returns:
            Job ID
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        job = QueuedJob(
            id=str(uuid.uuid4()),
            request=request,
            session_id=session_id,
            priority=priority,
            created_at=datetime.utcnow(),
            callback=callback,
            timeout=timeout or getattr(settings, 'LLM_JOB_TIMEOUT', 300.0),
            max_retries=getattr(settings, 'LLM_MAX_RETRIES', 3)
        )
        
        async with self._lock:
            # Add to job tracking
            self.jobs[job.id] = job
            
            # Add to session queue for FIFO processing
            if session_id:
                if session_id not in self.session_queues:
                    self.session_queues[session_id] = []
                self.session_queues[session_id].append(job.id)
            
            # Add to main queue
            await self.job_queue.put(job)
            self.stats['jobs_queued'] += 1
        
        logger.info(f"Job {job.id} submitted to queue (session: {session_id}, priority: {priority.name})")
        return job.id
    
    async def get_job_status(self, job_id: str) -> Optional[QueuedJob]:
        """Get status of a specific job"""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self.stats['jobs_cancelled'] += 1
                return True
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics"""
        async with self._lock:
            active_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.PROCESSING)
            queued_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.QUEUED)
            
            return {
                'running': self.running,
                'workers': len(self.workers),
                'concurrency_limit': self.concurrency_limit,
                'queue_size': self.job_queue.qsize(),
                'queue_capacity': self.queue_size,
                'active_jobs': active_jobs,
                'queued_jobs': queued_jobs,
                'session_queues': len(self.session_queues),
                'rate_limiters': self.throttler.get_status(),
                'statistics': self.stats.copy()
            }
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes jobs from the queue"""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get job from queue with timeout to check for shutdown
                try:
                    job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    continue
                
                # Process the job
                await self._process_job(job, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_job(self, job: QueuedJob, worker_name: str):
        """Process a single job with retry logic"""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        
        logger.info(f"Worker {worker_name} processing job {job.id} (attempt {job.current_retry + 1})")
        
        try:
            # Apply rate limiting
            wait_time = await self.throttler.acquire_permit(job.request.provider)
            if wait_time > 0:
                logger.debug(f"Job {job.id} waited {wait_time:.2f}s for rate limit permit")
                self.stats['total_wait_time'] += wait_time
            
            # Execute the job with timeout
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    self.llm_service.analyze_gene_disease(job.request),
                    timeout=job.timeout
                )
                
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                
                # Job completed successfully
                job.result = result
                job.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                
                if result.success:
                    self.stats['jobs_processed'] += 1
                    logger.info(f"Job {job.id} completed successfully in {processing_time:.2f}s")
                else:
                    self.stats['jobs_failed'] += 1
                    logger.warning(f"Job {job.id} failed: {result.error.message if result.error else 'Unknown error'}")
                
                # Execute callback if provided
                if job.callback:
                    try:
                        await job.callback(result)
                    except Exception as e:
                        logger.error(f"Callback failed for job {job.id}: {e}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Job {job.id} timed out after {job.timeout}s")
                await self._handle_job_retry(job, "Job timeout")
                
            except Exception as e:
                logger.error(f"Job {job.id} failed with exception: {e}")
                await self._handle_job_retry(job, str(e))
                
        except Exception as e:
            logger.error(f"Critical error processing job {job.id}: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error = f"Critical error: {str(e)}"
            job.completed_at = datetime.utcnow()
            self.stats['jobs_failed'] += 1
    
    async def _handle_job_retry(self, job: QueuedJob, error_message: str):
        """Handle job retry logic with exponential backoff"""
        job.current_retry += 1
        
        if job.current_retry >= job.max_retries:
            # Max retries reached, mark as failed
            job.status = JobStatus.FAILED
            job.error = f"Max retries ({job.max_retries}) exceeded. Last error: {error_message}"
            job.completed_at = datetime.utcnow()
            self.stats['jobs_failed'] += 1
            logger.error(f"Job {job.id} failed permanently after {job.max_retries} attempts")
        else:
            # Schedule retry with exponential backoff
            delay = self.backoff.get_delay(job.current_retry - 1)
            self.stats['retry_count'] += 1
            
            logger.info(f"Retrying job {job.id} in {delay:.2f}s (attempt {job.current_retry + 1}/{job.max_retries})")
            
            # Reset job status and schedule retry
            job.status = JobStatus.QUEUED
            job.started_at = None
            
            # Add delay before re-queuing
            await asyncio.sleep(delay)
            
            # Re-queue the job
            try:
                await self.job_queue.put(job)
            except asyncio.QueueFull:
                logger.error(f"Failed to re-queue job {job.id}: queue full")
                job.status = JobStatus.FAILED
                job.error = "Failed to re-queue: queue full"
                job.completed_at = datetime.utcnow()
                self.stats['jobs_failed'] += 1


# Global worker pool instance
_worker_pool: Optional[WorkerPool] = None


async def get_worker_pool() -> WorkerPool:
    """Get or create the global worker pool instance"""
    global _worker_pool
    
    if _worker_pool is None:
        from .llm_client import get_llm_service
        llm_service = await get_llm_service()
        
        _worker_pool = WorkerPool(llm_service=llm_service)
        await _worker_pool.start()
    
    return _worker_pool


async def shutdown_worker_pool():
    """Shutdown the global worker pool"""
    global _worker_pool
    
    if _worker_pool:
        await _worker_pool.stop()
        _worker_pool = None


# Convenience functions for external use

async def submit_llm_job(
    request: LLMAnalysisRequest,
    session_id: Optional[str] = None,
    priority: JobPriority = JobPriority.NORMAL,
    timeout: Optional[float] = None,
    callback: Optional[Callable[[LLMServiceResponse], Awaitable[None]]] = None
) -> str:
    """Submit an LLM job to the worker pool"""
    pool = await get_worker_pool()
    return await pool.submit_job(request, session_id, priority, timeout, callback)


async def get_job_status(job_id: str) -> Optional[QueuedJob]:
    """Get the status of a job"""
    pool = await get_worker_pool()
    return await pool.get_job_status(job_id)


async def cancel_job(job_id: str) -> bool:
    """Cancel a queued job"""
    pool = await get_worker_pool()
    return await pool.cancel_job(job_id)


async def get_pool_status() -> Dict[str, Any]:
    """Get worker pool status and statistics"""
    pool = await get_worker_pool()
    return await pool.get_queue_status()