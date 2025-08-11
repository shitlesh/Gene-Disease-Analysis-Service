from datetime import datetime
from typing import Dict, List, Optional
import uuid
import threading
from ..models.session import SessionInfo
from ..models.analysis import AnalysisResult, AnalysisStatus


class MemoryStore:
    """
    In-memory storage system for sessions and analyses
    Thread-safe implementation ready for database migration
    """
    
    def __init__(self):
        """
        Initialize storage with thread-safe data structures
        Uses locks to prevent race conditions in concurrent requests
        """
        # Session storage: session_id -> SessionInfo
        self._sessions: Dict[str, SessionInfo] = {}
        
        # Analysis storage: analysis_id -> AnalysisResult
        self._analyses: Dict[str, AnalysisResult] = {}
        
        # Session-to-analyses mapping: session_id -> List[analysis_id]
        self._session_analyses: Dict[str, List[str]] = {}
        
        # Thread locks for concurrent access safety
        self._session_lock = threading.RLock()
        self._analysis_lock = threading.RLock()
    
    # Session Management Methods
    
    def create_session(self, username: str, api_key: str) -> SessionInfo:
        """
        Creates a new user session with unique identifier
        Thread-safe session creation with automatic ID generation
        """
        with self._session_lock:
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            session = SessionInfo(
                session_id=session_id,
                username=username,
                api_key=api_key,
                created_at=now,
                last_accessed=now
            )
            
            self._sessions[session_id] = session
            self._session_analyses[session_id] = []
            
            return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Retrieves session by ID and updates last accessed time
        Returns None if session doesn't exist
        """
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                # Update last accessed time for session tracking
                session.last_accessed = datetime.utcnow()
            return session
    
    def session_exists(self, session_id: str) -> bool:
        """
        Checks if a session exists without updating access time
        Useful for authentication validation
        """
        with self._session_lock:
            return session_id in self._sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Removes session and all associated analyses
        Returns True if session existed, False otherwise
        """
        with self._session_lock:
            if session_id in self._sessions:
                # Remove session
                del self._sessions[session_id]
                
                # Remove associated analyses
                if session_id in self._session_analyses:
                    analysis_ids = self._session_analyses[session_id]
                    with self._analysis_lock:
                        for analysis_id in analysis_ids:
                            self._analyses.pop(analysis_id, None)
                    del self._session_analyses[session_id]
                
                return True
            return False
    
    # Analysis Management Methods
    
    def create_analysis(self, session_id: str, gene: str, disease: str) -> AnalysisResult:
        """
        Creates a new analysis record in pending state
        Links analysis to the requesting session
        """
        with self._analysis_lock:
            analysis_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            analysis = AnalysisResult(
                analysis_id=analysis_id,
                session_id=session_id,
                gene=gene,
                disease=disease,
                status=AnalysisStatus.PENDING,
                summary="Analysis pending...",
                confidence_score=0.0,
                key_findings=[],
                pathway_analysis=None,
                therapeutic_targets=None,
                created_at=now,
                completed_at=None,
                processing_time_seconds=None
            )
            
            self._analyses[analysis_id] = analysis
            
            # Add to session's analysis list
            with self._session_lock:
                if session_id in self._session_analyses:
                    self._session_analyses[session_id].append(analysis_id)
            
            return analysis
    
    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """
        Retrieves a specific analysis by ID
        Returns None if analysis doesn't exist
        """
        with self._analysis_lock:
            return self._analyses.get(analysis_id)
    
    def update_analysis(self, analysis_id: str, **updates) -> Optional[AnalysisResult]:
        """
        Updates analysis fields with provided values
        Returns updated analysis or None if not found
        """
        with self._analysis_lock:
            analysis = self._analyses.get(analysis_id)
            if analysis:
                # Create updated analysis with new values
                analysis_dict = analysis.dict()
                analysis_dict.update(updates)
                
                # Validate and recreate the analysis object
                updated_analysis = AnalysisResult(**analysis_dict)
                self._analyses[analysis_id] = updated_analysis
                
                return updated_analysis
            return None
    
    def get_session_analyses(self, session_id: str) -> List[AnalysisResult]:
        """
        Retrieves all analyses for a specific session
        Returns analyses in reverse chronological order (newest first)
        """
        with self._analysis_lock:
            if session_id not in self._session_analyses:
                return []
            
            analysis_ids = self._session_analyses[session_id]
            analyses = []
            
            for analysis_id in analysis_ids:
                analysis = self._analyses.get(analysis_id)
                if analysis:
                    analyses.append(analysis)
            
            # Sort by creation time, newest first
            analyses.sort(key=lambda x: x.created_at, reverse=True)
            return analyses
    
    def get_analysis_count(self, session_id: str) -> int:
        """
        Returns the total number of analyses for a session
        Useful for pagination and statistics
        """
        with self._session_lock:
            return len(self._session_analyses.get(session_id, []))
    
    # Utility Methods for System Management
    
    def get_total_sessions(self) -> int:
        """Returns total number of active sessions"""
        with self._session_lock:
            return len(self._sessions)
    
    def get_total_analyses(self) -> int:
        """Returns total number of analyses across all sessions"""
        with self._analysis_lock:
            return len(self._analyses)
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Removes sessions older than specified hours
        Returns number of sessions cleaned up
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        expired_sessions = []
        
        with self._session_lock:
            for session_id, session in self._sessions.items():
                if session.last_accessed.timestamp() < cutoff_time:
                    expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        return len(expired_sessions)


# Global storage instance
# In production, this would be replaced with database connections
storage = MemoryStore()