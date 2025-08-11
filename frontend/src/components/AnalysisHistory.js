import React, { memo, useState } from 'react';
import { useSelector } from 'react-redux';
import { selectAnalysisHistory } from '../features/analysis/analysisSlice';

/**
 * Individual analysis history item component
 * Displays collapsible analysis results
 * Memoized to prevent re-renders when other history items change
 */
const HistoryItem = memo(({ analysis }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  /**
   * Formats the timestamp for display
   */
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  /**
   * Truncates result text for preview
   */
  const getResultPreview = (result) => {
    return result.length > 100 ? result.substring(0, 100) + '...' : result;
  };

  return (
    <div className="history-item">
      <div className="history-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="history-info">
          <span className="gene-disease">
            <strong>{analysis.gene}</strong> × <strong>{analysis.disease}</strong>
          </span>
          <span className="timestamp">{formatTimestamp(analysis.timestamp)}</span>
        </div>
        <button className="expand-btn" type="button">
          {isExpanded ? '−' : '+'}
        </button>
      </div>
      
      <div className={`history-content ${isExpanded ? 'expanded' : 'collapsed'}`}>
        {isExpanded ? (
          <pre className="full-result">{analysis.result}</pre>
        ) : (
          <p className="result-preview">{getResultPreview(analysis.result)}</p>
        )}
      </div>
    </div>
  );
});

HistoryItem.displayName = 'HistoryItem';

/**
 * Analysis history component displaying all previous analyses
 * Shows analyses in reverse chronological order (newest first)
 * Optimized with memoization to prevent unnecessary re-renders
 */
const AnalysisHistory = memo(() => {
  const history = useSelector(selectAnalysisHistory);

  /**
   * Renders empty state when no history exists
   */
  const renderEmptyState = () => (
    <div className="empty-history">
      <p>No analysis history yet. Complete an analysis to see results here.</p>
    </div>
  );

  /**
   * Renders the list of historical analyses
   */
  const renderHistoryList = () => (
    <div className="history-list">
      {history.map((analysis) => (
        <HistoryItem 
          key={analysis.id} 
          analysis={analysis}
        />
      ))}
    </div>
  );

  return (
    <div className="analysis-history">
      <div className="history-header">
        <h3>Analysis History</h3>
        {history.length > 0 && (
          <span className="history-count">
            {history.length} analysis{history.length !== 1 ? 'es' : ''}
          </span>
        )}
      </div>
      
      {history.length === 0 ? renderEmptyState() : renderHistoryList()}
    </div>
  );
});

AnalysisHistory.displayName = 'AnalysisHistory';

export default AnalysisHistory;