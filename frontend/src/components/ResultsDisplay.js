import React, { memo } from 'react';
import { useSelector } from 'react-redux';
import { selectCurrentAnalysis, selectIsAnalyzing } from '../features/analysis/analysisSlice';

/**
 * Real-time results display component showing streaming analysis progress
 * Updates in real-time as analysis progresses
 * Memoized for optimal performance during frequent updates
 */
const ResultsDisplay = memo(() => {
  const currentAnalysis = useSelector(selectCurrentAnalysis);
  const isAnalyzing = useSelector(selectIsAnalyzing);

  /**
   * Renders the current analysis progress with visual indicators
   */
  const renderProgress = () => {
    if (!currentAnalysis.progress) {
      return null;
    }

    return (
      <div className="progress-section">
        <div className="progress-header">
          {isAnalyzing && <div className="loading-spinner"></div>}
          <h4>
            {isAnalyzing ? 'Analysis in Progress' : 'Analysis Complete'}
          </h4>
        </div>
        <div className="progress-content">
          <p className="progress-text">{currentAnalysis.progress}</p>
          {isAnalyzing && (
            <div className="progress-bar">
              <div className="progress-fill"></div>
            </div>
          )}
        </div>
      </div>
    );
  };

  /**
   * Renders error state if analysis failed
   */
  const renderError = () => {
    if (!currentAnalysis.error) {
      return null;
    }

    return (
      <div className="error-section">
        <h4>Analysis Failed</h4>
        <p className="error-text">{currentAnalysis.error}</p>
      </div>
    );
  };

  /**
   * Renders the current analysis details when active
   */
  const renderCurrentAnalysis = () => {
    if (!currentAnalysis.gene && !currentAnalysis.disease) {
      return (
        <div className="no-analysis">
          <p>No analysis in progress. Use the form above to start an analysis.</p>
        </div>
      );
    }

    return (
      <div className="current-analysis">
        <div className="analysis-header">
          <h4>Current Analysis</h4>
          <div className="analysis-details">
            <span className="detail-item">
              <strong>Gene:</strong> {currentAnalysis.gene}
            </span>
            <span className="detail-item">
              <strong>Disease:</strong> {currentAnalysis.disease}
            </span>
          </div>
        </div>
        
        {renderProgress()}
        {renderError()}
      </div>
    );
  };

  return (
    <div className="results-display">
      <h3>Real-Time Results</h3>
      {renderCurrentAnalysis()}
    </div>
  );
});

ResultsDisplay.displayName = 'ResultsDisplay';

export default ResultsDisplay;