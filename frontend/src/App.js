import React from 'react';
import { Provider } from 'react-redux';
import { store } from './store';
import AuthForm from './components/AuthForm';
import AnalysisForm from './components/AnalysisForm';
import ResultsDisplay from './components/ResultsDisplay';
import AnalysisHistory from './components/AnalysisHistory';
import './App.css';

/**
 * Main application component integrating all features
 * Provides Redux store context and orchestrates component layout
 */
const App = () => {
  return (
    <Provider store={store}>
      <div className="app">
        <header className="app-header">
          <h1>Gene-Disease Analysis Tool</h1>
          <p>Analyze relationships between genes and diseases using AI-powered analysis</p>
        </header>
        
        <main className="app-main">
          {/* Authentication section - only shows when not authenticated */}
          <section className="auth-section">
            <AuthForm />
          </section>
          
          {/* Analysis section - only shows when authenticated */}
          <section className="analysis-section">
            <AnalysisForm />
          </section>
          
          {/* Results section - shows real-time progress */}
          <section className="results-section">
            <ResultsDisplay />
          </section>
          
          {/* History section - shows previous analyses */}
          <section className="history-section">
            <AnalysisHistory />
          </section>
        </main>
        
        <footer className="app-footer">
          <p>Gene-Disease Analysis Tool - Research and Educational Use Only</p>
        </footer>
      </div>
    </Provider>
  );
};

export default App;