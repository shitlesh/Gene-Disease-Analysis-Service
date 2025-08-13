import React from 'react';
import { Provider } from 'react-redux';
import { store } from './store';
import AuthForm from './components/AuthForm';
import AnalysisForm from './components/AnalysisForm';
import ResultsDisplay from './components/ResultsDisplay';
import AnalysisHistory from './components/AnalysisHistory';
import './App.css';

const App = () => {
  return (
    <Provider store={store}>
      <div className="app">
        <header className="app-header">
          <h1>Gene-Disease Analysis Tool</h1>
          <p>Analyze relationships between genes and diseases using AI-powered analysis</p>
        </header>
        
        <main className="app-main">
          <section className="auth-section">
            <AuthForm />
          </section>
          <section className="analysis-section">
            <AnalysisForm />
          </section>
          <section className="results-section">
            <ResultsDisplay />
          </section>
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