import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

/**
 * Application entry point
 * Renders the main App component with React 18's createRoot API
 */
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);