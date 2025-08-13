import React, { useState, memo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { analyzeGeneDisease } from '../features/analysis/analysisSlice';
import { selectApiKey, selectIsAuthenticated, selectSessionId } from '../features/auth/authSlice';
import { selectIsAnalyzing } from '../features/analysis/analysisSlice';

/**
 * Gene-disease analysis form component
 * Allows users to input gene and disease names for analysis
 * Memoized to prevent unnecessary re-renders when parent updates
 */
const AnalysisForm = memo(() => {
  const dispatch = useDispatch();
  const apiKey = useSelector(selectApiKey);
  const sessionId = useSelector(selectSessionId);
  const isAuthenticated = useSelector(selectIsAuthenticated);
  const isAnalyzing = useSelector(selectIsAnalyzing);
  
  const [formData, setFormData] = useState({
    gene: '',
    disease: '',
  });
  const [errors, setErrors] = useState({});

  /**
   * Handles input field changes with real-time validation
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Only allow alphanumeric characters, hyphens, and spaces for gene/disease names
    const sanitizedValue = value.replace(/[^a-zA-Z0-9\s\-]/g, '');
    
    setFormData(prev => ({
      ...prev,
      [name]: sanitizedValue,
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: '',
      }));
    }
  };

  /**
   * Validates form inputs before submission
   */
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.gene.trim()) {
      newErrors.gene = 'Gene name is required';
    } else if (formData.gene.trim().length < 2) {
      newErrors.gene = 'Gene name must be at least 2 characters';
    }
    
    if (!formData.disease.trim()) {
      newErrors.disease = 'Disease name is required';
    } else if (formData.disease.trim().length < 3) {
      newErrors.disease = 'Disease name must be at least 3 characters';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handles form submission and dispatches analysis action
   */
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!isAuthenticated) {
      return;
    }
    
    if (validateForm() && !isAnalyzing) {
      dispatch(analyzeGeneDisease({
        gene: formData.gene.trim(),
        disease: formData.disease.trim(),
        apiKey,
        sessionId,
      }));
    }
  };

  // Don't render if not authenticated
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="analysis-form">
      <h2>Gene-Disease Analysis</h2>
      <p>Enter a gene name and disease to analyze their relationship.</p>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="gene">Gene Name:</label>
          <input
            type="text"
            id="gene"
            name="gene"
            value={formData.gene}
            onChange={handleInputChange}
            placeholder="e.g., BRCA1, TP53, CFTR"
            disabled={isAnalyzing}
            className={errors.gene ? 'error' : ''}
          />
          {errors.gene && <span className="error-message">{errors.gene}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="disease">Disease Name:</label>
          <input
            type="text"
            id="disease"
            name="disease"
            value={formData.disease}
            onChange={handleInputChange}
            placeholder="e.g., breast cancer, cystic fibrosis"
            disabled={isAnalyzing}
            className={errors.disease ? 'error' : ''}
          />
          {errors.disease && <span className="error-message">{errors.disease}</span>}
        </div>

        <button 
          type="submit" 
          className="submit-btn"
          disabled={isAnalyzing}
        >
          {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
        </button>
      </form>
    </div>
  );
});

AnalysisForm.displayName = 'AnalysisForm';

export default AnalysisForm;