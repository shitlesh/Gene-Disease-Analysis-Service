import React, { useState, memo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { setCredentials, selectIsAuthenticated } from '../features/auth/authSlice';

/**
 * Authentication form component for capturing user credentials
 * Handles username and API key input with form validation
 * Memoized to prevent unnecessary re-renders
 */
const AuthForm = memo(() => {
  const dispatch = useDispatch();
  const isAuthenticated = useSelector(selectIsAuthenticated);
  
  const [formData, setFormData] = useState({
    username: '',
    apiKey: '',
  });
  const [errors, setErrors] = useState({});

  /**
   * Handles input field changes with validation
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
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
   * Validates form data before submission
   */
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }
    
    if (!formData.apiKey.trim()) {
      newErrors.apiKey = 'API key is required';
    } else if (formData.apiKey.length < 10) {
      newErrors.apiKey = 'API key appears to be invalid';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handles form submission and stores credentials in Redux
   */
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (validateForm()) {
      dispatch(setCredentials({
        username: formData.username.trim(),
        apiKey: formData.apiKey.trim(),
      }));
    }
  };

  // Don't render if already authenticated
  if (isAuthenticated) {
    return null;
  }

  return (
    <div className="auth-form">
      <h2>Authentication Required</h2>
      <p>Please enter your credentials to access the gene analysis tool.</p>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username:</label>
          <input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleInputChange}
            placeholder="Enter your username"
            className={errors.username ? 'error' : ''}
          />
          {errors.username && <span className="error-message">{errors.username}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="apiKey">LLM API Key (OpenAI or Anthropic):</label>
          <input
            type="password"
            id="apiKey"
            name="apiKey"
            value={formData.apiKey}
            onChange={handleInputChange}
            placeholder="sk-... or anthropic-..."
            className={errors.apiKey ? 'error' : ''}
          />
          {errors.apiKey && <span className="error-message">{errors.apiKey}</span>}
        </div>

        <button type="submit" className="submit-btn">
          Authenticate
        </button>
      </form>
    </div>
  );
});

AuthForm.displayName = 'AuthForm';

export default AuthForm;