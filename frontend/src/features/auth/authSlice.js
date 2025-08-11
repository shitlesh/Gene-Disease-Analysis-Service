import { createSlice } from '@reduxjs/toolkit';

/**
 * Authentication slice managing user credentials
 * Handles username and LLM API key storage in memory only
 */
const authSlice = createSlice({
  name: 'auth',
  initialState: {
    username: '',
    apiKey: '',
    isAuthenticated: false,
  },
  reducers: {
    /**
     * Sets user credentials and marks as authenticated
     * @param {Object} action.payload - Contains username and apiKey
     */
    setCredentials: (state, action) => {
      const { username, apiKey } = action.payload;
      state.username = username;
      state.apiKey = apiKey;
      state.isAuthenticated = !!(username && apiKey);
    },
    
    /**
     * Clears all authentication data
     */
    clearCredentials: (state) => {
      state.username = '';
      state.apiKey = '';
      state.isAuthenticated = false;
    },
  },
});

export const { setCredentials, clearCredentials } = authSlice.actions;

// Selectors for optimized component re-renders
export const selectUsername = (state) => state.auth.username;
export const selectApiKey = (state) => state.auth.apiKey;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;

export default authSlice.reducer;