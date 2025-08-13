import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { apiService } from '../../services/api';

/**
 * Async thunk for creating a session with backend
 */
export const createSession = createAsyncThunk(
  'auth/createSession',
  async ({ username, apiKey }) => {
    const session = await apiService.createSession({
      username,
      api_key: apiKey
    });
    return { session, username, apiKey };
  }
);

/**
 * Authentication slice managing user credentials and session
 * Handles username, LLM API key, and session management
 */
const authSlice = createSlice({
  name: 'auth',
  initialState: {
    username: '',
    apiKey: '',
    sessionId: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
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
      state.sessionId = null;
      state.isAuthenticated = false;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(createSession.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(createSession.fulfilled, (state, action) => {
        const { session, username, apiKey } = action.payload;
        state.username = username;
        state.apiKey = apiKey;
        state.sessionId = session.id;
        state.isAuthenticated = true;
        state.isLoading = false;
        state.error = null;
      })
      .addCase(createSession.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to create session';
        state.isAuthenticated = false;
      });
  },
});

export const { setCredentials, clearCredentials } = authSlice.actions;

// Selectors for optimized component re-renders
export const selectUsername = (state) => state.auth.username;
export const selectApiKey = (state) => state.auth.apiKey;
export const selectSessionId = (state) => state.auth.sessionId;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;
export const selectAuthLoading = (state) => state.auth.isLoading;
export const selectAuthError = (state) => state.auth.error;

export default authSlice.reducer;