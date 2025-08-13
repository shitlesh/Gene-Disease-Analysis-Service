import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { apiService } from '../../services/api';

/**
 * Async thunk for performing gene-disease analysis
 * Uses real API with Server-Sent Events for real-time updates
 */
export const analyzeGeneDisease = createAsyncThunk(
  'analysis/analyzeGeneDisease',
  async ({ gene, disease, apiKey, sessionId }, { dispatch }) => {
    // Start the analysis via API
    const analysis = await apiService.analyzeGeneDisease({
      gene,
      disease,
      session_id: sessionId,
      api_key: apiKey
    });

    // Subscribe to real-time updates
    const eventSource = apiService.subscribeToAnalysis(
      analysis.id,
      (data) => {
        if (data.progress) {
          dispatch(updateAnalysisProgress(data.progress));
        }
      },
      (error) => {
        console.error('Analysis stream error:', error);
      }
    );

    // Return the analysis result
    return {
      id: analysis.id,
      gene,
      disease,
      eventSource, // Store for cleanup
      completedAt: new Date().toISOString()
    };
  }
);

/**
 * Async thunk for loading analysis history
 */
export const loadAnalysisHistory = createAsyncThunk(
  'analysis/loadHistory',
  async ({ sessionId }) => {
    const response = await apiService.getAnalysisHistory(sessionId);
    // Extract the analyses array from the response object
    return response.analyses || [];
  }
);

/**
 * Analysis slice managing gene-disease analysis state
 * Handles current analysis, history, and streaming updates
 */
const analysisSlice = createSlice({
  name: 'analysis',
  initialState: {
    // Current analysis state
    currentAnalysis: {
      isLoading: false,
      gene: '',
      disease: '',
      progress: '',
      error: null,
    },
    // History of all analyses in reverse chronological order
    history: [],
  },
  reducers: {
    /**
     * Updates the current analysis progress for real-time streaming
     * @param {Object} action.payload - Progress text to display
     */
    updateAnalysisProgress: (state, action) => {
      state.currentAnalysis.progress = action.payload;
    },
    
    /**
     * Clears the current analysis state
     */
    clearCurrentAnalysis: (state) => {
      state.currentAnalysis = {
        isLoading: false,
        gene: '',
        disease: '',
        progress: '',
        error: null,
      };
    },
  },
  extraReducers: (builder) => {
    builder
      // Handle analysis start
      .addCase(analyzeGeneDisease.pending, (state, action) => {
        const { gene, disease } = action.meta.arg;
        state.currentAnalysis = {
          isLoading: true,
          gene,
          disease,
          progress: 'Starting analysis...',
          error: null,
        };
      })
      
      // Handle successful analysis completion
      .addCase(analyzeGeneDisease.fulfilled, (state, action) => {
        const result = action.payload;
        
        // Add completed analysis to history at the beginning (reverse chronological)
        const completedAnalysis = {
          id: Date.now(),
          gene: state.currentAnalysis.gene,
          disease: state.currentAnalysis.disease,
          result: result.fullResult,
          timestamp: new Date().toISOString(),
        };
        
        state.history.unshift(completedAnalysis);
        
        // Update current analysis to show completion
        state.currentAnalysis.isLoading = false;
        state.currentAnalysis.progress = 'Analysis completed!';
      })
      
      // Handle analysis failure
      .addCase(analyzeGeneDisease.rejected, (state, action) => {
        state.currentAnalysis.isLoading = false;
        state.currentAnalysis.error = action.error.message || 'Analysis failed';
        state.currentAnalysis.progress = '';
      })
      
      // Handle loading analysis history
      .addCase(loadAnalysisHistory.fulfilled, (state, action) => {
        state.history = Array.isArray(action.payload) ? action.payload : [];
      });
  },
});

export const { updateAnalysisProgress, clearCurrentAnalysis } = analysisSlice.actions;

// Optimized selectors to prevent unnecessary re-renders
export const selectCurrentAnalysis = (state) => state.analysis.currentAnalysis;
export const selectAnalysisHistory = (state) => state.analysis.history;
export const selectIsAnalyzing = (state) => state.analysis.currentAnalysis.isLoading;

export default analysisSlice.reducer;