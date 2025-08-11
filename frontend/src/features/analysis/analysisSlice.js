import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { mockAnalyzeGeneDisease } from '../../services/mockApi';

/**
 * Async thunk for performing gene-disease analysis
 * Simulates streaming results by updating progress incrementally
 */
export const analyzeGeneDisease = createAsyncThunk(
  'analysis/analyzeGeneDisease',
  async ({ gene, disease, apiKey }, { dispatch }) => {
    // Start the mock analysis which will dispatch streaming updates
    const result = await mockAnalyzeGeneDisease(gene, disease, apiKey, (progress) => {
      // Dispatch streaming updates as the analysis progresses
      dispatch(updateAnalysisProgress(progress));
    });
    return result;
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
      });
  },
});

export const { updateAnalysisProgress, clearCurrentAnalysis } = analysisSlice.actions;

// Optimized selectors to prevent unnecessary re-renders
export const selectCurrentAnalysis = (state) => state.analysis.currentAnalysis;
export const selectAnalysisHistory = (state) => state.analysis.history;
export const selectIsAnalyzing = (state) => state.analysis.currentAnalysis.isLoading;

export default analysisSlice.reducer;