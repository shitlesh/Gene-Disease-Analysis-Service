import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { apiService } from '../../services/api';

export const analyzeGeneDisease = createAsyncThunk(
  'analysis/analyzeGeneDisease',
  async ({ gene, disease, apiKey, sessionId }, { dispatch }) => {
    const analysis = await apiService.analyzeGeneDisease({
      gene,
      disease,
      session_id: sessionId,
      api_key: apiKey
    });

    if (analysis.analysis_id) {
      const eventSource = apiService.subscribeToAnalysis(
        analysis.analysis_id,
        (data) => {
          if (data.progress) {
            dispatch(updateAnalysisProgress(data.progress));
          }
        },
        (error) => {
          console.error('Analysis stream error:', error);
        }
      );
    }

    return {
      id: analysis.analysis_id,
      gene,
      disease,
      status: analysis.status,
      completedAt: new Date().toISOString()
    };
  }
);

export const loadAnalysisHistory = createAsyncThunk(
  'analysis/loadHistory',
  async ({ sessionId }) => {
    const response = await apiService.getAnalysisHistory(sessionId);
    return response.analyses || [];
  }
);
const analysisSlice = createSlice({
  name: 'analysis',
  initialState: {
    currentAnalysis: {
      isLoading: false,
      gene: '',
      disease: '',
      progress: '',
      error: null,
    },
    history: [],
  },
  reducers: {
    updateAnalysisProgress: (state, action) => {
      state.currentAnalysis.progress = action.payload;
    },
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
      .addCase(analyzeGeneDisease.fulfilled, (state, action) => {
        const result = action.payload;
        const completedAnalysis = {
          id: Date.now(),
          gene: state.currentAnalysis.gene,
          disease: state.currentAnalysis.disease,
          result: result.fullResult,
          timestamp: new Date().toISOString(),
        };
        state.history.unshift(completedAnalysis);
        state.currentAnalysis.isLoading = false;
        state.currentAnalysis.progress = 'Analysis completed!';
      })
      .addCase(analyzeGeneDisease.rejected, (state, action) => {
        state.currentAnalysis.isLoading = false;
        state.currentAnalysis.error = action.error.message || 'Analysis failed';
        state.currentAnalysis.progress = '';
      })
      .addCase(loadAnalysisHistory.fulfilled, (state, action) => {
        state.history = Array.isArray(action.payload) ? action.payload : [];
      });
  },
});

export const { updateAnalysisProgress, clearCurrentAnalysis } = analysisSlice.actions;

export const selectCurrentAnalysis = (state) => state.analysis.currentAnalysis;
export const selectAnalysisHistory = (state) => state.analysis.history;
export const selectIsAnalyzing = (state) => state.analysis.currentAnalysis.isLoading;

export default analysisSlice.reducer;