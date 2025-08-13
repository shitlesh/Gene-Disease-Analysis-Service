import axios from 'axios';

// Base API configuration

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Core API service functions - simplified for requirements
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Session management - simplified to username + API key
  async createSession(userData) {
    const response = await api.post('/session/', userData);
    return response.data;
  },

  async getSession(sessionId) {
    const response = await api.get(`/session/${sessionId}`);
    return response.data;
  },

  // Gene-Disease Analysis - core functionality
  async analyzeGeneDisease(analysisData) {
    const response = await api.post('/analysis/', analysisData);
    return response.data;
  },

  async getAnalysis(analysisId) {
    const response = await api.get(`/analysis/${analysisId}`);
    return response.data;
  },

  async getAnalysisHistory(sessionId) {
    const response = await api.get(`/analysis/history/${sessionId}`);
    return response.data;
  },

  // Real-time updates using Server-Sent Events
  subscribeToAnalysis(analysisId, onUpdate, onError) {
    const streamUrl = `${API_BASE_URL}/analysis/stream/${analysisId}`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        console.error('Error parsing SSE data:', error);
        onError?.(error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE Error:', error);
      onError?.(error);
    };

    return eventSource;
  }
};

export default apiService;