import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);
export const apiService = {
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  async createSession(userData) {
    const response = await api.post('/api/v1/session', userData);
    return response.data;
  },

  async getSession(sessionId) {
    const response = await api.get(`/api/v1/session/${sessionId}`);
    return response.data;
  },

  async analyzeGeneDisease(analysisData) {
    const response = await api.post('/api/v1/analysis', analysisData);
    return response.data;
  },

  async getAnalysis(analysisId) {
    const response = await api.get(`/api/v1/analysis/${analysisId}`);
    return response.data;
  },

  async getAnalysisHistory(sessionId) {
    const response = await api.get(`/api/v1/history/${sessionId}`);
    return response.data;
  },

  subscribeToAnalysis(analysisId, onUpdate, onError) {
    const streamUrl = `${API_BASE_URL}/api/v1/stream/${analysisId}`;
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