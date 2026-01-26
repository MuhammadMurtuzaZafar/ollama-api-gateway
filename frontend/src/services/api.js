import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add API key to requests
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('apiKey');
  if (apiKey) {
    config.headers['Authorization'] = `Bearer ${apiKey}`;
  }
  return config;
});

// API Service methods
export const apiService = {
  // Authentication
  verifyApiKey: async (apiKey) => {
    try {
      // Try to get stats first - works for both admin and user
      const response = await axios.get(`${API_BASE_URL}/api/stats`, {
        headers: { 'Authorization': `Bearer ${apiKey}` }
      });
      
      // If successful, try to get keys to determine role (admin only)
      try {
        const keysResponse = await axios.get(`${API_BASE_URL}/api/keys`, {
          headers: { 'Authorization': `Bearer ${apiKey}` }
        });
        return { success: true, data: keysResponse.data, role: 'admin' };
      } catch {
        // Not admin, return with user role
        return { success: true, data: { api_keys: [] }, role: 'user' };
      }
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Invalid API key' };
    }
  },

  // API Keys Management
  createApiKey: async (name, role, rateLimit) => {
    try {
      const response = await api.post('/api/keys', { name, role, rate_limit: rateLimit });
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to create API key' };
    }
  },

  getApiKeys: async () => {
    try {
      const response = await api.get('/api/keys');
      return { success: true, data: response.data.api_keys || [] };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to fetch API keys' };
    }
  },

  revealApiKey: async (keyId) => {
    try {
      const response = await api.get(`/api/keys/${keyId}/reveal`);
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to reveal API key' };
    }
  },

  revokeApiKey: async (keyId) => {
    try {
      const response = await api.delete(`/api/keys/${keyId}`);
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to revoke API key' };
    }
  },

  // Statistics
  getStats: async () => {
    try {
      const response = await api.get('/api/stats');
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to fetch stats' };
    }
  },

  getDetailedStats: async () => {
    try {
      const response = await api.get('/api/stats/detailed');
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to fetch detailed stats' };
    }
  },

  // Admin: Get user statistics
  getUserStats: async () => {
    try {
      const response = await api.get('/api/admin/stats/users');
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to fetch user stats' };
    }
  },

  // EdgeLLM API proxied calls
  generateCompletion: async (model, prompt, options = {}) => {
    try {
      const response = await api.post('/api/generate', { model, prompt, ...options });
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to generate completion' };
    }
  },

  chatCompletion: async (model, messages, options = {}) => {
    try {
      const response = await api.post('/api/chat', { model, messages, ...options });
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to chat' };
    }
  },

  listModels: async () => {
    try {
      const response = await api.get('/api/models');
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: error.response?.data?.detail || 'Failed to list models' };
    }
  },
};

export default api;
