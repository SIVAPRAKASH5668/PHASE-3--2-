/**
 * API service for research discovery platform - Updated for your backend
 */

import axios from 'axios'
import { toast } from 'react-hot-toast'

// API Configuration
const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  TIMEOUT: 360000, // Increased for your research processing
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
}

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    config.metadata = { startTime: new Date() }
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    const duration = new Date() - response.config.metadata.startTime
    console.log(`✅ API Response: ${response.config.url} (${duration}ms)`)
    return response
  },
  async (error) => {
    const duration = new Date() - error.config?.metadata?.startTime
    console.error(`❌ API Error: ${error.config?.url} (${duration}ms)`, error.response?.data)
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token')
      toast.error('Session expired. Please log in again.')
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.')
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout. Please check your connection.')
    }
    
    return Promise.reject(error)
  }
)

// Retry mechanism
const retryRequest = async (fn, attempts = API_CONFIG.RETRY_ATTEMPTS) => {
  try {
    return await fn()
  } catch (error) {
    if (attempts > 1 && error.response?.status >= 500) {
      console.warn(`🔄 Retrying request... (${API_CONFIG.RETRY_ATTEMPTS - attempts + 1}/${API_CONFIG.RETRY_ATTEMPTS})`)
      await new Promise(resolve => setTimeout(resolve, API_CONFIG.RETRY_DELAY))
      return retryRequest(fn, attempts - 1)
    }
    throw error
  }
}

/**
 * Research Discovery API - Matching your backend exactly
 */
export const researchAPI = {
  /**
   * Discover research papers using your backend's format
   */
  discover: async (query, options = {}) => {
    // Format request to match your ResearchDiscoveryRequest model
    const requestData = {
      query: query.trim(),
      max_results: options.maxResults || 20,
      enable_multilingual: options.enableMultilingual !== false,
      sources: options.sources || ["core", "arxiv", "pubmed"],
      multilingual_keywords: options.multilingualKeywords || {},
      analysis_type: options.analysisDepth || "detailed",
      enable_graph: options.enableGraph !== false
    }

    return retryRequest(async () => {
      const response = await apiClient.post('/api/research/discover', requestData)
      
      // Your backend returns {success, data, metadata}
      if (!response.data.success) {
        throw new Error(response.data.error || 'Failed to discover research')
      }
      
      return {
        data: response.data.data,
        metadata: response.data.metadata,
        hasGraph: response.data.data?.has_graph_data || false
      }
    })
  },

  /**
   * Get paper details - Updated for your backend structure
   */
  getPaperDetails: async (paperId) => {
    if (!paperId) {
      throw new Error('Paper ID is required')
    }

    return retryRequest(async () => {
      // Try paper routes first (from your backend)
      try {
        const response = await apiClient.get(`/api/papers/details/${paperId}`)
        
        if (response.data.success !== false) {
          return {
            paper: response.data.paper || response.data,
            metadata: response.data.metadata || {}
          }
        }
      } catch (error) {
        // If paper routes don't work, try direct access
        console.warn('Paper routes not available, using fallback')
      }
      
      // Fallback - try to get from the discovered papers data
      throw new Error('Paper details not available - service may be initializing')
    })
  },

  /**
   * Health check for your backend
   */
  checkHealth: async () => {
    try {
      const response = await apiClient.get('/api/health', { timeout: 5000 })
      return {
        status: 'healthy',
        data: response.data
      }
    } catch (error) {
      // Try basic health endpoint
      try {
        const basicResponse = await apiClient.get('/health', { timeout: 5000 })
        return {
          status: basicResponse.data.status === 'healthy' ? 'healthy' : 'degraded',
          data: basicResponse.data
        }
      } catch (basicError) {
        return {
          status: 'unhealthy',
          error: error.message
        }
      }
    }
  },

  /**
   * Get system status
   */
  getStatus: async () => {
    return retryRequest(async () => {
      const response = await apiClient.get('/')
      return response.data
    })
  }
}

export default apiClient
