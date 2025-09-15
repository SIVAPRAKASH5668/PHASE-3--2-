import { useState, useCallback, useRef } from 'react'
import toast from 'react-hot-toast'

// Create API handler
const createAPI = () => ({
  discover: async (query, options = {}) => {
    console.log('🚀 Starting API call to backend...')
    
    try {
      const response = await fetch('http://localhost:8000/api/research/discover', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          query,
          max_papers: options.maxResults || 50,
          sources: options.sources || ['core', 'arxiv', 'pubmed'],
          analysis_depth: options.analysisDepth || 'detailed',
          enable_graph: true,
          ...options
        })
      })

      console.log('📡 Backend response received:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Backend Error: ${response.status} - ${errorText}`)
      }

      const data = await response.json()
      console.log('✅ Backend data parsed successfully')
      return { data }

    } catch (error) {
      console.error('❌ API Error:', error.message)
      throw error
    }
  },

  // **NEW: Paper details API call**
  getPaperDetails: async (paperId) => {
    console.log('📄 Fetching paper details for ID:', paperId)
    
    try {
      const response = await fetch(`http://localhost:8000/api/papers/details/${paperId}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      })

      console.log('📡 Paper details response:', response.status)

      if (!response.ok) {
        throw new Error(`Paper details API Error: ${response.status}`)
      }

      const data = await response.json()
      console.log('✅ Paper details received')
      return data

    } catch (error) {
      console.error('❌ Paper details error:', error.message)
      throw error
    }
  }
})

const researchAPI = createAPI()

export const useResearchData = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [nodeDetails, setNodeDetails] = useState(null)
  const [loadingDetails, setLoadingDetails] = useState(false)
  const [statistics, setStatistics] = useState(null)
  
  const abortControllerRef = useRef(null)
  const cacheRef = useRef(new Map())

  // Transform backend data format to frontend format
  const transformBackendData = useCallback((backendResponse) => {
    console.log('🔄 Transforming backend response...')
    
    if (!backendResponse?.data) {
      console.warn('⚠️ No data in backend response')
      return { nodes: [], links: [] }
    }

    const responseData = backendResponse.data

    // Check for nested data.graph structure first
    if (responseData.data?.graph?.nodes && responseData.data?.graph?.edges) {
      console.log('✅ Using nested data.graph structure from backend')
      const graphData = responseData.data.graph
      
      const nodes = graphData.nodes.map(node => ({
        id: String(node.id),
        title: node.title || 'Untitled Paper',
        label: truncateText(node.title || 'Untitled', 50),
        size: Math.max(15, Math.min(40, (node.size || 20) * 1.5)),
        color: node.color || getDomainColor(node.research_domain),
        quality_score: Number(node.quality_score || 0.5),
        research_domain: node.research_domain || 'General Research',
        authors: node.authors || 'Unknown Authors',
        published_date: node.published_date || '',
        citation_count: Number(node.citation_count || 0),
        paper_url: node.paper_url || '',
        abstract: node.abstract || node.context_summary || 'Abstract not available',
        context_summary: node.context_summary || '',
        methodology: node.methodology || 'Not specified',
        key_findings: Array.isArray(node.key_findings) ? node.key_findings : [],
        innovations: Array.isArray(node.innovations) ? node.innovations : [],
        contributions: Array.isArray(node.contributions) ? node.contributions : [],
        language: node.language || 'en',
        source: node.source || 'unknown',
        ai_agent_used: node.ai_agent_used || '',
        analysis_confidence: Number(node.analysis_confidence || 0.5),
        embedding_similarity: Number(node.embedding_similarity || 0),
        citation_potential: Number(node.citation_potential || 0.5)
      }))

      const links = graphData.edges.map(edge => ({
        source: String(edge.source),
        target: String(edge.target),
        strength: Math.max(0.3, Number(edge.strength || 0.5)),
        weight: Number(edge.weight || edge.strength || 0.5),
        type: edge.relationship_type || 'related',
        context: edge.context || 'Related research',
        reasoning: edge.reasoning || 'Semantic similarity',
        confidence: Number(edge.confidence_score || 0.5),
        semantic_similarity: Number(edge.semantic_similarity || 0.3)
      }))

      console.log('📊 Nested graph transformation complete:', { 
        nodes: nodes.length, 
        links: links.length 
      })

      return { nodes, links }
    }

    // Check for direct graph structure
    if (responseData.graph?.nodes && responseData.graph?.edges) {
      console.log('✅ Using direct graph structure from backend')
      
      const nodes = responseData.graph.nodes.map(node => ({
        id: String(node.id),
        title: node.title || 'Untitled Paper',
        label: truncateText(node.title || 'Untitled', 50),
        size: Math.max(15, Math.min(40, (node.size || 20) * 1.5)),
        color: node.color || getDomainColor(node.research_domain),
        quality_score: Number(node.quality_score || 0.5),
        research_domain: node.research_domain || 'General Research',
        authors: node.authors || 'Unknown Authors',
        published_date: node.published_date || '',
        citation_count: Number(node.citation_count || 0),
        paper_url: node.paper_url || '',
        abstract: node.abstract || node.context_summary || 'Abstract not available',
        context_summary: node.context_summary || '',
        methodology: node.methodology || 'Not specified',
        key_findings: Array.isArray(node.key_findings) ? node.key_findings : [],
        innovations: Array.isArray(node.innovations) ? node.innovations : [],
        contributions: Array.isArray(node.contributions) ? node.contributions : [],
        language: node.language || 'en',
        source: node.source || 'unknown',
        ai_agent_used: node.ai_agent_used || '',
        analysis_confidence: Number(node.analysis_confidence || 0.5),
        embedding_similarity: Number(node.embedding_similarity || 0),
        citation_potential: Number(node.citation_potential || 0.5)
      }))

      const links = responseData.graph.edges.map(edge => ({
        source: String(edge.source),
        target: String(edge.target),
        strength: Math.max(0.3, Number(edge.strength || 0.5)),
        weight: Number(edge.weight || edge.strength || 0.5),
        type: edge.relationship_type || 'related',
        context: edge.context || 'Related research',
        reasoning: edge.reasoning || 'Semantic similarity',
        confidence: Number(edge.confidence_score || 0.5),
        semantic_similarity: Number(edge.semantic_similarity || 0.3)
      }))

      console.log('📊 Direct graph transformation complete:', { 
        nodes: nodes.length, 
        links: links.length 
      })

      return { nodes, links }
    }

    // Handle papers + relationships format (nested or direct)
    let papers = responseData.data?.papers || responseData.papers || []
    let relationships = responseData.data?.relationships || responseData.relationships || []

    if (papers && Array.isArray(papers) && papers.length > 0) {
      console.log('✅ Using papers + relationships data from backend')
      
      const nodes = papers.map((paper, index) => ({
        id: String(paper.id || `paper-${index}`),
        title: paper.title || `Research Paper ${index + 1}`,
        label: truncateText(paper.title || `Paper ${index + 1}`, 50),
        size: calculateNodeSize(paper),
        color: getDomainColor(paper.research_domain),
        quality_score: Number(paper.context_quality_score || paper.quality_score || 0.5),
        research_domain: paper.research_domain || 'General Research',
        authors: paper.authors || 'Unknown Authors',
        published_date: paper.published_date || '',
        citation_count: Number(paper.citation_count || 0),
        paper_url: paper.paper_url || '',
        abstract: paper.abstract || paper.context_summary || 'Abstract not available',
        context_summary: paper.context_summary || '',
        methodology: paper.methodology || 'Not specified',
        key_findings: Array.isArray(paper.key_findings) ? paper.key_findings : [],
        innovations: Array.isArray(paper.innovations) ? paper.innovations : [],
        contributions: Array.isArray(paper.contributions) ? paper.contributions : [],
        language: paper.language || 'en',
        source: paper.source || 'unknown',
        ai_agent_used: paper.ai_agent_used || '',
        analysis_confidence: Number(paper.analysis_confidence || 0.5)
      }))

      // Transform relationships to links
      const nodeIds = new Set(nodes.map(n => n.id))
      const links = relationships
        .filter(rel => {
          const sourceExists = nodeIds.has(String(rel.paper1_id))
          const targetExists = nodeIds.has(String(rel.paper2_id))
          return sourceExists && targetExists
        })
        .map(rel => ({
          source: String(rel.paper1_id),
          target: String(rel.paper2_id),
          strength: Math.max(0.3, Number(rel.relationship_strength || 0.5)),
          type: rel.relationship_type || 'related',
          context: rel.relationship_context || 'Related research',
          reasoning: rel.connection_reasoning || 'Semantic similarity',
          confidence: Number(rel.confidence_score || 0.5),
          semantic_similarity: Number(rel.semantic_similarity || 0.3)
        }))

      console.log('📊 Papers transformation complete:', { 
        nodes: nodes.length, 
        links: links.length 
      })

      return { nodes, links }
    }

    console.warn('⚠️ No recognized data format in backend response')
    console.log('🔍 Response structure:', Object.keys(responseData))
    if (responseData.data) {
      console.log('🔍 Nested data structure:', Object.keys(responseData.data))
    }
    
    return { nodes: [], links: [] }
  }, [])

  // Main search function
  const searchPapers = useCallback(async (query, filters = {}) => {
    const cacheKey = `${query}-${JSON.stringify(filters)}`
    
    // Check cache first
    if (cacheRef.current.has(cacheKey)) {
      const cached = cacheRef.current.get(cacheKey)
      setData(cached)
      setStatistics(calculateStatistics(cached))
      toast.success('📂 Loaded from cache', { duration: 2000 })
      return cached
    }

    try {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      abortControllerRef.current = new AbortController()

      setLoading(true)
      setError(null)
      
      // Show extended loading message
      const loadingToast = toast.loading(
        '🧠 Backend is processing your query... This may take 2-5 minutes for comprehensive analysis.', 
        { duration: Infinity }
      )

      console.log('🚀 Initiating search for:', query)
      console.log('🔧 Search parameters:', filters)

      const result = await researchAPI.discover(query, {
        maxResults: Math.min(100, filters.maxResults || 50),
        sources: filters.sources || ['core', 'arxiv', 'pubmed'],
        analysisDepth: filters.analysisDepth || 'detailed',
        ...filters
      })

      console.log('✅ Backend processing complete!')
      console.log('📊 Raw backend result:', result)

      const transformedData = transformBackendData(result)
      
      if (transformedData.nodes.length === 0) {
        throw new Error('No research papers found for this query. Try different keywords or sources.')
      }

      // Cache result
      cacheRef.current.set(cacheKey, transformedData)
      
      // Limit cache size
      if (cacheRef.current.size > 3) {
        const firstKey = cacheRef.current.keys().next().value
        cacheRef.current.delete(firstKey)
      }

      setData(transformedData)
      
      const stats = calculateStatistics(transformedData)
      setStatistics(stats)
      
      // Dismiss loading toast and show success
      toast.dismiss(loadingToast)
      toast.success(
        `✅ Built research graph: ${transformedData.nodes.length} papers with ${transformedData.links.length} connections!`, 
        { duration: 4000 }
      )
      
      return transformedData

    } catch (err) {
      // Don't treat abort as error since backend continues processing
      if (err.name === 'AbortError') {
        console.log('🔄 Frontend request cancelled, but backend continues processing')
        toast.info('⏳ Request sent to backend. Processing continues...', { duration: 3000 })
        return null
      }

      const errorMessage = err.message || 'Failed to discover research papers'
      console.error('❌ Search failed:', errorMessage)
      
      setError(new Error(errorMessage))
      toast.error(`❌ ${errorMessage}`, { duration: 6000 })
      
      return null
    } finally {
      setLoading(false)
    }
  }, [transformBackendData])

  // **FIXED: Get node details with proper API call**
  const getNodeDetails = useCallback(async (nodeId) => {
    if (!nodeId) return null

    try {
      setLoadingDetails(true)
      
      console.log('📄 Fetching details for paper ID:', nodeId)
      
      // Try local data first
      if (data?.nodes) {
        const node = data.nodes.find(n => String(n.id) === String(nodeId))
        if (node) {
          console.log('📋 Found node in local data')
          const details = transformNodeToDetails(node)
          setNodeDetails(details)
          return details
        }
      }

      // Try API call
      try {
        const apiDetails = await researchAPI.getPaperDetails(nodeId)
        
        if (apiDetails) {
          console.log('📄✅ Paper details loaded from API')
          const details = apiDetails.paper || apiDetails.data || apiDetails
          setNodeDetails(details)
          toast.success('📄 Paper details loaded', { duration: 2000 })
          return details
        }
      } catch (apiError) {
        console.warn('📄❌ API call failed:', apiError.message)
      }

      toast.warning('Paper details not found', { duration: 3000 })
      return null

    } catch (err) {
      console.warn('Paper details error:', err.message)
      toast.error(`Failed to load paper details: ${err.message}`, { duration: 3000 })
      return null
    } finally {
      setLoadingDetails(false)
    }
  }, [data])

  // Clear all data
  const clearData = useCallback(() => {
    setData(null)
    setNodeDetails(null)
    setStatistics(null)
    setError(null)
    cacheRef.current.clear()
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    
    toast.success('🔄 Data cleared', { duration: 2000 })
  }, [])

  return {
    data,
    loading,
    error,
    nodeDetails,
    loadingDetails,
    statistics,
    searchPapers,
    getNodeDetails,
    clearData
  }
}

// Helper functions (same as before)
const calculateNodeSize = (paper) => {
  const citations = Number(paper.citation_count || 0)
  const quality = Number(paper.context_quality_score || paper.quality_score || 0.5)
  
  const citationFactor = Math.log(citations + 1) * 2
  const qualityFactor = quality * 15
  
  return Math.max(15, Math.min(40, citationFactor + qualityFactor))
}

const getDomainColor = (domain) => {
  const colors = {
    'Machine Learning': '#3B82F6',
    'Computer Vision': '#8B5CF6',
    'Healthcare': '#10B981',
    'Healthcare Informatics': '#06B6D4',
    'Biomedical Engineering, Machine Learning, Nanotechnology': '#EC4899',
    'Medical Imaging': '#F59E0B',
    'Climate Science': '#F59E0B',
    'Physics': '#EF4444',
    'Biology': '#84CC16',
    'Chemistry': '#EC4899',
    'Mathematics': '#F97316',
    'Robotics': '#A855F7',
    'General Research': '#6B7280',
    'Unknown': '#6B7280'
  }
  return colors[domain] || colors['General Research']
}

const truncateText = (text, maxLength) => {
  if (!text || text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

const transformNodeToDetails = (node) => ({
  id: node.id,
  title: node.title,
  abstract: node.abstract || node.context_summary || 'No abstract available',
  authors: node.authors,
  published_date: node.published_date,
  research_domain: node.research_domain,
  citation_count: node.citation_count,
  quality_score: node.quality_score,
  paper_url: node.paper_url,
  methodology: node.methodology,
  key_findings: node.key_findings || [],
  innovations: node.innovations || [],
  contributions: node.contributions || []
})

const calculateStatistics = (graphData) => {
  if (!graphData?.nodes?.length) {
    return {
      totalPapers: 0,
      totalConnections: 0,
      avgQuality: 0,
      uniqueDomains: 0,
      topDomain: 'Unknown'
    }
  }

  const nodes = graphData.nodes
  const links = graphData.links || []

  const qualityScores = nodes.map(n => n.quality_score || 0).filter(q => q > 0)
  const avgQuality = qualityScores.length > 0 
    ? qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length 
    : 0

  const domains = nodes.map(n => n.research_domain).filter(d => d && d !== 'Unknown')
  const uniqueDomains = [...new Set(domains)].length
  
  const domainCounts = domains.reduce((acc, domain) => {
    acc[domain] = (acc[domain] || 0) + 1
    return acc
  }, {})
  
  const topDomain = Object.entries(domainCounts)
    .sort(([,a], [,b]) => b - a)[0]?.[0] || 'General Research'

  return {
    totalPapers: nodes.length,
    totalConnections: links.length,
    avgQuality,
    uniqueDomains,
    topDomain,
    connectionDensity: links.length / Math.max(1, nodes.length)
  }
}
