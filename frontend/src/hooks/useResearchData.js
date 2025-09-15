import { useState, useCallback, useRef, useMemo } from 'react'
import { toast } from 'react-hot-toast'
import { researchAPI } from '../utils/api'

export const useResearchData = () => {
  const [rawData, setRawData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [nodeDetails, setNodeDetails] = useState(null)
  const [loadingDetails, setLoadingDetails] = useState(false)
  const [statistics, setStatistics] = useState(null)
  
  const abortControllerRef = useRef(null)
  const stableDataRef = useRef(null) // 🔧 STABLE REFERENCE

  // 🔧 STABLE DATA: Only recreate when rawData actually changes
  const data = useMemo(() => {
    if (!rawData || !rawData.nodes || !rawData.links) {
      stableDataRef.current = null
      return null
    }
    
    // 🔧 CHECK: Only recreate if rawData timestamp changed
    if (stableDataRef.current && 
        stableDataRef.current._originalTimestamp === rawData._originalTimestamp) {
      console.log('♻️ Returning existing stable data (no changes)')
      return stableDataRef.current
    }
    
    console.log('🔄 Creating NEW stable graph data')
    
    try {
      const stableData = {
        nodes: rawData.nodes.map((node, index) => ({
          // Create stable node object
          id: String(node.id),
          title: node.title || `Paper ${index + 1}`,
          label: node.label || truncateText(node.title || `Paper ${index + 1}`, 50),
          size: Math.max(10, Number(node.size) || 15),
          color: node.color || getDomainColor(node.research_domain),
          quality_score: Number(node.quality_score) || 0.5,
          research_domain: node.research_domain || 'Unknown',
          context_summary: node.context_summary || '',
          methodology: node.methodology || '',
          innovations: Array.isArray(node.innovations) ? [...node.innovations] : [],
          contributions: Array.isArray(node.contributions) ? [...node.contributions] : [],
          authors: node.authors || 'Unknown Authors',
          published_date: node.published_date || '',
          citation_count: Number(node.citation_count) || 0,
          paper_url: node.paper_url || '',
          pdf_url: node.pdf_url || '',
          abstract: node.abstract || node.context_summary || '',
          language: node.language || 'en',
          source: node.source || 'unknown',
          ai_agent_used: node.ai_agent_used || '',
          analysis_confidence: Number(node.analysis_confidence) || 0.5,
          key_findings: Array.isArray(node.key_findings) ? [...node.key_findings] : [],
          limitations: Array.isArray(node.limitations) ? [...node.limitations] : [],
          future_work: Array.isArray(node.future_work) ? [...node.future_work] : []
        })),
        links: rawData.links.map(link => ({
          source: String(link.source),
          target: String(link.target),
          strength: Number(link.strength || 0.5),
          weight: Number(link.weight || 0.5),
          type: link.type || 'related',
          context: link.context || 'Related research',
          reasoning: link.reasoning || 'Semantic similarity',
          confidence: Number(link.confidence || 0.5)
        })),
        _originalTimestamp: rawData._originalTimestamp,
        _stableVersion: 1
      }
      
      // Validate data
      const nodeIds = new Set(stableData.nodes.map(n => n.id))
      const validLinks = stableData.links.filter(link => 
        nodeIds.has(link.source) && nodeIds.has(link.target)
      )
      
      const finalData = {
        ...stableData,
        links: validLinks
      }
      
      console.log('✅ STABLE data created:', {
        nodes: finalData.nodes.length,
        links: finalData.links.length,
        timestamp: finalData._originalTimestamp,
        version: finalData._stableVersion
      })
      
      // Cache the stable data
      stableDataRef.current = finalData
      return finalData
      
    } catch (error) {
      console.error('❌ Failed to create stable data:', error)
      return null
    }
  }, [rawData?._originalTimestamp]) // 🔧 ONLY depend on timestamp, not the whole object

  const searchPapers = useCallback(async (query, filters = {}) => {
    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }

      abortControllerRef.current = new AbortController()
      
      setLoading(true)
      setError(null)
      
      toast.loading('🧠 Building research knowledge graph...', { id: 'search' })

      const result = await researchAPI.discover(query, {
        maxResults: 50,
        enableMultilingual: true,
        sources: ["core", "arxiv", "pubmed"],
        analysisDepth: 'detailed',
        enableGraph: true,
        ...filters
      })

      console.log('🔍 RAW BACKEND RESPONSE:', result.data?.papers?.length, 'papers')

      const transformedData = transformTiDBBackendData(result.data)
      
      if (transformedData.links.length === 0) {
        console.warn('⚠️ No links found in transformed data!')
      }
      
      console.log('📊 STORING RAW DATA with timestamp')
      
      // 🔧 CRITICAL: Add unique timestamp to force single update
      const timestampedData = {
        nodes: transformedData.nodes.map(node => ({ ...node })),
        links: transformedData.links.map(link => ({ ...link })),
        _originalTimestamp: Date.now() // This will trigger exactly one update
      }
      
      setRawData(timestampedData)
      
      const stats = calculateStatistics(transformedData)
      setStatistics(stats)
      
      toast.success(`✅ Built graph: ${transformedData.nodes.length} papers, ${transformedData.links.length} connections`, { id: 'search' })
      
      return transformedData
      
    } catch (err) {
      if (err.name !== 'AbortError') {
        const errorMessage = err.response?.data?.error || err.message || 'Failed to discover research papers'
        setError(new Error(errorMessage))
        toast.error(errorMessage, { id: 'search' })
        console.error('Search error:', err)
      }
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  // Enhanced paper details fetching
  const getNodeDetails = useCallback(async (nodeId) => {
    try {
      setLoadingDetails(true)
      
      toast.loading('📄 Loading paper details...', { id: 'paper-details' })
      
      console.log('🔍 FETCHING paper details for ID:', nodeId)
      
      // Try API endpoints
      const endpoints = [
        `http://localhost:8000/api/papers/details/${nodeId}`,
        `http://localhost:8000/api/papers/${nodeId}`,
        `http://localhost:8000/api/research/papers/${nodeId}`,
        `http://localhost:8000/papers/${nodeId}`
      ]
      
      for (const endpoint of endpoints) {
        try {
          console.log(`🌐 Trying: ${endpoint}`)
          
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 5000)
          
          const response = await fetch(endpoint, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            signal: controller.signal
          })
          
          clearTimeout(timeoutId)
          
          console.log(`📡 Response: ${response.status}`)
          
          if (response.ok) {
            const result = await response.json()
            
            if (result && (result.paper || result.data || result.id || result.title)) {
              const paperDetails = result.paper || result.data || result
              setNodeDetails(paperDetails)
              toast.success('Paper details loaded from API', { id: 'paper-details' })
              return paperDetails
            }
          }
        } catch (apiError) {
          console.warn(`❌ ${endpoint} failed:`, apiError.message)
          continue
        }
      }
      
      // Fallback to cached data  
      console.log('🔄 Using cached data fallback')
      
      const dataSources = [
        { name: 'rawData', data: rawData },
        { name: 'stableData', data: stableDataRef.current }
      ]
      
      for (const { name, data: sourceData } of dataSources) {
        if (sourceData && sourceData.nodes) {
          const node = sourceData.nodes.find(n => String(n.id) === String(nodeId))
          
          if (node) {
            console.log(`✅ Found node in ${name}:`, node.id)
            const paperDetails = transformNodeToPaperDetails(node)
            setNodeDetails(paperDetails)
            toast.success(`Paper details loaded from ${name}`, { id: 'paper-details' })
            return paperDetails
          }
        }
      }
      
      throw new Error(`Paper with ID ${nodeId} not found`)
      
    } catch (err) {
      console.error('❌ Paper fetch failed:', err.message)
      toast.error(`Failed to load paper: ${err.message}`, { id: 'paper-details' })
      return null
    } finally {
      setLoadingDetails(false)
    }
  }, [rawData])

  const clearData = useCallback(() => {
    setRawData(null)
    setNodeDetails(null)
    setStatistics(null)
    setError(null)
    stableDataRef.current = null // Clear stable reference
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
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

// Helper functions remain the same...
const transformTiDBBackendData = (backendData) => {
  console.log('🔄 TRANSFORMING TiDB BACKEND DATA')
  
  if (!backendData) {
    return { nodes: [], links: [] }
  }

  const papers = backendData.papers || []
  const relationships = backendData.relationships || []
  const graph = backendData.graph || {}

  console.log('📊 Data sources:', {
    papers: papers.length,
    relationships: relationships.length,
    graphNodes: graph.nodes?.length || 0,
    graphEdges: graph.edges?.length || 0
  })

  let nodes = []
  let links = []

  if (graph.nodes && graph.nodes.length > 0) {
    console.log('✅ Using graph nodes:', graph.nodes.length)
    nodes = graph.nodes.map((node, index) => ({
      id: String(node.id),
      label: truncateText(node.label || node.title || `Paper ${index + 1}`, 50),
      title: node.title || `Paper ${index + 1}`,
      research_domain: node.research_domain || 'Unknown',
      context_summary: node.context_summary || '',
      methodology: node.methodology || '',
      innovations: Array.isArray(node.innovations) ? [...node.innovations] : [],
      contributions: Array.isArray(node.contributions) ? [...node.contributions] : [],
      quality_score: Number(node.quality_score || 0.5),
      authors: node.authors || 'Unknown Authors',
      published_date: node.published_date || '',
      citation_count: Number(node.citation_count || 0),
      paper_url: node.paper_url || '',
      pdf_url: node.pdf_url || '',
      size: Math.max(10, Number(node.size || 15) * 0.8),
      color: node.color || getDomainColor(node.research_domain),
      abstract: node.abstract || node.context_summary || '',
      language: node.language || 'en',
      source: node.source || 'unknown',
      ai_agent_used: node.ai_agent_used || '',
      analysis_confidence: Number(node.analysis_confidence || 0.5)
    }))
  } else {
    console.log('✅ Using papers:', papers.length)
    nodes = papers.map((paper, index) => ({
      id: String(paper.id),
      label: truncateText(paper.title || `Paper ${index + 1}`, 50),
      title: paper.title || `Paper ${index + 1}`,
      research_domain: paper.research_domain || 'Unknown',
      context_summary: paper.context_summary || '',
      methodology: paper.methodology || '',
      innovations: Array.isArray(paper.innovations) ? [...paper.innovations] : [],
      contributions: Array.isArray(paper.contributions) ? [...paper.contributions] : [],
      quality_score: Number(paper.context_quality_score || 0.5),
      authors: paper.authors || 'Unknown Authors',
      published_date: paper.published_date || '',
      citation_count: Number(paper.citation_count || 0),
      paper_url: paper.paper_url || '',
      pdf_url: paper.pdf_url || '',
      size: Math.max(10, calculateNodeSize(paper) * 0.8),
      color: getDomainColor(paper.research_domain),
      abstract: paper.abstract || paper.context_summary || '',
      language: paper.language || 'en',
      source: paper.source || 'unknown',
      ai_agent_used: paper.ai_agent_used || '',
      analysis_confidence: Number(paper.analysis_confidence || 0.5),
      key_findings: Array.isArray(paper.key_findings) ? [...paper.key_findings] : [],
      limitations: Array.isArray(paper.limitations) ? [...paper.limitations] : [],
      future_work: Array.isArray(paper.future_work) ? [...paper.future_work] : []
    }))
  }

  if (graph.edges && graph.edges.length > 0) {
    console.log('✅ Using graph edges:', graph.edges.length)
    links = graph.edges.map(edge => ({
      source: String(edge.source),
      target: String(edge.target),
      strength: Number(edge.strength || 0.5),
      type: edge.relationship_type || 'related',
      weight: Math.max(0.5, Number(edge.strength || 0.5) * 2),
      context: edge.context || 'Related research',
      reasoning: edge.reasoning || 'Semantic similarity',
      confidence: Number(edge.confidence_score || 0.5)
    }))
  } else if (relationships && relationships.length > 0) {
    console.log('✅ Using relationships:', relationships.length)
    links = relationships.map(rel => ({
      source: String(rel.paper1_id),
      target: String(rel.paper2_id),
      strength: Number(rel.relationship_strength || 0.5),
      type: rel.relationship_type || 'related',
      weight: Math.max(0.5, Number(rel.relationship_strength || 0.5) * 2),
      context: rel.relationship_context || 'Related research',
      reasoning: rel.connection_reasoning || 'Semantic similarity',
      confidence: Number(rel.confidence_score || 0.5),
      semantic_similarity: Number(rel.semantic_similarity || 0),
      domain_overlap: Number(rel.domain_overlap || 0)
    }))
  }

  // Validate links
  const nodeIds = new Set(nodes.map(n => n.id))
  const validLinks = links.filter(link => {
    const sourceExists = nodeIds.has(link.source)
    const targetExists = nodeIds.has(link.target)
    return sourceExists && targetExists
  })

  console.log('🎯 Final transformation result:', {
    nodes: nodes.length,
    validLinks: validLinks.length,
    invalidLinks: links.length - validLinks.length
  })

  return { 
    nodes, 
    links: validLinks 
  }
}

const transformNodeToPaperDetails = (node) => {
  return {
    id: node.id,
    title: node.title || 'Untitled Paper',
    abstract: node.abstract || node.context_summary || 'No abstract available',
    authors: node.authors || 'Unknown Authors',
    published_date: node.published_date || 'Unknown Date',
    research_domain: node.research_domain || 'Unknown Domain',
    methodology: node.methodology || 'Not specified',
    key_findings: node.key_findings || [],
    innovations: node.innovations || [],
    contributions: node.contributions || [],
    limitations: node.limitations || [],
    future_work: node.future_work || [],
    context_summary: node.context_summary || 'No summary available',
    context_quality_score: node.quality_score || 0.5,
    paper_url: node.paper_url || '',
    pdf_url: node.pdf_url || '',
    citation_count: node.citation_count || 0,
    doi: node.doi || '',
    language: node.language || 'en',
    source: node.source || 'unknown',
    ai_agent_used: node.ai_agent_used || '',
    analysis_confidence: node.analysis_confidence || 0.5
  }
}

const calculateNodeSize = (paper) => {
  const citationCount = paper.citation_count || 0
  const qualityScore = paper.context_quality_score || 0.5
  
  const citationFactor = Math.log(citationCount + 1) * 2
  const qualityFactor = qualityScore * 15
  
  return Math.max(15, Math.min(35, citationFactor + qualityFactor))
}

const getDomainColor = (domain) => {
  const colors = {
    'Machine Learning': '#3B82F6',
    'Computer Vision': '#8B5CF6', 
    'Healthcare': '#10B981',
    'Healthcare Informatics': '#06B6D4',
    'Climate Science': '#F59E0B',
    'Physics': '#EF4444',
    'Chemistry': '#EC4899',
    'Biology': '#84CC16',
    'Mathematics': '#F97316',
    'Neuroscience': '#A855F7',
    'Psychiatry': '#DB2777',
    'General Research': '#6B7280',
    'Unknown': '#6B7280',
    'default': '#64748B'
  }
  return colors[domain] || colors.default
}

const truncateText = (text, maxLength) => {
  if (!text) return 'Untitled'
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text
}

const calculateStatistics = (graphData) => {
  if (!graphData || !graphData.nodes) {
    return {
      totalPapers: 0,
      totalConnections: 0,
      avgQuality: 0,
      uniqueDomains: 0,
      avgCitations: 0
    }
  }

  const nodes = graphData.nodes
  const links = graphData.links || []

  const totalPapers = nodes.length
  const totalConnections = links.length
  
  const qualityScores = nodes.map(n => n.quality_score || 0).filter(q => q > 0)
  const avgQuality = qualityScores.length > 0 
    ? qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length 
    : 0

  const citations = nodes.map(n => n.citation_count || 0)
  const avgCitations = citations.length > 0
    ? citations.reduce((sum, c) => sum + c, 0) / citations.length
    : 0

  const uniqueDomains = [...new Set(nodes.map(n => n.research_domain))].filter(d => d).length

  return {
    totalPapers,
    totalConnections,
    avgQuality,
    uniqueDomains,
    avgCitations,
    topDomain: getMostFrequentDomain(nodes),
    connectionDensity: totalConnections / Math.max(1, totalPapers)
  }
}

const getMostFrequentDomain = (nodes) => {
  const domainCounts = nodes.reduce((acc, node) => {
    const domain = node.research_domain || 'Unknown'
    acc[domain] = (acc[domain] || 0) + 1
    return acc
  }, {})
  
  return Object.entries(domainCounts).sort(([,a], [,b]) => b - a)[0]?.[0] || 'Unknown'
}
