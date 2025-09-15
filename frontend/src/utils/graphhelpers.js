/**
 * Graph utility functions for research network visualization
 */

// Node positioning algorithms
export const layoutAlgorithms = {
  force: 'force-directed',
  circular: 'circular',
  hierarchical: 'hierarchical',
  grid: 'grid'
}

// Color schemes for different research domains
export const domainColorSchemes = {
  'Machine Learning': {
    primary: '#3B82F6',
    secondary: '#1E40AF',
    gradient: 'linear-gradient(135deg, #3B82F6, #1E40AF)'
  },
  'Computer Vision': {
    primary: '#8B5CF6',
    secondary: '#7C3AED',
    gradient: 'linear-gradient(135deg, #8B5CF6, #7C3AED)'
  },
  'Healthcare': {
    primary: '#10B981',
    secondary: '#059669',
    gradient: 'linear-gradient(135deg, #10B981, #059669)'
  },
  'Climate Science': {
    primary: '#F59E0B',
    secondary: '#D97706',
    gradient: 'linear-gradient(135deg, #F59E0B, #D97706)'
  },
  'Physics': {
    primary: '#EF4444',
    secondary: '#DC2626',
    gradient: 'linear-gradient(135deg, #EF4444, #DC2626)'
  },
  'Chemistry': {
    primary: '#06B6D4',
    secondary: '#0891B2',
    gradient: 'linear-gradient(135deg, #06B6D4, #0891B2)'
  },
  'Biology': {
    primary: '#84CC16',
    secondary: '#65A30D',
    gradient: 'linear-gradient(135deg, #84CC16, #65A30D)'
  },
  'Mathematics': {
    primary: '#F97316',
    secondary: '#EA580C',
    gradient: 'linear-gradient(135deg, #F97316, #EA580C)'
  },
  'default': {
    primary: '#64748B',
    secondary: '#475569',
    gradient: 'linear-gradient(135deg, #64748B, #475569)'
  }
}

/**
 * Calculate node size based on various metrics
 */
export const calculateNodeSize = (node, config = {}) => {
  const {
    minSize = 4,
    maxSize = 20,
    sizeFactor = 'citation_count',
    logarithmic = true
  } = config

  let value = node[sizeFactor] || 1
  
  if (logarithmic) {
    value = Math.log(value + 1)
  }
  
  const normalizedSize = Math.max(minSize, Math.min(maxSize, value * 2))
  return normalizedSize
}

/**
 * Get color for a node based on its domain
 */
export const getNodeColor = (node, selectedNodeId = null) => {
  const scheme = domainColorSchemes[node.research_domain] || domainColorSchemes.default
  
  if (node.id === selectedNodeId) {
    return '#FBBF24' // Gold for selected node
  }
  
  return scheme.primary
}

/**
 * Calculate link strength and color
 */
export const calculateLinkProperties = (link, config = {}) => {
  const {
    minOpacity = 0.2,
    maxOpacity = 1.0,
    minWidth = 1,
    maxWidth = 5
  } = config

  const strength = link.strength || 0.5
  
  return {
    opacity: Math.max(minOpacity, strength * maxOpacity),
    width: Math.max(minWidth, strength * maxWidth),
    color: `rgba(100, 116, 139, ${Math.max(minOpacity, strength)})`
  }
}

/**
 * Filter graph data based on criteria
 */
export const filterGraphData = (data, filters = {}) => {
  if (!data || !data.nodes || !data.links) return data

  const {
    minCitations = 0,
    minQuality = 0,
    hiddenDomains = [],
    yearRange = null,
    authorFilter = null
  } = filters

  // Filter nodes
  const filteredNodes = data.nodes.filter(node => {
    // Citation filter
    if ((node.citation_count || 0) < minCitations) return false
    
    // Quality filter
    if ((node.quality_score || 0) < minQuality) return false
    
    // Domain filter
    if (hiddenDomains.includes(node.research_domain)) return false
    
    // Year filter
    if (yearRange && node.published_date) {
      const year = new Date(node.published_date).getFullYear()
      if (year < yearRange.start || year > yearRange.end) return false
    }
    
    // Author filter
    if (authorFilter && node.authors) {
      const authorMatch = node.authors.toLowerCase().includes(authorFilter.toLowerCase())
      if (!authorMatch) return false
    }
    
    return true
  })

  // Get IDs of filtered nodes
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
  
  // Filter links to only include connections between remaining nodes
  const filteredLinks = data.links.filter(link =>
    filteredNodeIds.has(link.source) && filteredNodeIds.has(link.target)
  )

  return {
    nodes: filteredNodes,
    links: filteredLinks
  }
}

/**
 * Calculate graph statistics
 */
export const calculateGraphStatistics = (data) => {
  if (!data || !data.nodes || !data.links) {
    return {
      totalNodes: 0,
      totalLinks: 0,
      avgQuality: 0,
      avgCitations: 0,
      domains: [],
      topAuthors: [],
      yearRange: null
    }
  }

  const nodes = data.nodes
  const links = data.links

  // Basic counts
  const totalNodes = nodes.length
  const totalLinks = links.length

  // Quality metrics
  const qualityScores = nodes.map(n => n.quality_score || 0).filter(q => q > 0)
  const avgQuality = qualityScores.length > 0 
    ? qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length 
    : 0

  // Citation metrics
  const citations = nodes.map(n => n.citation_count || 0)
  const avgCitations = citations.length > 0
    ? citations.reduce((sum, c) => sum + c, 0) / citations.length
    : 0
  const maxCitations = Math.max(...citations)

  // Domain analysis
  const domainCounts = nodes.reduce((acc, node) => {
    const domain = node.research_domain || 'Unknown'
    acc[domain] = (acc[domain] || 0) + 1
    return acc
  }, {})

  const domains = Object.entries(domainCounts)
    .sort(([,a], [,b]) => b - a)
    .map(([domain, count]) => ({ domain, count, percentage: (count / totalNodes) * 100 }))

  // Author analysis
  const authorCounts = nodes.reduce((acc, node) => {
    if (node.authors) {
      const authors = node.authors.split(';').map(a => a.trim())
      authors.forEach(author => {
        acc[author] = (acc[author] || 0) + 1
      })
    }
    return acc
  }, {})

  const topAuthors = Object.entries(authorCounts)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 10)
    .map(([author, count]) => ({ author, count }))

  // Year analysis
  const years = nodes
    .map(n => n.published_date ? new Date(n.published_date).getFullYear() : null)
    .filter(y => y !== null)
    .sort((a, b) => a - b)

  const yearRange = years.length > 0 ? {
    start: years[0],
    end: years[years.length - 1]
  } : null

  // Network topology metrics
  const nodeDegrees = nodes.map(node => {
    return links.filter(link => 
      link.source === node.id || link.target === node.id
    ).length
  })

  const avgDegree = nodeDegrees.length > 0
    ? nodeDegrees.reduce((sum, d) => sum + d, 0) / nodeDegrees.length
    : 0

  const maxDegree = Math.max(...nodeDegrees)

  // Clustering coefficient (simplified)
  const density = totalNodes > 1 
    ? (2 * totalLinks) / (totalNodes * (totalNodes - 1))
    : 0

  return {
    totalNodes,
    totalLinks,
    avgQuality,
    avgCitations,
    maxCitations,
    domains,
    topAuthors,
    yearRange,
    avgDegree,
    maxDegree,
    networkDensity: density,
    topDomain: domains[0]?.domain || 'Unknown'
  }
}

/**
 * Find shortest path between two nodes
 */
export const findShortestPath = (data, sourceId, targetId) => {
  if (!data || !data.nodes || !data.links) return null

  // Build adjacency list
  const graph = {}
  data.nodes.forEach(node => {
    graph[node.id] = []
  })

  data.links.forEach(link => {
    graph[link.source].push(link.target)
    graph[link.target].push(link.source)
  })

  // BFS to find shortest path
  const queue = [{ node: sourceId, path: [sourceId] }]
  const visited = new Set([sourceId])

  while (queue.length > 0) {
    const { node, path } = queue.shift()

    if (node === targetId) {
      return path
    }

    const neighbors = graph[node] || []
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor)
        queue.push({
          node: neighbor,
          path: [...path, neighbor]
        })
      }
    }
  }

  return null // No path found
}

/**
 * Get node neighbors and their relationships
 */
export const getNodeNeighbors = (data, nodeId, depth = 1) => {
  if (!data || !data.nodes || !data.links) return { nodes: [], links: [] }

  const visited = new Set()
  const resultNodes = new Set()
  const resultLinks = []

  const traverse = (currentId, currentDepth) => {
    if (currentDepth > depth || visited.has(currentId)) return
    
    visited.add(currentId)
    resultNodes.add(currentId)

    const connectedLinks = data.links.filter(link =>
      link.source === currentId || link.target === currentId
    )

    connectedLinks.forEach(link => {
      const neighborId = link.source === currentId ? link.target : link.source
      resultLinks.push(link)
      
      if (currentDepth < depth) {
        traverse(neighborId, currentDepth + 1)
      } else {
        resultNodes.add(neighborId)
      }
    })
  }

  traverse(nodeId, 0)

  const neighborNodes = data.nodes.filter(node => resultNodes.has(node.id))

  return {
    nodes: neighborNodes,
    links: resultLinks
  }
}

/**
 * Export graph data in various formats
 */
export const exportGraphData = (data, format = 'json') => {
  if (!data) return null

  switch (format) {
    case 'json':
      return JSON.stringify(data, null, 2)

    case 'csv-nodes':
      if (!data.nodes) return null
      const nodeHeaders = ['id', 'title', 'authors', 'research_domain', 'citation_count', 'quality_score', 'published_date']
      const nodeRows = data.nodes.map(node => 
        nodeHeaders.map(header => node[header] || '').join(',')
      )
      return [nodeHeaders.join(','), ...nodeRows].join('\n')

    case 'csv-links':
      if (!data.links) return null
      const linkHeaders = ['source', 'target', 'strength', 'type']
      const linkRows = data.links.map(link =>
        linkHeaders.map(header => link[header] || '').join(',')
      )
      return [linkHeaders.join(','), ...linkRows].join('\n')

    case 'gexf':
      // GEXF format for Gephi
      return generateGEXF(data)

    case 'graphml':
      // GraphML format
      return generateGraphML(data)

    default:
      return JSON.stringify(data, null, 2)
  }
}

/**
 * Generate GEXF format for Gephi
 */
const generateGEXF = (data) => {
  const nodes = data.nodes || []
  const links = data.links || []

  let gexf = `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <graph mode="static" defaultedgetype="undirected">
    <attributes class="node">
      <attribute id="0" title="domain" type="string"/>
      <attribute id="1" title="citations" type="integer"/>
      <attribute id="2" title="quality" type="float"/>
    </attributes>
    <nodes>
`

  nodes.forEach(node => {
    gexf += `      <node id="${node.id}" label="${node.title || node.label || ''}">
        <attvalues>
          <attvalue for="0" value="${node.research_domain || ''}"/>
          <attvalue for="1" value="${node.citation_count || 0}"/>
          <attvalue for="2" value="${node.quality_score || 0}"/>
        </attvalues>
      </node>
`
  })

  gexf += `    </nodes>
    <edges>
`

  links.forEach((link, index) => {
    gexf += `      <edge id="${index}" source="${link.source}" target="${link.target}" weight="${link.strength || 1}"/>
`
  })

  gexf += `    </edges>
  </graph>
</gexf>`

  return gexf
}

/**
 * Generate GraphML format
 */
const generateGraphML = (data) => {
  const nodes = data.nodes || []
  const links = data.links || []

  let graphml = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
                             http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="title" attr.type="string"/>
  <key id="d1" for="node" attr.name="domain" attr.type="string"/>
  <key id="d2" for="node" attr.name="citations" attr.type="int"/>
  <key id="d3" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="undirected">
`

  nodes.forEach(node => {
    graphml += `    <node id="${node.id}">
      <data key="d0">${node.title || node.label || ''}</data>
      <data key="d1">${node.research_domain || ''}</data>
      <data key="d2">${node.citation_count || 0}</data>
    </node>
`
  })

  links.forEach((link, index) => {
    graphml += `    <edge id="e${index}" source="${link.source}" target="${link.target}">
      <data key="d3">${link.strength || 1}</data>
    </edge>
`
  })

  graphml += `  </graph>
</graphml>`

  return graphml
}

/**
 * Validate graph data structure
 */
export const validateGraphData = (data) => {
  const errors = []
  const warnings = []

  if (!data) {
    errors.push('Graph data is null or undefined')
    return { isValid: false, errors, warnings }
  }

  if (!data.nodes || !Array.isArray(data.nodes)) {
    errors.push('Graph data must have a nodes array')
  } else if (data.nodes.length === 0) {
    warnings.push('Graph has no nodes')
  }

  if (!data.links || !Array.isArray(data.links)) {
    errors.push('Graph data must have a links array')
  } else if (data.links.length === 0) {
    warnings.push('Graph has no links')
  }

  // Validate node structure
  if (data.nodes) {
    const nodeIds = new Set()
    data.nodes.forEach((node, index) => {
      if (!node.id) {
        errors.push(`Node at index ${index} missing required 'id' field`)
      } else if (nodeIds.has(node.id)) {
        errors.push(`Duplicate node ID: ${node.id}`)
      } else {
        nodeIds.add(node.id)
      }
    })

    // Validate link references
    if (data.links) {
      data.links.forEach((link, index) => {
        if (!link.source || !link.target) {
          errors.push(`Link at index ${index} missing source or target`)
        } else {
          if (!nodeIds.has(link.source)) {
            errors.push(`Link references non-existent source node: ${link.source}`)
          }
          if (!nodeIds.has(link.target)) {
            errors.push(`Link references non-existent target node: ${link.target}`)
          }
        }
      })
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}
