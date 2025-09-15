import { useState, useCallback, useRef } from 'react'

export const useGraphVisualization = (data) => {
  const [graphConfig, setGraphConfig] = useState({
    nodeSize: 5,
    linkWidth: 1,
    linkOpacity: 0.6,
    linkDistance: 200,
    nodeRepulsion: 800,
    simulationSpeed: 1,
    showLabels: true,
    paused: false,
    minCitations: 0,
    minQuality: 0,
    hiddenDomains: []
  })

  const graphRef = useRef()
  const cameraPositionRef = useRef()

  const updateGraphConfig = useCallback((newConfig) => {
    setGraphConfig(prev => ({ ...prev, ...newConfig }))
  }, [])

  const resetView = useCallback(() => {
    if (graphRef.current) {
      // Reset camera position for 3D
      if (graphRef.current.cameraPosition) {
        graphRef.current.cameraPosition({ x: 0, y: 0, z: 1000 }, 2000)
      }
      // Fit graph to view
      if (graphRef.current.zoomToFit) {
        graphRef.current.zoomToFit(2000)
      }
    }
  }, [])

  const focusNode = useCallback((nodeId) => {
    if (graphRef.current && nodeId) {
      const node = data?.nodes?.find(n => n.id === nodeId)
      if (node) {
        // For 3D graphs
        if (graphRef.current.cameraPosition) {
          graphRef.current.cameraPosition(
            { x: node.x, y: node.y, z: node.z + 300 },
            1500
          )
        }
        // For 2D graphs
        else if (graphRef.current.centerAt) {
          graphRef.current.centerAt(node.x, node.y, 1500)
        }
      }
    }
  }, [data])

  const exportGraph = useCallback((format = 'png') => {
    if (!graphRef.current) return

    try {
      switch (format) {
        case 'png':
          if (graphRef.current.renderer) {
            // For 3D graphs
            const canvas = graphRef.current.renderer().domElement
            const link = document.createElement('a')
            link.download = 'research-graph.png'
            link.href = canvas.toDataURL()
            link.click()
          }
          break

        case 'json':
          const dataStr = JSON.stringify(data, null, 2)
          const dataBlob = new Blob([dataStr], { type: 'application/json' })
          const url = URL.createObjectURL(dataBlob)
          const link = document.createElement('a')
          link.download = 'research-graph.json'
          link.href = url
          link.click()
          URL.revokeObjectURL(url)
          break

        case 'svg':
          // For 2D graphs
          if (graphRef.current.getGraphBbox) {
            // Implementation would depend on the specific graph library
            console.log('SVG export not implemented yet')
          }
          break

        default:
          console.warn('Unsupported export format:', format)
      }
    } catch (error) {
      console.error('Export failed:', error)
    }
  }, [data])

  const getFilteredData = useCallback(() => {
    if (!data) return null

    const filteredNodes = data.nodes.filter(node => {
      // Citation filter
      if ((node.citation_count || 0) < graphConfig.minCitations) return false
      
      // Quality filter
      if ((node.quality_score || 0) < graphConfig.minQuality) return false
      
      // Domain filter
      if (graphConfig.hiddenDomains.includes(node.research_domain)) return false
      
      return true
    })

    const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
    
    const filteredLinks = data.links.filter(link =>
      filteredNodeIds.has(link.source) && filteredNodeIds.has(link.target)
    )

    return {
      nodes: filteredNodes,
      links: filteredLinks
    }
  }, [data, graphConfig])

  return {
    graphConfig,
    updateGraphConfig,
    resetView,
    focusNode,
    exportGraph,
    getFilteredData,
    graphRef
  }
}
