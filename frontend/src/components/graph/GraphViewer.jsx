import React, { forwardRef, useEffect, useRef, useState, useCallback, memo, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ForceGraph2D from 'react-force-graph-2d'

const GraphViewer = memo(forwardRef(({
  data,
  mode = '2d',
  config = {},
  selectedNode,
  hoveredNode,
  onNodeInteraction,
  onConfigChange
}, ref) => {
  const graphRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, node: null })
  const forceSetupDone = useRef(false)

  // Enhanced default config
  const defaultConfig = {
    nodeSize: 1.0,
    linkWidth: 2,
    linkOpacity: 0.8,
    showLabels: true,
    paused: false,
    ...config
  }

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setDimensions({
          width: rect.width,
          height: rect.height
        })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Expose graph methods to parent
  useEffect(() => {
    if (ref) {
      ref.current = graphRef.current
    }
  }, [ref])

  // 🔧 STABLE force configuration
  useEffect(() => {
    if (graphRef.current && data && data.nodes && data.nodes.length > 0 && data.links && !forceSetupDone.current) {
      console.log('🎯 Setting up STABLE graph forces for', data.nodes.length, 'nodes and', data.links.length, 'links')
      forceSetupDone.current = true
      
      const setupTimer = setTimeout(() => {
        try {
          if (graphRef.current.d3Force) {
            const nodeSize = defaultConfig.nodeSize
            const repulsionStrength = -600 * Math.max(0.5, nodeSize)
            const linkDistance = 100 * Math.max(0.8, nodeSize)
            
            graphRef.current.d3Force('charge')?.strength(repulsionStrength)
            graphRef.current.d3Force('link')?.distance(linkDistance).strength(0.6)
            graphRef.current.d3Force('center')?.strength(0.05)
            
            console.log('✅ STABLE forces applied')
          }
        } catch (error) {
          console.warn('⚠️ Force setup error:', error)
        }
      }, 500)

      return () => {
        clearTimeout(setupTimer)
        forceSetupDone.current = false
      }
    }
  }, [data?._stableVersion, data?._originalTimestamp, defaultConfig.nodeSize]) // 🔧 FIXED: Use both version indicators

  // Fixed node click handler
  const handleNodeClick = useCallback((node, event) => {
    if (node && node.id) {
      console.log('✅ Node clicked:', node.id, node.title?.substring(0, 30) + '...')
      onNodeInteraction(node.id, 'click')
    } else {
      console.log('🚫 Background clicked')
      onNodeInteraction(null, 'click')
    }
  }, [onNodeInteraction])

  // Fixed node hover handler
  const handleNodeHover = useCallback((node, prevNode) => {
    if (node && node.id) {
      setTooltip({
        show: true,
        x: 0,
        y: 0,
        node: node
      })
      onNodeInteraction(node.id, 'hover')
    } else {
      setTooltip({ show: false, x: 0, y: 0, node: null })
      onNodeInteraction(null, 'unhover')
    }
  }, [onNodeInteraction])

  // 🔧 SIMPLIFIED pointer area paint
  const nodePointerAreaPaint = useCallback((node, color, ctx, globalScale) => {
    if (!node || typeof node.x !== 'number' || typeof node.y !== 'number') {
      return
    }

    const baseRadius = node.size || 15
    const actualRadius = baseRadius * defaultConfig.nodeSize
    const interactionRadius = Math.max(actualRadius, 15)

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(node.x, node.y, interactionRadius, 0, 2 * Math.PI)
    ctx.fill()
  }, [defaultConfig.nodeSize])

  // 🔧 FIXED: Calculate default node size as NUMBER
  const defaultNodeSize = useMemo(() => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return 15
    }
    
    const avgSize = data.nodes.reduce((sum, node) => sum + (node.size || 15), 0) / data.nodes.length
    return avgSize * defaultConfig.nodeSize
  }, [data?.nodes, defaultConfig.nodeSize])

  // Node visual rendering
  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    if (!node || typeof node.x !== 'number' || typeof node.y !== 'number' || 
        !isFinite(node.x) || !isFinite(node.y) || isNaN(node.x) || isNaN(node.y)) {
      return
    }

    const label = node.label || node.title
    const fontSize = Math.max(10/globalScale, 6)
    const baseRadius = node.size || 15
    const actualRadius = baseRadius * defaultConfig.nodeSize
    const nodeRadius = Math.max(8, actualRadius)

    if (!isFinite(nodeRadius) || nodeRadius <= 0) {
      return
    }

    // Node color
    let nodeColor = node.color || '#64748B'
    if (node.id === selectedNode) {
      nodeColor = '#FBBF24'
    } else if (node.id === hoveredNode) {
      nodeColor = '#60A5FA'
    }
    
    // Draw main node circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI)
    ctx.fillStyle = nodeColor
    ctx.fill()

    // Border
    const borderWidth = Math.max(1, nodeRadius / 10)
    ctx.strokeStyle = '#FFFFFF'
    ctx.lineWidth = (node.id === selectedNode || node.id === hoveredNode) ? borderWidth * 2 : borderWidth
    ctx.stroke()

    // Labels
    if (defaultConfig.showLabels && globalScale > 0.3 && label) {
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`
      ctx.fillStyle = '#FFFFFF'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      
      ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'
      ctx.shadowBlur = 4
      ctx.shadowOffsetX = 1
      ctx.shadowOffsetY = 1
      
      const maxLabelLength = Math.max(15, Math.min(25, Math.floor(nodeRadius)))
      const displayLabel = label.length > maxLabelLength ? label.substring(0, maxLabelLength) + '...' : label
      const labelY = node.y + nodeRadius + fontSize + 8
      
      ctx.fillText(displayLabel, node.x, labelY)
      
      ctx.shadowColor = 'transparent'
      ctx.shadowBlur = 0
      ctx.shadowOffsetX = 0
      ctx.shadowOffsetY = 0
    }

    // Quality indicator
    if (nodeRadius > 15 && node.quality_score) {
      const qualityRadius = Math.max(3, nodeRadius * 0.25)
      const qualityX = node.x + nodeRadius * 0.7
      const qualityY = node.y - nodeRadius * 0.7
      
      ctx.beginPath()
      ctx.arc(qualityX, qualityY, qualityRadius, 0, 2 * Math.PI)
      
      if (node.quality_score > 0.7) {
        ctx.fillStyle = '#10B981'
      } else if (node.quality_score > 0.5) {
        ctx.fillStyle = '#F59E0B'
      } else {
        ctx.fillStyle = '#EF4444'
      }
      
      ctx.fill()
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }, [selectedNode, hoveredNode, defaultConfig.showLabels, defaultConfig.nodeSize])

  // 🔧 PROTECTED link rendering - prevent mutation
  const linkCanvasObject = useCallback((link, ctx) => {
    // Create a protected copy of link data to prevent mutation
    const protectedLink = {
      source: link.source,
      target: link.target,
      strength: link.strength || 0.5
    }
    
    const start = protectedLink.source
    const end = protectedLink.target

    if (!start || !end || 
        typeof start.x !== 'number' || typeof start.y !== 'number' ||
        typeof end.x !== 'number' || typeof end.y !== 'number' ||
        !isFinite(start.x) || !isFinite(start.y) || 
        !isFinite(end.x) || !isFinite(end.y)) {
      return
    }

    const strength = protectedLink.strength
    const linkOpacity = Math.max(0.4, strength * defaultConfig.linkOpacity)
    const linkWidth = Math.max(1, defaultConfig.linkWidth * Math.max(strength, 0.3))

    let strokeColor = `rgba(100, 116, 139, ${linkOpacity})`
    if (strength > 0.7) {
      strokeColor = `rgba(59, 130, 246, ${Math.min(1, linkOpacity + 0.2)})`
    } else if (strength > 0.5) {
      strokeColor = `rgba(16, 185, 129, ${linkOpacity})`
    } else if (strength > 0.3) {
      strokeColor = `rgba(245, 158, 11, ${linkOpacity})`
    }

    ctx.strokeStyle = strokeColor
    ctx.lineWidth = linkWidth
    ctx.lineCap = 'round'
    
    const startRadius = ((start.size || 15) * defaultConfig.nodeSize) + 2
    const endRadius = ((end.size || 15) * defaultConfig.nodeSize) + 2
    
    const dx = end.x - start.x
    const dy = end.y - start.y
    const distance = Math.sqrt(dx * dx + dy * dy)
    
    if (distance > startRadius + endRadius) {
      const startX = start.x + (dx / distance) * startRadius
      const startY = start.y + (dy / distance) * startRadius
      const endX = end.x - (dx / distance) * endRadius
      const endY = end.y - (dy / distance) * endRadius
      
      ctx.beginPath()
      ctx.moveTo(startX, startY)
      ctx.lineTo(endX, endY)
      ctx.stroke()
      
      if (strength > 0.6) {
        const midX = (startX + endX) / 2
        const midY = (startY + endY) / 2
        
        ctx.beginPath()
        ctx.arc(midX, midY, Math.max(2, linkWidth / 2), 0, 2 * Math.PI)
        ctx.fillStyle = strokeColor
        ctx.fill()
      }
    }
  }, [defaultConfig.linkOpacity, defaultConfig.linkWidth, defaultConfig.nodeSize])

  // Enhanced render logging
  if (data && data.nodes) {
    console.log('🎨 Graph render:', {
      nodes: data.nodes.length,
      links: data.links.length,
      stable: !!(data._stableVersion || data._originalTimestamp),
      hasStableVersion: !!data._stableVersion,
      hasTimestamp: !!data._originalTimestamp
    })
  }

  if (!data || !data.nodes || data.nodes.length === 0) {
    return (
      <motion.div
        className="graph-empty"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="empty-content">
          <h3>No Research Network Found</h3>
          <p>Search for research topics to discover paper connections and build your knowledge graph.</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="graph-viewer"
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
      onMouseMove={(e) => {
        if (tooltip.show) {
          setTooltip(prev => ({
            ...prev,
            x: e.clientX,
            y: e.clientY
          }))
        }
      }}
    >
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={data}
        
        nodeLabel={node => `${node.title || 'Untitled'}\n${node.research_domain || 'Unknown'}\nQuality: ${((node.quality_score || 0) * 100).toFixed(0)}%`}
        nodeColor={node => {
          if (node.id === selectedNode) return '#FBBF24'
          if (node.id === hoveredNode) return '#60A5FA'
          return node.color || '#64748B'
        }}
        
        // 🔧 FIXED: Use number instead of function
        nodeRelSize={defaultNodeSize}
        
        nodePointerAreaPaint={nodePointerAreaPaint}
        nodeCanvasObject={nodeCanvasObject}
        linkCanvasObject={linkCanvasObject}
        
        linkWidth={link => Math.max(1, defaultConfig.linkWidth * Math.max(link.strength || 0.5, 0.3))}
        linkOpacity={defaultConfig.linkOpacity}
        linkColor={link => {
          const strength = link.strength || 0.5
          if (strength > 0.7) return 'rgba(59, 130, 246, 0.8)'
          if (strength > 0.5) return 'rgba(16, 185, 129, 0.7)'
          if (strength > 0.3) return 'rgba(245, 158, 11, 0.6)'
          return 'rgba(100, 116, 139, 0.5)'
        }}
        
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={(event) => handleNodeClick(null, event)}
        
        cooldownTicks={200}
        d3AlphaDecay={0.01}
        d3VelocityDecay={0.2}
        
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        enablePointerInteraction={true}
        
        backgroundColor="rgba(0,0,0,0)"
      />

      {/* Tooltip */}
      <AnimatePresence>
        {tooltip.show && tooltip.node && (
          <motion.div
            className="node-tooltip"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'fixed',
              left: Math.min(tooltip.x + 20, window.innerWidth - 320),
              top: Math.max(tooltip.y - 10, 10),
              zIndex: 1100,
              pointerEvents: 'none'
            }}
          >
            <div className="tooltip-content">
              <h4>{tooltip.node.title || 'Untitled Paper'}</h4>
              <div className="tooltip-meta">
                <span className="domain-tag">{tooltip.node.research_domain || 'Unknown'}</span>
                {tooltip.node.citation_count !== undefined && (
                  <span className="citation-count">
                    {tooltip.node.citation_count} citations
                  </span>
                )}
                <span className="quality-badge">
                  Quality: {((tooltip.node.quality_score || 0) * 100).toFixed(0)}%
                </span>
              </div>
              {tooltip.node.authors && (
                <p className="authors">
                  {tooltip.node.authors.split(';')[0]}{tooltip.node.authors.includes(';') ? ' et al.' : ''}
                </p>
              )}
              <div className="tooltip-hint">
                Click to view details →
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Overlay */}
      <div className="graph-info-overlay">
        <div className="info-item">
          <span className="info-label">Papers:</span>
          <span className="info-value">{data.nodes.length}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Connections:</span>
          <span className="info-value">{data.links.length}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Mode:</span>
          <span className="info-value">
            {(data._stableVersion || data._originalTimestamp) ? 'STABLE 🎯' : 'LOADING...'}
          </span>
        </div>
      </div>
    </motion.div>
  )
}))

GraphViewer.displayName = 'GraphViewer'

export default GraphViewer
