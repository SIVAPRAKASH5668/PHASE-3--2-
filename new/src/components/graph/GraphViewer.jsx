import React, { forwardRef, useEffect, useRef, useState, useCallback, memo, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const StaticGraphViewer = memo(forwardRef(({
  data,
  mode = '2d',
  config = {},
  selectedNode,
  hoveredNode,
  onNodeInteraction,
  onConfigChange
}, ref) => {
  const canvasRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, node: null })
  const [zoomLevel, setZoomLevel] = useState(1)
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  // **STATIC LAYOUT CONFIGURATION**
  const defaultConfig = useMemo(() => ({
    nodeSize: 20,
    linkWidth: 2,
    linkOpacity: 0.7,
    showLabels: true,
    nodeSpacing: 150, // Minimum distance between nodes
    circleRadius: 200, // Base radius for circular layout
    gridSpacing: 180, // Grid spacing for grid layout
    layoutType: 'force_directed', // 'circular', 'grid', 'force_directed'
    ...config
  }), [config])

  // **STATIC FORCE-DIRECTED LAYOUT CALCULATION**
  const calculateStaticLayout = useCallback((nodes, links) => {
    if (!nodes?.length) return []

    const nodeCount = nodes.length
    const iterations = 300 // Static simulation iterations
    const repulsionStrength = 5000
    const attractionStrength = 0.01
    const damping = 0.9
    const minDistance = defaultConfig.nodeSpacing

    // Initialize positions
    let positions = nodes.map((node, index) => {
      // Start with circular distribution
      const angle = (index / nodeCount) * 2 * Math.PI
      const radius = Math.max(defaultConfig.circleRadius, nodeCount * 15)
      
      return {
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        vx: 0,
        vy: 0
      }
    })

    // Create adjacency map for connected nodes
    const adjacency = new Map()
    links?.forEach(link => {
      const sourceId = String(link.source)
      const targetId = String(link.target)
      
      if (!adjacency.has(sourceId)) adjacency.set(sourceId, new Set())
      if (!adjacency.has(targetId)) adjacency.set(targetId, new Set())
      
      adjacency.get(sourceId).add(targetId)
      adjacency.get(targetId).add(sourceId)
    })

    // Static force simulation
    for (let iter = 0; iter < iterations; iter++) {
      const alpha = Math.max(0.01, 1 - (iter / iterations)) // Cooling

      positions.forEach((nodeA, i) => {
        let fx = 0, fy = 0

        // Repulsion forces (all nodes repel each other)
        positions.forEach((nodeB, j) => {
          if (i === j) return

          const dx = nodeA.x - nodeB.x
          const dy = nodeA.y - nodeB.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance > 0) {
            const repulsion = repulsionStrength / (distance * distance) * alpha
            fx += (dx / distance) * repulsion
            fy += (dy / distance) * repulsion
          }
        })

        // Attraction forces (connected nodes attract)
        const connectedNodes = adjacency.get(String(nodeA.id)) || new Set()
        connectedNodes.forEach(connectedId => {
          const connectedNode = positions.find(n => String(n.id) === connectedId)
          if (!connectedNode) return

          const dx = connectedNode.x - nodeA.x
          const dy = connectedNode.y - nodeA.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance > 0) {
            const attraction = attractionStrength * distance * alpha
            fx += (dx / distance) * attraction
            fy += (dy / distance) * attraction
          }
        })

        // Apply forces with damping
        nodeA.vx = (nodeA.vx + fx) * damping
        nodeA.vy = (nodeA.vy + fy) * damping
        
        // Update position
        nodeA.x += nodeA.vx
        nodeA.y += nodeA.vy
      })

      // Collision detection and resolution
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          const nodeA = positions[i]
          const nodeB = positions[j]
          
          const dx = nodeA.x - nodeB.x
          const dy = nodeA.y - nodeB.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance < minDistance && distance > 0) {
            const overlap = minDistance - distance
            const moveDistance = overlap / 2
            const moveX = (dx / distance) * moveDistance
            const moveY = (dy / distance) * moveDistance
            
            nodeA.x += moveX
            nodeA.y += moveY
            nodeB.x -= moveX
            nodeB.y -= moveY
          }
        }
      }
    }

    console.log('✅ Static layout calculated for', nodeCount, 'nodes with spacing:', minDistance)
    return positions
  }, [defaultConfig.nodeSpacing, defaultConfig.circleRadius])

  // **PROCESS GRAPH DATA**
  const processedGraphData = useMemo(() => {
    if (!data?.nodes?.length) return null

    const nodes = data.nodes.map(node => ({
      id: String(node.id),
      title: node.title || 'Untitled',
      label: node.label || node.title || 'Untitled',
      size: Math.max(15, Number(node.size) || defaultConfig.nodeSize),
      color: node.color || '#64748B',
      quality_score: Number(node.quality_score) || 0.5,
      research_domain: node.research_domain || 'Unknown',
      authors: node.authors || 'Unknown Authors',
      citation_count: Number(node.citation_count) || 0,
      ...node
    }))

    const links = (data.links || []).map(link => ({
      source: String(link.source),
      target: String(link.target),
      strength: Number(link.strength) || 0.5,
      type: link.type || 'related',
      ...link
    }))

    // Calculate static positions
    const positionedNodes = calculateStaticLayout(nodes, links)

    return {
      nodes: positionedNodes,
      links: links
    }
  }, [data, defaultConfig.nodeSize, calculateStaticLayout])

  // **CANVAS DRAWING**
  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !processedGraphData) return

    const ctx = canvas.getContext('2d')
    const { width, height } = dimensions
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height)
    
    // Apply zoom and pan transformations
    ctx.save()
    ctx.translate(width / 2 + panOffset.x, height / 2 + panOffset.y)
    ctx.scale(zoomLevel, zoomLevel)

    // Draw links first (behind nodes)
    processedGraphData.links.forEach(link => {
      const sourceNode = processedGraphData.nodes.find(n => n.id === link.source)
      const targetNode = processedGraphData.nodes.find(n => n.id === link.target)
      
      if (!sourceNode || !targetNode) return

      const strength = link.strength || 0.5
      const linkWidth = Math.max(1, defaultConfig.linkWidth * strength)
      
      let strokeColor = `rgba(100, 116, 139, ${defaultConfig.linkOpacity})`
      if (link.type === 'complements') {
        strokeColor = `rgba(16, 185, 129, ${defaultConfig.linkOpacity})`
      } else if (link.type === 'domain_overlap') {
        strokeColor = `rgba(168, 85, 247, ${defaultConfig.linkOpacity})`
      } else if (strength > 0.7) {
        strokeColor = `rgba(59, 130, 246, ${defaultConfig.linkOpacity})`
      }

      ctx.beginPath()
      ctx.moveTo(sourceNode.x, sourceNode.y)
      ctx.lineTo(targetNode.x, targetNode.y)
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = linkWidth
      ctx.lineCap = 'round'
      ctx.stroke()
    })

    // Draw nodes
    processedGraphData.nodes.forEach(node => {
      const nodeRadius = node.size || defaultConfig.nodeSize
      let nodeColor = node.color || '#64748B'
      let borderColor = '#FFFFFF'
      let borderWidth = 2

      // Highlight selected/hovered nodes
      if (String(node.id) === String(selectedNode)) {
        nodeColor = '#FBBF24'
        borderColor = '#F59E0B'
        borderWidth = 3
      } else if (String(node.id) === String(hoveredNode)) {
        nodeColor = '#60A5FA'
        borderColor = '#3B82F6'
        borderWidth = 3
      }

      // Draw node shadow/glow
      ctx.shadowColor = nodeColor
      ctx.shadowBlur = 6
      ctx.beginPath()
      ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI)
      ctx.fillStyle = nodeColor
      ctx.fill()

      // Draw node border
      ctx.shadowBlur = 0
      ctx.strokeStyle = borderColor
      ctx.lineWidth = borderWidth
      ctx.stroke()

      // Draw labels
      if (defaultConfig.showLabels && zoomLevel > 0.3) {
        const label = node.label || node.title
        const fontSize = Math.max(10, 14 / zoomLevel)
        
        ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`
        ctx.fillStyle = '#FFFFFF'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)'
        ctx.shadowBlur = 2
        
        const maxLength = Math.floor(15 / Math.max(0.3, zoomLevel))
        const displayLabel = label.length > maxLength ? label.substring(0, maxLength) + '...' : label
        const labelY = node.y + nodeRadius + fontSize + 5
        
        ctx.fillText(displayLabel, node.x, labelY)
        ctx.shadowBlur = 0
      }

      // Quality indicator
      if (zoomLevel > 0.5 && node.quality_score) {
        const qualityRadius = nodeRadius * 0.2
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
    })

    ctx.restore()
  }, [processedGraphData, dimensions, zoomLevel, panOffset, selectedNode, hoveredNode, defaultConfig])

  // **MOUSE INTERACTION**
  const getNodeAtPosition = useCallback((clientX, clientY) => {
    if (!processedGraphData || !canvasRef.current) return null

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    
    // Convert screen coordinates to canvas coordinates
    const canvasX = clientX - rect.left
    const canvasY = clientY - rect.top
    
    // Convert to graph coordinates
    const graphX = (canvasX - dimensions.width / 2 - panOffset.x) / zoomLevel
    const graphY = (canvasY - dimensions.height / 2 - panOffset.y) / zoomLevel

    // Find node under cursor
    return processedGraphData.nodes.find(node => {
      const dx = graphX - node.x
      const dy = graphY - node.y
      const distance = Math.sqrt(dx * dx + dy * dy)
      return distance <= (node.size || defaultConfig.nodeSize) + 10 // Click tolerance
    })
  }, [processedGraphData, dimensions, zoomLevel, panOffset, defaultConfig.nodeSize])

  const handleMouseDown = useCallback((e) => {
    const node = getNodeAtPosition(e.clientX, e.clientY)
    
    if (node) {
      console.log('✅ Node clicked:', node.id, '-', node.title)
      onNodeInteraction?.(String(node.id), 'click')
    } else {
      // Start panning
      setIsDragging(true)
      setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y })
      onNodeInteraction?.(null, 'click')
    }
  }, [getNodeAtPosition, onNodeInteraction, panOffset])

  const handleMouseMove = useCallback((e) => {
    if (isDragging) {
      // Handle panning
      setPanOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    } else {
      // Handle hover
      const node = getNodeAtPosition(e.clientX, e.clientY)
      
      if (node) {
        setTooltip({
          show: true,
          x: e.clientX,
          y: e.clientY,
          node: node
        })
        onNodeInteraction?.(String(node.id), 'hover')
      } else {
        setTooltip({ show: false, x: 0, y: 0, node: null })
        onNodeInteraction?.(null, 'unhover')
      }
    }
  }, [isDragging, dragStart, getNodeAtPosition, onNodeInteraction])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleWheel = useCallback((e) => {
    // Don't prevent default - will be handled by useEffect with proper passive: false
    const delta = e.deltaY * -0.001
    const newZoom = Math.min(Math.max(0.1, zoomLevel + delta), 3)
    setZoomLevel(newZoom)
  }, [zoomLevel])

  // **ZOOM CONTROLS**
  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 1.2, 3))
  }, [])

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.1))
  }, [])

  const resetView = useCallback(() => {
    setZoomLevel(1)
    setPanOffset({ x: 0, y: 0 })
  }, [])

  // **EFFECTS**
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setDimensions({ width: rect.width, height: rect.height })
        
        if (canvasRef.current) {
          canvasRef.current.width = rect.width
          canvasRef.current.height = rect.height
        }
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Handle wheel events with proper passive: false
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleWheelEvent = (e) => {
      e.preventDefault()
      const delta = e.deltaY * -0.001
      const newZoom = Math.min(Math.max(0.1, zoomLevel + delta), 3)
      setZoomLevel(newZoom)
    }

    // Add event listener with passive: false to allow preventDefault
    canvas.addEventListener('wheel', handleWheelEvent, { passive: false })
    
    return () => {
      canvas.removeEventListener('wheel', handleWheelEvent)
    }
  }, [zoomLevel])

  useEffect(() => {
    drawGraph()
  }, [drawGraph])

  // Expose methods to parent
  useEffect(() => {
    if (ref) {
      ref.current = {
        zoomIn,
        zoomOut,
        resetView,
        zoomToFit: resetView
      }
    }
  }, [ref, zoomIn, zoomOut, resetView])

  if (!processedGraphData || !processedGraphData.nodes.length) {
    return (
      <motion.div
        className="graph-empty"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'transparent',
          borderRadius: '12px'
        }}
      >
        <div style={{ textAlign: 'center', color: '#64748B' }}>
          <h3>No Research Network Found</h3>
          <p>Provide graph data to visualize paper connections.</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="static-graph-viewer"
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '12px',
        background: 'transparent',
        cursor: isDragging ? 'grabbing' : 'grab'
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          width: '100%',
          height: '100%',
          display: 'block'
        }}
      />

      {/* Enhanced Tooltip */}
      <AnimatePresence>
        {tooltip.show && tooltip.node && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'fixed',
              left: Math.min(tooltip.x + 15, window.innerWidth - 320),
              top: Math.max(tooltip.y - 10, 10),
              zIndex: 1000,
              background: 'rgba(0, 0, 0, 0.9)',
              color: 'white',
              padding: '12px 16px',
              borderRadius: '8px',
              fontSize: '14px',
              maxWidth: '300px',
              pointerEvents: 'none',
              backdropFilter: 'blur(4px)',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}
          >
            <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
              {tooltip.node.title || 'Untitled Paper'}
            </div>
            <div style={{ fontSize: '12px', opacity: 0.8, marginBottom: '8px' }}>
              {tooltip.node.research_domain || 'Unknown Domain'}
              {tooltip.node.citation_count !== undefined && (
                <span style={{ marginLeft: '8px' }}>
                  • {tooltip.node.citation_count} citations
                </span>
              )}
            </div>
            {tooltip.node.authors && (
              <div style={{ fontSize: '12px', opacity: 0.7 }}>
                {tooltip.node.authors.split(';')[0]}
                {tooltip.node.authors.includes(';') ? ' et al.' : ''}
              </div>
            )}
            <div style={{ fontSize: '11px', marginTop: '8px', opacity: 0.6 }}>
              Click to select • ID: {tooltip.node.id}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Graph Info */}
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '6px',
        fontSize: '14px',
        display: 'flex',
        gap: '16px'
      }}>
        <span>Papers: {processedGraphData.nodes.length}</span>
        <span>Connections: {processedGraphData.links.length}</span>
        <span style={{ color: '#10B981' }}>✅ Static Layout</span>
      </div>

      {/* Zoom Controls */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        right: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        <button
          onClick={zoomIn}
          style={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            border: 'none',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            fontSize: '18px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          +
        </button>
        <button
          onClick={zoomOut}
          style={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            border: 'none',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            fontSize: '18px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          -
        </button>
        <button
          onClick={resetView}
          title="Reset View"
          style={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            border: 'none',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            fontSize: '16px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          ⌂
        </button>
      </div>

      {/* Zoom Level Indicator */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        left: 16,
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px'
      }}>
        Zoom: {Math.round(zoomLevel * 100)}%
      </div>
    </motion.div>
  )
}))

StaticGraphViewer.displayName = 'StaticGraphViewer'

export default StaticGraphViewer