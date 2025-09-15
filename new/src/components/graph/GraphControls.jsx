import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
  Camera,
  Play,
  Pause,
  Eye,
  EyeOff,
  Settings,
  Download,
  Square,
  AlertTriangle,
  Activity,
  Zap
} from 'lucide-react'

const GraphControls = ({
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleFullscreen,
  onScreenshot,
  onToggleLabels,
  onTogglePhysics,
  onStabilizeGraph,
  isFullscreen = false,
  showLabels = true,
  physicsEnabled = true,
  graphMode = '2d',
  onModeChange,
  className = ''
}) => {
  const [expanded, setExpanded] = useState(false)

  const handleControlClick = useCallback((action) => {
    try {
      action?.()
    } catch (error) {
      console.warn('Control action error:', error.message)
    }
  }, [])

  const primaryControls = [
    {
      id: 'zoomIn',
      icon: ZoomIn,
      label: 'Zoom In',
      action: onZoomIn
    },
    {
      id: 'zoomOut',
      icon: ZoomOut,
      label: 'Zoom Out',
      action: onZoomOut
    },
    {
      id: 'reset',
      icon: RotateCcw,
      label: 'Reset View',
      action: onResetView
    },
    {
      id: 'physics',
      icon: physicsEnabled ? Pause : Play,
      label: physicsEnabled ? 'Pause Physics' : 'Resume Physics',
      action: onTogglePhysics,
      active: physicsEnabled
    }
  ]

  const secondaryControls = [
    {
      id: 'labels',
      icon: showLabels ? Eye : EyeOff,
      label: showLabels ? 'Hide Labels' : 'Show Labels',
      action: onToggleLabels,
      active: showLabels
    },
    {
      id: 'stabilize',
      icon: Activity,
      label: 'Stabilize Graph',
      action: onStabilizeGraph
    },
    {
      id: 'fullscreen',
      icon: isFullscreen ? Minimize2 : Maximize2,
      label: isFullscreen ? 'Exit Fullscreen' : 'Fullscreen',
      action: onToggleFullscreen
    },
    {
      id: 'screenshot',
      icon: Camera,
      label: 'Screenshot',
      action: onScreenshot
    }
  ]

  return (
    <div className={`graph-controls ${className}`}>
      {/* Primary Controls - Always Visible */}
      <div className="controls-main">
        {primaryControls.map((control) => (
          <motion.button
            key={control.id}
            className={`control-btn ${control.active ? 'active' : ''}`}
            onClick={() => handleControlClick(control.action)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title={control.label}
          >
            <control.icon size={18} />
          </motion.button>
        ))}

        {/* Expand Toggle */}
        <motion.button
          className={`control-btn ${expanded ? 'active' : ''}`}
          onClick={() => setExpanded(!expanded)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title="More Controls"
        >
          <Settings size={18} />
        </motion.button>
      </div>

      {/* Extended Controls */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            className="controls-extended"
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ duration: 0.3 }}
          >
            {/* Physics Status */}
            <div className="control-group">
              <div className="group-label">
                <Zap size={14} />
                Physics Status
              </div>
              <div className={`physics-indicator ${physicsEnabled ? 'active' : 'paused'}`}>
                {physicsEnabled 
                  ? '🌊 Physics Active - Nodes moving naturally' 
                  : '⏸️ Physics Paused - Stable positions'
                }
              </div>
            </div>

            {/* Display Controls */}
            <div className="control-group">
              <div className="group-label">Display Options</div>
              <div className="group-controls">
                {secondaryControls.map((control) => (
                  <motion.button
                    key={control.id}
                    className={`control-btn ${control.active ? 'active' : ''}`}
                    onClick={() => handleControlClick(control.action)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    title={control.label}
                  >
                    <control.icon size={16} />
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Mode Selection */}
            <div className="control-group">
              <div className="group-label">View Mode</div>
              <div className="group-controls">
                <motion.button
                  className={`control-btn ${graphMode === '2d' ? 'active' : ''}`}
                  onClick={() => handleControlClick(() => onModeChange?.('2d'))}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  title="2D View"
                >
                  <Square size={16} />
                </motion.button>
                
                <motion.button
                  className="control-btn disabled"
                  title="3D View (Temporarily Disabled)"
                  disabled
                >
                  <AlertTriangle size={16} />
                </motion.button>
              </div>
            </div>

            {/* Usage Tips */}
            <div className="control-group warning-group">
              <div className="group-label warning">
                <AlertTriangle size={14} />
                Tips
              </div>
              <div className="warning-text">
                • Pause physics for easier node selection<br/>
                • Use stabilize to organize layout<br/>
                • Zoom and pan to explore the graph
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default GraphControls
