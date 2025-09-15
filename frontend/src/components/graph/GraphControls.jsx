import React, { useState } from 'react'
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
  Move3D,
  Square,
  Layers,
  AlertTriangle
} from 'lucide-react'

const GraphControls = ({
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleFullscreen,
  onScreenshot,
  onToggleLabels,
  onTogglePhysics,
  isFullscreen = false,
  showLabels = true,
  physicsEnabled = true,
  graphMode = '2d', // 🔧 DEFAULT TO 2D
  onModeChange,
  className = ''
}) => {
  const [expanded, setExpanded] = useState(false)
  const [activeTooltip, setActiveTooltip] = useState(null)

  const controlGroups = [
    {
      id: 'view',
      label: 'View Controls',
      controls: [
        {
          id: 'zoomIn',
          icon: ZoomIn,
          label: 'Zoom In',
          shortcut: '+',
          onClick: onZoomIn
        },
        {
          id: 'zoomOut',
          icon: ZoomOut,
          label: 'Zoom Out',
          shortcut: '-',
          onClick: onZoomOut
        },
        {
          id: 'reset',
          icon: RotateCcw,
          label: 'Reset View',
          shortcut: 'R',
          onClick: onResetView
        },
        {
          id: 'fullscreen',
          icon: isFullscreen ? Minimize2 : Maximize2,
          label: isFullscreen ? 'Exit Fullscreen' : 'Fullscreen',
          shortcut: 'F',
          onClick: onToggleFullscreen
        }
      ]
    },
    {
      id: 'display',
      label: 'Display Options',
      controls: [
        {
          id: 'labels',
          icon: showLabels ? Eye : EyeOff,
          label: showLabels ? 'Hide Labels' : 'Show Labels',
          shortcut: 'L',
          onClick: onToggleLabels,
          active: showLabels
        },
        {
          id: 'physics',
          icon: physicsEnabled ? Pause : Play,
          label: physicsEnabled ? 'Pause Physics' : 'Resume Physics',
          shortcut: 'P',
          onClick: onTogglePhysics,
          active: physicsEnabled
        }
      ]
    },
    {
      id: 'mode',
      label: 'View Mode',
      controls: [
        {
          id: '2d',
          icon: Square,
          label: '2D View',
          shortcut: '2',
          onClick: () => onModeChange('2d'),
          active: graphMode === '2d'
        },
        {
          id: '3d',
          icon: Move3D,
          label: '3D View (Temporarily Disabled)', // 🔧 UPDATED LABEL
          shortcut: '3',
          onClick: () => {
            // 🔧 BLOCK 3D MODE WITH WARNING
            console.warn('⚠️ 3D mode temporarily disabled due to link mutation issue')
            // Don't call onModeChange('3d')
          },
          active: graphMode === '3d',
          disabled: true // 🔧 DISABLE 3D BUTTON
        }
      ]
    },
    {
      id: 'export',
      label: 'Export Options',
      controls: [
        {
          id: 'screenshot',
          icon: Camera,
          label: 'Screenshot',
          shortcut: 'S',
          onClick: onScreenshot
        },
        {
          id: 'download',
          icon: Download,
          label: 'Export Data',
          shortcut: 'E',
          onClick: () => console.log('Export data')
        }
      ]
    }
  ]

  const handleControlClick = (control) => {
    if (control.disabled) return // 🔧 PREVENT DISABLED CLICKS
    control.onClick?.()
    setActiveTooltip(null)
  }

  return (
    <div className={`graph-controls ${className}`}>
      {/* Main Controls - Always Visible */}
      <div className="controls-main">
        {controlGroups[0].controls.map((control) => (
          <motion.button
            key={control.id}
            className={`control-btn ${control.active ? 'active' : ''} ${control.disabled ? 'disabled' : ''}`}
            onClick={() => handleControlClick(control)}
            onMouseEnter={() => setActiveTooltip(control.id)}
            onMouseLeave={() => setActiveTooltip(null)}
            whileHover={{ scale: control.disabled ? 1 : 1.05 }}
            whileTap={{ scale: control.disabled ? 1 : 0.95 }}
            title={`${control.label} (${control.shortcut})`}
            disabled={control.disabled}
          >
            <control.icon size={18} />
          </motion.button>
        ))}

        {/* Expand/Collapse Toggle */}
        <motion.button
          className={`control-btn expand-toggle ${expanded ? 'active' : ''}`}
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
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            {/* 🔧 WARNING NOTICE FOR 3D MODE */}
            <div className="control-group warning-group">
              <div className="group-label warning">
                <AlertTriangle size={14} />
                Technical Notice
              </div>
              <div className="warning-text">
                3D mode temporarily disabled while fixing connection display issues
              </div>
            </div>

            {controlGroups.slice(1).map((group) => (
              <div key={group.id} className="control-group">
                <div className="group-label">{group.label}</div>
                <div className="group-controls">
                  {group.controls.map((control) => (
                    <motion.button
                      key={control.id}
                      className={`control-btn ${control.active ? 'active' : ''} ${control.disabled ? 'disabled' : ''}`}
                      onClick={() => handleControlClick(control)}
                      onMouseEnter={() => setActiveTooltip(control.id)}
                      onMouseLeave={() => setActiveTooltip(null)}
                      whileHover={{ scale: control.disabled ? 1 : 1.05 }}
                      whileTap={{ scale: control.disabled ? 1 : 0.95 }}
                      title={`${control.label} (${control.shortcut})`}
                      disabled={control.disabled}
                    >
                      <control.icon size={16} />
                    </motion.button>
                  ))}
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tooltip */}
      <AnimatePresence>
        {activeTooltip && (
          <motion.div
            className="control-tooltip"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
          >
            {controlGroups
              .flatMap(g => g.controls)
              .find(c => c.id === activeTooltip)?.label}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Keyboard Shortcuts Hint */}
      {expanded && (
        <motion.div
          className="shortcuts-hint"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Layers size={12} />
          <span>Use keyboard shortcuts for quick access</span>
        </motion.div>
      )}
    </div>
  )
}

export default GraphControls
