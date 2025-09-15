import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Settings,
  Sliders,
  Eye,
  Palette,
  Download,
  RotateCcw,
  Play,
  Pause,
  Filter,
  BarChart3
} from 'lucide-react'

const ControlPanel = ({
  config,
  onConfigChange,
  onResetView,
  onExport,
  statistics,
  graphMode,
  onGraphModeChange
}) => {
  const [activeSection, setActiveSection] = useState('display')

  const sections = [
    { id: 'display', label: 'Display', icon: Eye },
    { id: 'physics', label: 'Physics', icon: Settings },
    { id: 'filters', label: 'Filters', icon: Filter },
    { id: 'stats', label: 'Stats', icon: BarChart3 }
  ]

  const handleSliderChange = (key, value) => {
    onConfigChange({
      ...config,
      [key]: value
    })
  }

  return (
    <motion.div
      className="control-panel"
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
    >
      {/* Header */}
      <div className="panel-header">
        <div className="header-title">
          <Settings size={20} />
          <h2>Graph Controls</h2>
        </div>
      </div>

      {/* Section Tabs */}
      <div className="panel-tabs">
        {sections.map((section) => (
          <motion.button
            key={section.id}
            className={`tab-btn ${activeSection === section.id ? 'active' : ''}`}
            onClick={() => setActiveSection(section.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <section.icon size={16} />
            {section.label}
          </motion.button>
        ))}
      </div>

      {/* Content */}
      <div className="panel-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeSection === 'display' && (
              <div className="control-section">
                <h3>Display Settings</h3>
                
                <div className="control-group">
                  <label>Node Size</label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={config.nodeSize || 5}
                    onChange={(e) => handleSliderChange('nodeSize', parseInt(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.nodeSize || 5}</span>
                </div>

                <div className="control-group">
                  <label>Link Width</label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.1"
                    value={config.linkWidth || 1}
                    onChange={(e) => handleSliderChange('linkWidth', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.linkWidth || 1}</span>
                </div>

                <div className="control-group">
                  <label>Link Opacity</label>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.1"
                    value={config.linkOpacity || 0.6}
                    onChange={(e) => handleSliderChange('linkOpacity', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{Math.round((config.linkOpacity || 0.6) * 100)}%</span>
                </div>

                <div className="control-group checkbox">
                  <input
                    type="checkbox"
                    id="showLabels"
                    checked={config.showLabels !== false}
                    onChange={(e) => handleSliderChange('showLabels', e.target.checked)}
                  />
                  <label htmlFor="showLabels">Show Node Labels</label>
                </div>
              </div>
            )}

            {activeSection === 'physics' && (
              <div className="control-section">
                <h3>Physics Engine</h3>
                
                <div className="control-group">
                  <label>Simulation Speed</label>
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={config.simulationSpeed || 1}
                    onChange={(e) => handleSliderChange('simulationSpeed', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.simulationSpeed || 1}x</span>
                </div>

                <div className="control-group">
                  <label>Link Distance</label>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    value={config.linkDistance || 200}
                    onChange={(e) => handleSliderChange('linkDistance', parseInt(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.linkDistance || 200}</span>
                </div>

                <div className="control-group">
                  <label>Node Repulsion</label>
                  <input
                    type="range"
                    min="100"
                    max="2000"
                    value={config.nodeRepulsion || 800}
                    onChange={(e) => handleSliderChange('nodeRepulsion', parseInt(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.nodeRepulsion || 800}</span>
                </div>

                <div className="control-actions">
                  <motion.button
                    className="control-btn"
                    onClick={() => handleSliderChange('paused', !config.paused)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {config.paused ? <Play size={16} /> : <Pause size={16} />}
                    {config.paused ? 'Resume' : 'Pause'} Physics
                  </motion.button>
                </div>
              </div>
            )}

            {activeSection === 'filters' && (
              <div className="control-section">
                <h3>Graph Filters</h3>
                
                <div className="filter-group">
                  <label>Citation Threshold</label>
                  <input
                    type="range"
                    min="0"
                    max="1000"
                    value={config.minCitations || 0}
                    onChange={(e) => handleSliderChange('minCitations', parseInt(e.target.value))}
                    className="slider"
                  />
                  <span className="value">{config.minCitations || 0}+ citations</span>
                </div>

                <div className="filter-group">
                  <label>Quality Score</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={(config.minQuality || 0) * 100}
                    onChange={(e) => handleSliderChange('minQuality', parseInt(e.target.value) / 100)}
                    className="slider"
                  />
                  <span className="value">{Math.round((config.minQuality || 0) * 100)}%+</span>
                </div>

                <div className="filter-domains">
                  <label>Research Domains</label>
                  {['Machine Learning', 'Healthcare', 'Computer Vision', 'Physics', 'Chemistry'].map(domain => (
                    <div key={domain} className="domain-filter">
                      <input
                        type="checkbox"
                        id={`domain-${domain}`}
                        checked={!config.hiddenDomains?.includes(domain)}
                        onChange={(e) => {
                          const hidden = config.hiddenDomains || []
                          const newHidden = e.target.checked 
                            ? hidden.filter(d => d !== domain)
                            : [...hidden, domain]
                          handleSliderChange('hiddenDomains', newHidden)
                        }}
                      />
                      <label htmlFor={`domain-${domain}`}>{domain}</label>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeSection === 'stats' && (
              <div className="control-section">
                <h3>Network Statistics</h3>
                
                {statistics && (
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-label">Total Papers</div>
                      <div className="stat-value">{statistics.totalPapers || 0}</div>
                    </div>
                    
                    <div className="stat-card">
                      <div className="stat-label">Connections</div>
                      <div className="stat-value">{statistics.totalConnections || 0}</div>
                    </div>
                    
                    <div className="stat-card">
                      <div className="stat-label">Avg Quality</div>
                      <div className="stat-value">{Math.round((statistics.avgQuality || 0) * 100)}%</div>
                    </div>
                    
                    <div className="stat-card">
                      <div className="stat-label">Domains</div>
                      <div className="stat-value">{statistics.uniqueDomains || 0}</div>
                    </div>
                  </div>
                )}

                <div className="export-section">
                  <h4>Export Options</h4>
                  <div className="export-buttons">
                    <motion.button
                      className="export-btn"
                      onClick={() => onExport('png')}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Download size={16} />
                      PNG Image
                    </motion.button>
                    
                    <motion.button
                      className="export-btn"
                      onClick={() => onExport('json')}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Download size={16} />
                      JSON Data
                    </motion.button>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer Actions */}
      <div className="panel-footer">
        <motion.button
          className="footer-btn secondary"
          onClick={onResetView}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw size={16} />
          Reset View
        </motion.button>
      </div>
    </motion.div>
  )
}

export default ControlPanel
