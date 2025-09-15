import React from 'react'
import { motion } from 'framer-motion'
import { Activity, Database, Search, Zap, Network } from 'lucide-react'

const StatusBar = ({ 
  nodeCount, 
  linkCount, 
  selectedNode, 
  searchQuery, 
  statistics 
}) => {
  return (
    <motion.div 
      className="status-bar"
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <div className="status-content">
        <div className="status-section">
          <div className="status-item">
            <Database size={16} />
            <span className="status-label">Network:</span>
            <span className="status-value">
              {nodeCount} papers, {linkCount} connections
            </span>
          </div>
          
          {searchQuery && (
            <div className="status-item">
              <Search size={16} />
              <span className="status-label">Query:</span>
              <span className="status-value">"{searchQuery}"</span>
            </div>
          )}
          
          {selectedNode && (
            <div className="status-item">
              <Activity size={16} />
              <span className="status-label">Selected:</span>
              <span className="status-value">Paper #{selectedNode}</span>
            </div>
          )}
        </div>
        
        <div className="status-indicators">
          <div className="indicator active">
            <Network size={14} />
            <span>Live Graph</span>
          </div>
          
          {statistics && statistics.totalConnections > 0 && (
            <div className="indicator">
              <span className="metric">
                Avg Quality: {Math.round((statistics.avgQuality || 0) * 100)}%
              </span>
            </div>
          )}
          
          {statistics && statistics.topDomain && (
            <div className="indicator">
              <span className="metric">
                Top: {statistics.topDomain}
              </span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default StatusBar
