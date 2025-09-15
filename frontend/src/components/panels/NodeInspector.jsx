import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  ExternalLink,
  Download,
  Share,
  BookOpen,
  Users,
  Calendar,
  Award,
  TrendingUp,
  Lightbulb,
  AlertCircle,
  HelpCircle,
  ArrowUpRight,
  Copy,
  Check
} from 'lucide-react'

const NodeInspector = ({
  node,
  nodeDetails,
  loading,
  onClose,
  onRelatedNodeClick
}) => {
  const [activeTab, setActiveTab] = useState('overview')
  const [copied, setCopied] = useState(false)

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BookOpen },
    { id: 'details', label: 'Details', icon: TrendingUp },
    { id: 'connections', label: 'Connections', icon: Users }
  ]

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(nodeDetails?.doi || nodeDetails?.url || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: nodeDetails?.title || 'Research Paper',
          text: nodeDetails?.context_summary || '',
          url: nodeDetails?.doi || nodeDetails?.url || window.location.href
        })
      } catch (err) {
        console.error('Share failed:', err)
      }
    } else {
      handleCopyLink()
    }
  }

  return (
    <motion.div
      className="node-inspector"
      initial={{ x: -400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: -400, opacity: 0 }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
    >
      {/* Header */}
      <div className="inspector-header">
        <div className="header-title">
          <BookOpen size={20} />
          <h2>Paper Inspector</h2>
        </div>
        <div className="header-actions">
          <motion.button
            className="action-btn"
            onClick={handleShare}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title="Share"
          >
            <Share size={18} />
          </motion.button>
          <motion.button
            className="action-btn"
            onClick={onClose}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <X size={18} />
          </motion.button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="inspector-tabs">
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <tab.icon size={16} />
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Content */}
      <div className="inspector-content">
        {loading ? (
          <div className="inspector-loading">
            <motion.div
              className="loading-spinner"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            <p>Loading paper details...</p>
          </div>
        ) : nodeDetails ? (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === 'overview' && (
                <div className="tab-content">
                  {/* Paper Header */}
                  <div className="paper-header">
                    <h1 className="paper-title">{nodeDetails.title}</h1>
                    
                    <div className="paper-meta">
                      {nodeDetails.authors && (
                        <div className="meta-item">
                          <Users size={16} />
                          <span>{nodeDetails.authors.split(';').slice(0, 3).join(', ')}{nodeDetails.authors.split(';').length > 3 ? ' et al.' : ''}</span>
                        </div>
                      )}
                      
                      {nodeDetails.published_date && (
                        <div className="meta-item">
                          <Calendar size={16} />
                          <span>{nodeDetails.published_date}</span>
                        </div>
                      )}
                      
                      {nodeDetails.citation_count && (
                        <div className="meta-item">
                          <Award size={16} />
                          <span>{nodeDetails.citation_count} citations</span>
                        </div>
                      )}
                    </div>

                    <div className="paper-badges">
                      <span className="domain-badge">{nodeDetails.research_domain}</span>
                      {nodeDetails.quality_score && (
                        <span className="quality-badge">
                          {Math.round(nodeDetails.quality_score * 100)}% Quality
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Abstract */}
                  {nodeDetails.abstract && (
                    <div className="content-section">
                      <h3>Abstract</h3>
                      <p className="abstract-text">{nodeDetails.abstract}</p>
                    </div>
                  )}

                  {/* Context Summary */}
                  {nodeDetails.context_summary && (
                    <div className="content-section">
                      <h3>AI Summary</h3>
                      <div className="ai-summary">
                        <p>{nodeDetails.context_summary}</p>
                      </div>
                    </div>
                  )}

                  {/* Action Links */}
                  <div className="action-links">
                    {nodeDetails.doi && (
                      <motion.a
                        href={nodeDetails.doi}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link primary"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <ExternalLink size={16} />
                        View Paper
                      </motion.a>
                    )}
                    
                    {nodeDetails.pdf_url && (
                      <motion.a
                        href={nodeDetails.pdf_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Download size={16} />
                        Download PDF
                      </motion.a>
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'details' && (
                <div className="tab-content">
                  {/* Key Findings */}
                  {nodeDetails.key_findings && nodeDetails.key_findings.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <Lightbulb size={18} />
                        Key Findings
                      </h3>
                      <div className="findings-list">
                        {nodeDetails.key_findings.map((finding, index) => (
                          <motion.div
                            key={index}
                            className="finding-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <TrendingUp size={16} className="finding-icon" />
                            <p>{finding}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Limitations */}
                  {nodeDetails.limitations && nodeDetails.limitations.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <AlertCircle size={18} />
                        Limitations
                      </h3>
                      <div className="limitations-list">
                        {nodeDetails.limitations.map((limitation, index) => (
                          <motion.div
                            key={index}
                            className="limitation-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <AlertCircle size={16} className="limitation-icon" />
                            <p>{limitation}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Future Directions */}
                  {nodeDetails.future_directions && nodeDetails.future_directions.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <ArrowUpRight size={18} />
                        Future Directions
                      </h3>
                      <div className="future-list">
                        {nodeDetails.future_directions.map((direction, index) => (
                          <motion.div
                            key={index}
                            className="future-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <ArrowUpRight size={16} className="future-icon" />
                            <p>{direction}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Quality Metrics */}
                  {nodeDetails.quality_metrics && (
                    <div className="content-section">
                      <h3>Quality Metrics</h3>
                      <div className="quality-grid">
                        {Object.entries(nodeDetails.quality_metrics).map(([key, value]) => (
                          <div key={key} className="quality-item">
                            <span className="metric-label">{key.replace('_', ' ')}</span>
                            <div className="metric-bar">
                              <motion.div
                                className="metric-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${value * 100}%` }}
                                transition={{ duration: 1, delay: 0.5 }}
                              />
                            </div>
                            <span className="metric-value">{Math.round(value * 100)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'connections' && (
                <div className="tab-content">
                  {/* Related Papers */}
                  {nodeDetails.related_papers && nodeDetails.related_papers.length > 0 && (
                    <div className="content-section">
                      <h3>Related Papers</h3>
                      <div className="related-papers">
                        {nodeDetails.related_papers.map((paper, index) => (
                          <motion.div
                            key={paper.id || index}
                            className="related-paper"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            onClick={() => onRelatedNodeClick(paper.id, 'click')}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <h4>{paper.title}</h4>
                            <p className="paper-meta">
                              {paper.authors && (
                                <span>{paper.authors.split(';')[0]} et al.</span>
                              )}
                              {paper.published_date && (
                                <span> • {paper.published_date}</span>
                              )}
                            </p>
                            <div className="connection-strength">
                              Connection: {Math.round((paper.similarity || 0.5) * 100)}%
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Co-authors Network */}
                  {nodeDetails.co_authors && nodeDetails.co_authors.length > 0 && (
                    <div className="content-section">
                      <h3>Co-authors</h3>
                      <div className="author-network">
                        {nodeDetails.co_authors.map((author, index) => (
                          <motion.div
                            key={index}
                            className="author-card"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.05 }}
                          >
                            <div className="author-avatar">
                              <Users size={20} />
                            </div>
                            <div className="author-info">
                              <h4>{author.name}</h4>
                              <p>{author.papers_count || 0} papers</p>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        ) : (
          <div className="inspector-error">
            <HelpCircle size={48} />
            <h3>No details available</h3>
            <p>Unable to load paper details at this time.</p>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default NodeInspector
