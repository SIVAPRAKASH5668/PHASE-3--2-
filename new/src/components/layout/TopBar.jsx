import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Settings,
  Info,
  Filter,
  Zap,
  RefreshCw,
  Plus,
  BarChart3,
  Calendar,
  Database,
  Globe,
  ChevronDown,
  Sparkles
} from 'lucide-react'

const TopBar = ({
  onSearch,
  searchQuery,
  onSearchQueryChange,
  searchInputRef,
  searchFilters,
  onFiltersChange,
  loading,
  hasData,
  onToggleControls,
  onToggleInspector,
  showControls,
  showInspector,
  onNewSearch,
  statistics
}) => {
  const [showFilters, setShowFilters] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const filtersRef = useRef(null)

  // Sample suggestions based on popular research topics
  const suggestions = [
    { text: 'Machine Learning in Healthcare', category: 'AI', icon: Sparkles },
    { text: 'Computer Vision Applications', category: 'AI', icon: Sparkles },
    { text: 'Natural Language Processing', category: 'NLP', icon: Sparkles },
    { text: 'Climate Change Research', category: 'Environment', icon: Globe },
    { text: 'Quantum Computing', category: 'Physics', icon: Zap },
    { text: 'Deep Learning Neural Networks', category: 'AI', icon: Sparkles },
    { text: 'Sustainable Energy Systems', category: 'Energy', icon: Globe },
    { text: 'Medical Imaging Analysis', category: 'Healthcare', icon: Sparkles }
  ]

  const filteredSuggestions = suggestions.filter(suggestion =>
    suggestion.text.toLowerCase().includes(searchQuery.toLowerCase())
  ).slice(0, 6)

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      onSearch(searchQuery.trim(), searchFilters)
      setShowSuggestions(false)
    }
  }

  const handleSearchChange = (value) => {
    onSearchQueryChange(value)
    setShowSuggestions(value.length > 1)
  }

  const handleSuggestionSelect = (suggestion) => {
    onSearchQueryChange(suggestion)
    onSearch(suggestion, searchFilters)
    setShowSuggestions(false)
  }

  const handleFilterChange = (key, value) => {
    const updatedFilters = { ...searchFilters, [key]: value }
    onFiltersChange(updatedFilters)
  }

  const toggleSource = (source) => {
    const currentSources = searchFilters.sources
    const updatedSources = currentSources.includes(source)
      ? currentSources.filter(s => s !== source)
      : [...currentSources, source]
    handleFilterChange('sources', updatedSources)
  }

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (filtersRef.current && !filtersRef.current.contains(event.target)) {
        setShowFilters(false)
      }
      if (searchInputRef.current && !searchInputRef.current.contains(event.target)) {
        setShowSuggestions(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <motion.header
      className="professional-top-bar"
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      {/* Main Header Container */}
      <div className="header-container">
        {/* Brand Section */}
        <motion.div
          className="brand-section"
          whileHover={{ scale: 1.02 }}
        >
          <div className="brand-icon">
            <motion.div
              className="neural-network-mini"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <div className="neural-dot" />
              <div className="neural-dot" />
              <div className="neural-dot" />
            </motion.div>
          </div>
          <div className="brand-text">
            <h1>Research<span className="brand-accent">Graph</span></h1>
            <span className="brand-subtitle">AI Discovery Platform</span>
          </div>
        </motion.div>

        {/* Search Section */}
        <div className="search-section" ref={searchInputRef}>
          <form onSubmit={handleSearchSubmit} className="search-form">
            <div className="search-input-container">
              <Search size={20} className="search-icon" />
              <input
                ref={searchInputRef}
                type="text"
                placeholder="Discover research connections... (e.g., machine learning, climate change)"
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                onFocus={() => setShowSuggestions(searchQuery.length > 1)}
                className="search-input"
                disabled={loading}
              />
              
              {/* Loading Indicator */}
              {loading && (
                <motion.div
                  className="search-loading"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <RefreshCw size={16} />
                </motion.div>
              )}

              {/* Filter Section */}
              <div className="filter-section" ref={filtersRef}>
                <motion.button
                  type="button"
                  className={`filter-toggle-btn ${showFilters ? 'active' : ''}`}
                  onClick={() => setShowFilters(!showFilters)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  title="Search Filters"
                >
                  <Filter size={18} />
                  <ChevronDown size={14} className={`chevron ${showFilters ? 'rotated' : ''}`} />
                </motion.button>

                {/* Advanced Filters Dropdown */}
                <AnimatePresence>
                  {showFilters && (
                    <motion.div
                      className="filters-dropdown"
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="filters-content">
                        {/* Data Sources */}
                        <div className="filter-group">
                          <div className="filter-label">
                            <Database size={16} />
                            <span>Data Sources</span>
                          </div>
                          <div className="filter-options">
                            {[
                              { key: 'core', label: 'CORE', desc: 'Open access papers' },
                              { key: 'arxiv', label: 'arXiv', desc: 'Preprint repository' },
                              { key: 'pubmed', label: 'PubMed', desc: 'Medical literature' }
                            ].map(source => (
                              <label key={source.key} className="filter-checkbox">
                                <input
                                  type="checkbox"
                                  checked={searchFilters.sources.includes(source.key)}
                                  onChange={() => toggleSource(source.key)}
                                />
                                <span className="checkbox-custom"></span>
                                <div className="checkbox-text">
                                  <span className="source-name">{source.label}</span>
                                  <span className="source-desc">{source.desc}</span>
                                </div>
                              </label>
                            ))}
                          </div>
                        </div>

                        {/* Date Range */}
                        <div className="filter-group">
                          <div className="filter-label">
                            <Calendar size={16} />
                            <span>Publication Date</span>
                          </div>
                          <select
                            className="filter-select"
                            value={searchFilters.dateRange}
                            onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                          >
                            <option value="all">All Time</option>
                            <option value="year">Past Year</option>
                            <option value="3years">Past 3 Years</option>
                            <option value="5years">Past 5 Years</option>
                            <option value="decade">Past Decade</option>
                          </select>
                        </div>

                        {/* Results Limit */}
                        <div className="filter-group">
                          <div className="filter-label">
                            <BarChart3 size={16} />
                            <span>Max Results</span>
                          </div>
                          <select
                            className="filter-select"
                            value={searchFilters.maxResults}
                            onChange={(e) => handleFilterChange('maxResults', parseInt(e.target.value))}
                          >
                            <option value={25}>25 Papers</option>
                            <option value={50}>50 Papers</option>
                            <option value={100}>100 Papers</option>
                            <option value={200}>200 Papers</option>
                          </select>
                        </div>

                        {/* Analysis Depth */}
                        <div className="filter-group">
                          <div className="filter-label">
                            <Sparkles size={16} />
                            <span>Analysis Depth</span>
                          </div>
                          <select
                            className="filter-select"
                            value={searchFilters.analysisDepth}
                            onChange={(e) => handleFilterChange('analysisDepth', e.target.value)}
                          >
                            <option value="basic">Basic</option>
                            <option value="detailed">Detailed</option>
                            <option value="comprehensive">Comprehensive</option>
                          </select>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Search Button */}
              <motion.button
                type="submit"
                className="search-btn"
                disabled={loading || !searchQuery.trim()}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Zap size={18} />
                <span>Discover</span>
              </motion.button>
            </div>
          </form>

          {/* Search Suggestions */}
          <AnimatePresence>
            {showSuggestions && filteredSuggestions.length > 0 && (
              <motion.div
                className="suggestions-dropdown"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <div className="suggestions-header">
                  <Sparkles size={14} />
                  <span>Suggested Research Topics</span>
                </div>
                <div className="suggestions-list">
                  {filteredSuggestions.map((suggestion, index) => (
                    <motion.button
                      key={suggestion.text}
                      className="suggestion-item"
                      onClick={() => handleSuggestionSelect(suggestion.text)}
                      whileHover={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <suggestion.icon size={16} className="suggestion-icon" />
                      <div className="suggestion-content">
                        <span className="suggestion-text">{suggestion.text}</span>
                        <span className="suggestion-category">{suggestion.category}</span>
                      </div>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Action Controls */}
        <div className="action-controls">
          {/* Statistics Display */}
          {hasData && statistics && (
            <motion.div
              className="stats-display"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="stat-item">
                <span className="stat-value">{statistics.totalPapers}</span>
                <span className="stat-label">Papers</span>
              </div>
              <div className="stat-separator" />
              <div className="stat-item">
                <span className="stat-value">{statistics.totalConnections}</span>
                <span className="stat-label">Links</span>
              </div>
            </motion.div>
          )}

          {/* Control Buttons */}
          {hasData && (
            <div className="control-buttons">
              <motion.button
                className={`control-btn ${showControls ? 'active' : ''}`}
                onClick={onToggleControls}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Toggle Graph Controls"
              >
                <Settings size={18} />
              </motion.button>

              <motion.button
                className={`control-btn ${showInspector ? 'active' : ''}`}
                onClick={onToggleInspector}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Toggle Node Inspector"
              >
                <Info size={18} />
              </motion.button>
            </div>
          )}

          {/* New Search Button */}
          <motion.button
            className="new-search-btn"
            onClick={onNewSearch}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title="Start New Search"
          >
            <Plus size={18} />
            <span>New</span>
          </motion.button>
        </div>
      </div>
    </motion.header>
  )
}

export default TopBar
