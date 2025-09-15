import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Search,
  Zap,
  Globe,
  Brain,
  Microscope,
  Sparkles,
  BookOpen,
  TrendingUp,
  Clock,
  Filter
} from 'lucide-react'

const SearchInterface = ({
  onSearch,
  searchInputRef,
  recentSearches = [],
  popularTopics = []
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const handleSearch = (query) => {
    if (query.trim()) {
      onSearch(query.trim())
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    handleSearch(searchQuery)
  }

  const quickTopics = [
    { text: 'Machine Learning', icon: Brain, color: 'from-blue-500 to-indigo-600' },
    { text: 'Healthcare AI', icon: BookOpen, color: 'from-green-500 to-emerald-600' },
    { text: 'Computer Vision', icon: Sparkles, color: 'from-purple-500 to-violet-600' },
    { text: 'Climate Science', icon: Globe, color: 'from-orange-500 to-red-600' },
    { text: 'Quantum Computing', icon: Zap, color: 'from-teal-500 to-cyan-600' },
    { text: 'Robotics', icon: Microscope, color: 'from-pink-500 to-rose-600' }
  ]

  return (
    <motion.div
      className="search-interface"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      {/* Hero Section */}
      <div className="hero-section">
        {/* Neural Network Animation */}
        <motion.div
          className="neural-network"
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
        >
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="neural-node"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{
                delay: 0.5 + i * 0.1,
                duration: 0.5,
                type: "spring",
                stiffness: 260,
                damping: 20
              }}
            />
          ))}
          <svg className="neural-connections" viewBox="0 0 200 200">
            {[...Array(12)].map((_, i) => (
              <motion.line
                key={i}
                x1={Math.random() * 200}
                y1={Math.random() * 200}
                x2={Math.random() * 200}
                y2={Math.random() * 200}
                stroke="rgba(59, 130, 246, 0.3)"
                strokeWidth="1"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.6 }}
                transition={{
                  duration: 2,
                  delay: 0.8 + i * 0.1,
                  ease: "easeInOut"
                }}
              />
            ))}
          </svg>
        </motion.div>

        {/* Main Title */}
        <motion.div
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <h1 className="hero-title">
            Discover Research
            <span className="gradient-text">Connections</span>
          </h1>
          <p className="hero-description">
            Explore the interconnected world of scientific research with AI-powered discovery.
            Uncover hidden patterns, relationships, and insights across millions of papers.
          </p>
        </motion.div>

        {/* Main Search */}
        <motion.div
          className="main-search"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <form onSubmit={handleSubmit} className="search-form-main">
            <div className="search-input-main-wrapper">
              <Search size={24} className="search-icon-main" />
              <input
                ref={searchInputRef}
                type="text"
                placeholder="Enter your research query..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-input-main"
              />
              <motion.button
                type="button"
                className="filter-btn"
                onClick={() => setShowFilters(!showFilters)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Filter size={20} />
              </motion.button>
              <motion.button
                type="submit"
                className="search-btn-main"
                disabled={!searchQuery.trim()}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Zap size={20} />
                Discover
              </motion.button>
            </div>
          </form>
        </motion.div>
      </div>

      {/* Quick Topics */}
      <motion.div
        className="quick-topics-section"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.0 }}
      >
        <h3>Quick Start Topics</h3>
        <div className="quick-topics-grid">
          {quickTopics.map((topic, index) => (
            <motion.button
              key={topic.text}
              className={`topic-card bg-gradient-to-r ${topic.color}`}
              onClick={() => handleSearch(topic.text)}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 + index * 0.1 }}
            >
              <topic.icon size={24} />
              <span>{topic.text}</span>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Categories Section */}
      <motion.div
        className="categories-section"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.4 }}
      >
        <h3>Popular Research Areas</h3>
        <div className="categories-grid">
          {popularTopics.map((category, index) => (
            <motion.div
              key={category}
              className="category-card"
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleSearch(category)}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.6 + index * 0.05 }}
            >
              <div className="category-icon">
                <TrendingUp size={20} />
              </div>
              <h4>{category}</h4>
              <p>Explore cutting-edge research in {category.toLowerCase()}</p>
              <div className="category-stats">
                <span>{Math.floor(Math.random() * 10000) + 1000}+ papers</span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Recent Searches */}
      {recentSearches.length > 0 && (
        <motion.div
          className="recent-searches-section"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.8 }}
        >
          <h3>Recent Searches</h3>
          <div className="recent-searches">
            {recentSearches.map((search, index) => (
              <motion.button
                key={index}
                className="recent-search-item"
                onClick={() => handleSearch(search)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Clock size={16} />
                {search}
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default SearchInterface
