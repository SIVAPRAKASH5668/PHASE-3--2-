import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, BookOpen, Microscope, Sparkles, Zap, Filter } from 'lucide-react'

const QuickActions = ({ onQuickSearch, showFilters }) => {
  const quickTopics = [
    { text: 'Machine Learning', icon: Brain, color: 'from-blue-500 to-indigo-600' },
    { text: 'Healthcare AI', icon: BookOpen, color: 'from-green-500 to-emerald-600' },
    { text: 'Computer Vision', icon: Sparkles, color: 'from-purple-500 to-violet-600' },
    { text: 'Physics Research', icon: Microscope, color: 'from-orange-500 to-red-600' },
    { text: 'Climate Science', icon: Zap, color: 'from-teal-500 to-cyan-600' }
  ]

  return (
    <motion.div
      className="quick-actions-bar"
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="quick-actions-content">
        <div className="actions-section">
          <span className="section-label">Quick Topics:</span>
          <div className="topic-buttons">
            {quickTopics.map((topic, index) => (
              <motion.button
                key={topic.text}
                className={`topic-btn bg-gradient-to-r ${topic.color}`}
                onClick={() => onQuickSearch(topic.text)}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <topic.icon size={14} />
                {topic.text}
              </motion.button>
            ))}
          </div>
        </div>

        <AnimatePresence>
          {showFilters && (
            <motion.div
              className="filters-section"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="filter-controls">
                <div className="filter-group">
                  <Filter size={14} />
                  <span>Advanced Filters</span>
                </div>
                <div className="filter-options">
                  <select className="filter-select">
                    <option value="">All Years</option>
                    <option value="2024">2024</option>
                    <option value="2023">2023</option>
                    <option value="2022">2022</option>
                  </select>
                  <select className="filter-select">
                    <option value="">All Sources</option>
                    <option value="arxiv">ArXiv</option>
                    <option value="pubmed">PubMed</option>
                    <option value="ieee">IEEE</option>
                  </select>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default QuickActions
