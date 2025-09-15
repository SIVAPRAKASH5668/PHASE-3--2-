import React from 'react'
import { motion } from 'framer-motion'
import { Clock, TrendingUp } from 'lucide-react'

const SearchAutocomplete = ({ query, onSelect, suggestions }) => {
  const filteredSuggestions = suggestions.filter(suggestion =>
    suggestion.text.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <motion.div
      className="search-autocomplete"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
    >
      <div className="autocomplete-content">
        {filteredSuggestions.length > 0 && (
          <div className="suggestions-section">
            <div className="section-header">
              <TrendingUp size={14} />
              <span>Suggestions</span>
            </div>
            {filteredSuggestions.map((suggestion, index) => (
              <motion.button
                key={index}
                className="suggestion-item"
                onClick={() => onSelect(suggestion.text)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
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
        )}
        
        {query && (
          <div className="quick-search-section">
            <div className="section-header">
              <Clock size={14} />
              <span>Search for</span>
            </div>
            <motion.button
              className="quick-search-item"
              onClick={() => onSelect(query)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span>"{query}"</span>
            </motion.button>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default SearchAutocomplete
