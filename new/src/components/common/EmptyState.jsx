import React from 'react'
import { motion } from 'framer-motion'
import { Search, Zap, Globe } from 'lucide-react'

const EmptyState = ({ 
  title = "No Data Available",
  message = "Start exploring to see results here",
  actionText = "Get Started",
  onAction,
  icon: Icon = Search
}) => {
  return (
    <motion.div
      className="empty-state"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="empty-content">
        <motion.div
          className="empty-icon"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 260, damping: 20 }}
        >
          <Icon size={48} />
        </motion.div>
        
        <h3>{title}</h3>
        <p>{message}</p>
        
        {onAction && (
          <motion.button
            className="empty-action-btn"
            onClick={onAction}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Zap size={16} />
            {actionText}
          </motion.button>
        )}
      </div>
    </motion.div>
  )
}

export default EmptyState
