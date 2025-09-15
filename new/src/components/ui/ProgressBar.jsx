import React from 'react'
import { motion } from 'framer-motion'

const ProgressBar = ({ progress = 0, className = '', showPercentage = false }) => {
  return (
    <div className={`progress-bar ${className}`}>
      <div className="progress-track">
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
        <motion.div
          className="progress-glow"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      {showPercentage && (
        <span className="progress-percentage">
          {Math.round(progress)}%
        </span>
      )}
    </div>
  )
}

export default ProgressBar
