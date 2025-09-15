import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const MainContent = ({ currentView, children }) => {
  return (
    <motion.main 
      className="main-content"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className={`content-wrapper view-${currentView}`}>
        <AnimatePresence mode="wait">
          {children}
        </AnimatePresence>
      </div>
      
      {/* Background Grid Pattern */}
      <div className="background-pattern" />
      
      {/* Ambient Lighting Effects */}
      <div className="ambient-lights">
        <div className="light-orb orb-1" />
        <div className="light-orb orb-2" />
        <div className="light-orb orb-3" />
      </div>
    </motion.main>
  )
}

export default MainContent
