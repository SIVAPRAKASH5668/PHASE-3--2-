import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Database, Network, Zap } from 'lucide-react'

const LoadingScreen = ({ message = "Processing...", progress = 0 }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [timeElapsed, setTimeElapsed] = useState(0)

  const steps = [
    { icon: Database, text: "Connecting to TiDB vector database...", duration: 30 },
    { icon: Brain, text: "Processing query with AI agents...", duration: 60 },
    { icon: Network, text: "Building research knowledge graph...", duration: 120 },
    { icon: Zap, text: "Optimizing visualization layout...", duration: 60 }
  ]

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeElapsed(prev => {
        const newElapsed = prev + 1
        
        // Update step based on elapsed time
        let totalDuration = 0
        for (let i = 0; i < steps.length; i++) {
          totalDuration += steps[i].duration
          if (newElapsed <= totalDuration) {
            setCurrentStep(i)
            break
          }
        }
        
        return newElapsed
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const calculateProgress = () => {
    if (progress > 0) return progress
    // Calculate based on time (5 minutes = 100%)
    return Math.min(95, (timeElapsed / 300) * 100)
  }

  return (
    <motion.div
      className="loading-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="loading-content">
        {/* Neural Network Animation */}
        <motion.div
          className="neural-network-loader"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.8 }}
        >
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="neural-node-loader"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                delay: i * 0.2,
                duration: 0.5,
                repeat: Infinity,
                repeatType: "reverse",
                repeatDelay: 2
              }}
            />
          ))}
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {message}
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          TiDB AgentX is processing your research query...
          <br />
          <strong>Elapsed: {formatTime(timeElapsed)}</strong> | Expected: ~3-5 minutes
        </motion.p>

        {/* Progress Bar */}
        <motion.div
          className="progress-container"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.7 }}
        >
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${calculateProgress()}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <div className="progress-text">
            {Math.round(calculateProgress())}% Complete
            {timeElapsed > 180 && " - Almost done, building final graph..."}
          </div>
        </motion.div>

        {/* Loading Steps */}
        <motion.div
          className="loading-steps"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          {steps.map((step, index) => {
            const StepIcon = step.icon
            return (
              <motion.div
                key={index}
                className={`loading-step ${index === currentStep ? 'active' : ''} ${index < currentStep ? 'completed' : ''}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.2 + index * 0.2 }}
              >
                <div className="step-indicator">
                  <StepIcon size={16} />
                </div>
                <span>{step.text}</span>
                {index === currentStep && (
                  <motion.div
                    className="step-spinner"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    <Zap size={12} />
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </motion.div>

        {/* Helpful Tip */}
        {timeElapsed > 120 && (
          <motion.div
            className="loading-tip"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <p>💡 <strong>Tip:</strong> Complex research queries with many connections take longer to process. 
            Your TiDB backend is working hard to build the perfect knowledge graph!</p>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}

export default LoadingScreen
