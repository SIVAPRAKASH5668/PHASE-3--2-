import React from 'react'
import { motion } from 'framer-motion'

const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  className = '',
  onClick,
  type = 'button',
  ...props
}) => {
  const baseClass = 'btn'
  const variantClass = `btn-${variant}`
  const sizeClass = `btn-${size}`
  const stateClass = loading ? 'btn-loading' : disabled ? 'btn-disabled' : ''

  return (
    <motion.button
      type={type}
      className={`${baseClass} ${variantClass} ${sizeClass} ${stateClass} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled && !loading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !loading ? { scale: 0.98 } : {}}
      {...props}
    >
      {loading && (
        <motion.div
          className="btn-spinner"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
      )}
      <span className={loading ? 'btn-content-hidden' : 'btn-content'}>
        {children}
      </span>
    </motion.button>
  )
}

export default Button
