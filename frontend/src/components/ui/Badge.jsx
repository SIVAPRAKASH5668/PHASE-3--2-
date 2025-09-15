import React from 'react'
import { motion } from 'framer-motion'

const Badge = ({
  children,
  variant = 'default',
  size = 'md',
  className = '',
  onClick,
  ...props
}) => {
  const baseClass = 'badge'
  const variantClass = `badge-${variant}`
  const sizeClass = `badge-${size}`
  const clickableClass = onClick ? 'badge-clickable' : ''

  const Component = onClick ? motion.button : motion.span

  return (
    <Component
      className={`${baseClass} ${variantClass} ${sizeClass} ${clickableClass} ${className}`}
      onClick={onClick}
      whileHover={onClick ? { scale: 1.05 } : {}}
      whileTap={onClick ? { scale: 0.95 } : {}}
      {...props}
    >
      {children}
    </Component>
  )
}

export default Badge
