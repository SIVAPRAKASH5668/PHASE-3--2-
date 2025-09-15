import { useEffect } from 'react'

export const useKeyboardShortcuts = (shortcuts) => {
  useEffect(() => {
    const handleKeyDown = (event) => {
      const key = []
      
      if (event.ctrlKey || event.metaKey) key.push('cmd')
      if (event.shiftKey) key.push('shift')
      if (event.altKey) key.push('alt')
      
      // Add the actual key
      if (event.key === ' ') {
        key.push('space')
      } else if (event.key === 'Escape') {
        key.push('escape')
      } else {
        key.push(event.key.toLowerCase())
      }
      
      const shortcut = key.join('+')
      
      if (shortcuts[shortcut]) {
        event.preventDefault()
        shortcuts[shortcut]()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [shortcuts])
}
