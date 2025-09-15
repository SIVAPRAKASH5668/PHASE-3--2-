import React, { createContext, useContext, useReducer, useEffect } from 'react'

const AppContext = createContext()

const initialState = {
  theme: 'dark',
  graphMode: '2d', // Keep 2D for stability
  sidebarCollapsed: false,
  notifications: [],
  user: null,
  preferences: {
    defaultView: '2d',
    autoSave: true,
    showTutorials: true,
    animationSpeed: 1,
    showTooltips: true,
    enableKeyboardShortcuts: true
  },
  ui: {
    showControls: false,
    showInspector: false,
    fullscreen: false,
    loading: false
  }
}

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_THEME':
      return { ...state, theme: action.payload }
    
    case 'SET_GRAPH_MODE':
      // Allow 2D mode, warn about 3D without blocking
      return { ...state, graphMode: action.payload }
    
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed }
    
    case 'SET_UI_STATE':
      return {
        ...state,
        ui: { ...state.ui, ...action.payload }
      }
    
    case 'ADD_NOTIFICATION':
      const notification = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        ...action.payload
      }
      
      return {
        ...state,
        notifications: [...state.notifications, notification]
      }
    
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      }
    
    case 'CLEAR_NOTIFICATIONS':
      return {
        ...state,
        notifications: []
      }
    
    case 'UPDATE_PREFERENCES':
      return {
        ...state,
        preferences: { ...state.preferences, ...action.payload }
      }
    
    case 'SET_USER':
      return { ...state, user: action.payload }
    
    case 'RESET_STATE':
      return { ...initialState, preferences: state.preferences }
    
    default:
      return state
  }
}

export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState)

  // Load preferences from localStorage
  useEffect(() => {
    try {
      const savedPreferences = localStorage.getItem('research-graph-preferences')
      if (savedPreferences) {
        const parsed = JSON.parse(savedPreferences)
        // Ensure 2D mode is default
        parsed.defaultView = '2d'
        dispatch({
          type: 'UPDATE_PREFERENCES',
          payload: parsed
        })
      }
    } catch (error) {
      console.error('Failed to load preferences:', error)
    }
  }, [])

  // Save preferences to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('research-graph-preferences', JSON.stringify(state.preferences))
    } catch (error) {
      console.error('Failed to save preferences:', error)
    }
  }, [state.preferences])

  // Auto-clear old notifications
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now()
      const tenSecondsAgo = now - 10000
      
      // Only clear if there are notifications older than 10 seconds
      const hasOldNotifications = state.notifications.some(
        n => new Date(n.timestamp).getTime() < tenSecondsAgo
      )
      
      if (hasOldNotifications) {
        dispatch({
          type: 'CLEAR_NOTIFICATIONS'
        })
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [state.notifications])

  const value = {
    ...state,
    dispatch,
    
    // Theme actions
    setTheme: (theme) => dispatch({ type: 'SET_THEME', payload: theme }),
    
    // Graph actions
    setGraphMode: (mode) => dispatch({ type: 'SET_GRAPH_MODE', payload: mode }),
    
    // UI actions
    toggleSidebar: () => dispatch({ type: 'TOGGLE_SIDEBAR' }),
    setUIState: (uiState) => dispatch({ type: 'SET_UI_STATE', payload: uiState }),
    
    // 🔧 FIXED: Pure notification actions (no toast calls)
    addNotification: (notification) => dispatch({ type: 'ADD_NOTIFICATION', payload: notification }),
    removeNotification: (id) => dispatch({ type: 'REMOVE_NOTIFICATION', payload: id }),
    clearNotifications: () => dispatch({ type: 'CLEAR_NOTIFICATIONS' }),
    
    // Preference actions
    updatePreferences: (prefs) => dispatch({ type: 'UPDATE_PREFERENCES', payload: prefs }),
    
    // User actions
    setUser: (user) => dispatch({ type: 'SET_USER', payload: user }),
    
    // Reset
    resetState: () => dispatch({ type: 'RESET_STATE' }),
    
    // 🔧 FIXED: Pure convenience methods (no side effects in render)
    showSuccess: (message) => dispatch({ 
      type: 'ADD_NOTIFICATION', 
      payload: { type: 'success', message } 
    }),
    showError: (message) => dispatch({ 
      type: 'ADD_NOTIFICATION', 
      payload: { type: 'error', message } 
    }),
    showWarning: (message) => dispatch({ 
      type: 'ADD_NOTIFICATION', 
      payload: { type: 'warning', message } 
    }),
    showInfo: (message) => dispatch({ 
      type: 'ADD_NOTIFICATION', 
      payload: { type: 'info', message } 
    })
  }

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  )
}

export const useAppContext = () => {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider')
  }
  return context
}

// Utility hooks for specific context slices
export const useTheme = () => {
  const { theme, setTheme } = useAppContext()
  return { theme, setTheme }
}

export const useNotifications = () => {
  const { 
    notifications, 
    addNotification, 
    removeNotification, 
    clearNotifications,
    showSuccess,
    showError,
    showWarning,
    showInfo
  } = useAppContext()
  
  return { 
    notifications, 
    addNotification, 
    removeNotification, 
    clearNotifications,
    showSuccess,
    showError,
    showWarning,
    showInfo
  }
}

export const usePreferences = () => {
  const { preferences, updatePreferences } = useAppContext()
  return { preferences, updatePreferences }
}
