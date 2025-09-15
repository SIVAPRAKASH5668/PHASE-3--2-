import React, { createContext, useContext, useReducer, useEffect } from 'react'

const AppContext = createContext()

const initialState = {
  theme: 'dark',
  graphMode: '3d',
  sidebarCollapsed: false,
  notifications: [],
  user: null,
  preferences: {
    defaultView: '3d',
    autoSave: true,
    showTutorials: true,
    animationSpeed: 1
  }
}

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_THEME':
      return { ...state, theme: action.payload }
    
    case 'SET_GRAPH_MODE':
      return { ...state, graphMode: action.payload }
    
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed }
    
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [...state.notifications, {
          id: Date.now(),
          ...action.payload
        }]
      }
    
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      }
    
    case 'UPDATE_PREFERENCES':
      return {
        ...state,
        preferences: { ...state.preferences, ...action.payload }
      }
    
    case 'SET_USER':
      return { ...state, user: action.payload }
    
    default:
      return state
  }
}

export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState)

  // Load preferences from localStorage
  useEffect(() => {
    try {
      const savedPreferences = localStorage.getItem('research-app-preferences')
      if (savedPreferences) {
        dispatch({
          type: 'UPDATE_PREFERENCES',
          payload: JSON.parse(savedPreferences)
        })
      }
    } catch (error) {
      console.error('Failed to load preferences:', error)
    }
  }, [])

  // Save preferences to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('research-app-preferences', JSON.stringify(state.preferences))
    } catch (error) {
      console.error('Failed to save preferences:', error)
    }
  }, [state.preferences])

  const value = {
    ...state,
    dispatch,
    // Action creators
    setTheme: (theme) => dispatch({ type: 'SET_THEME', payload: theme }),
    setGraphMode: (mode) => dispatch({ type: 'SET_GRAPH_MODE', payload: mode }),
    toggleSidebar: () => dispatch({ type: 'TOGGLE_SIDEBAR' }),
    addNotification: (notification) => dispatch({ type: 'ADD_NOTIFICATION', payload: notification }),
    removeNotification: (id) => dispatch({ type: 'REMOVE_NOTIFICATION', payload: id }),
    updatePreferences: (prefs) => dispatch({ type: 'UPDATE_PREFERENCES', payload: prefs }),
    setUser: (user) => dispatch({ type: 'SET_USER', payload: user })
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
