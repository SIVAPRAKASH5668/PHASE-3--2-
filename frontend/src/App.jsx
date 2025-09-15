import React, { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster } from 'react-hot-toast'

// Layout Components
import TopBar from './components/layout/TopBar'
import MainContent from './components/layout/MainContent'
import SearchInterface from './components/search/SearchInterface'
import GraphViewer from './components/graph/GraphViewer'
import NodeInspector from './components/panels/NodeInspector'
import ControlPanel from './components/panels/controlPanel'
import StatusBar from './components/layout/StatusBar'

// UI Components  
import LoadingScreen from './components/ui/LoadingScreen'
import ErrorBoundary from './components/common/ErrorBoundary'

// Hooks and Context
import { useResearchData } from './hooks/useResearchData'
import { useGraphVisualization } from './hooks/useGraphVisualization'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { AppProvider, useAppContext } from './contexts/AppContext'

// Styles
import './styles/modern.css'
import './styles/components.css'
import './styles/animations.css'
import './styles/responsive.css'

function AppCore() {
  // App State
  const [currentView, setCurrentView] = useState('empty')
  const [selectedNode, setSelectedNode] = useState(null)
  const [hoveredNode, setHoveredNode] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [graphMode, setGraphMode] = useState('2d') // 🔧 FORCE 2D MODE
  const [showControls, setShowControls] = useState(false)
  const [showInspector, setShowInspector] = useState(false)

  // Refs
  const graphContainerRef = useRef(null)
  const searchInputRef = useRef(null)

  // Context
  const { addNotification } = useAppContext()

  // Hooks
  const {
    data: graphData,
    loading,
    error,
    searchPapers,
    getNodeDetails,
    nodeDetails,
    statistics,
    loadingDetails
  } = useResearchData()

  const {
    graphConfig,
    updateGraphConfig,
    resetView,
    focusNode,
    exportGraph,
    getFilteredData
  } = useGraphVisualization(graphData)

  // Keyboard shortcuts
  useKeyboardShortcuts({
    'cmd+k': () => {
      searchInputRef.current?.focus()
      addNotification({ type: 'info', message: 'Search focused' })
    },
    'cmd+shift+c': () => setShowControls(!showControls),
    'cmd+shift+i': () => setShowInspector(!showInspector),
    'escape': () => {
      setSelectedNode(null)
      setShowInspector(false)
    },
    'f': () => {
      if (graphContainerRef.current) {
        if (document.fullscreenElement) {
          document.exitFullscreen()
        } else {
          graphContainerRef.current.requestFullscreen()
        }
      }
    },
    'r': () => resetView(),
    // 🔧 DISABLE 3D SHORTCUT
    '3': () => {
      console.warn('⚠️ 3D mode temporarily disabled')
      addNotification({ 
        type: 'warning', 
        message: '3D mode temporarily disabled due to technical issues' 
      })
    }
  })

  // Handlers
  const handleSearch = useCallback(async (query, filters = {}) => {
    if (!query.trim()) return
    
    setCurrentView('loading')
    setSelectedNode(null)
    setSearchQuery(query)
    
    try {
      const result = await searchPapers(query, filters)
      if (result && result.nodes?.length > 0) {
        setCurrentView('graph')
        addNotification({ 
          type: 'success', 
          message: `Found ${result.nodes.length} papers with ${result.links?.length || 0} connections` 
        })
      } else {
        setCurrentView('empty')
        addNotification({ type: 'warning', message: 'No results found' })
      }
    } catch (err) {
      setCurrentView('error')
      console.error('Search error:', err)
    }
  }, [searchPapers, addNotification])

  const handleNodeInteraction = useCallback(async (nodeId, interactionType) => {
    switch (interactionType) {
      case 'hover':
        setHoveredNode(nodeId)
        break
      case 'click':
        if (nodeId) {
          setSelectedNode(nodeId)
          setShowInspector(true)
          await getNodeDetails(nodeId)
          focusNode(nodeId)
        } else {
          setSelectedNode(null)
          setShowInspector(false)
        }
        break
      case 'unhover':
        setHoveredNode(null)
        break
    }
  }, [getNodeDetails, focusNode])

  // 🔧 BLOCK 3D MODE CHANGES
  const handleViewChange = useCallback((newView) => {
    if (newView === '3d') {
      console.warn('⚠️ 3D mode blocked at app level')
      addNotification({ 
        type: 'warning', 
        message: '3D mode temporarily disabled while fixing connection display issues' 
      })
      return
    }
    setGraphMode(newView)
    addNotification({ type: 'info', message: `Switched to ${newView.toUpperCase()} view` })
  }, [addNotification])

  const handleExport = useCallback(async (format) => {
    try {
      await exportGraph(format)
      addNotification({ type: 'success', message: `Graph exported as ${format.toUpperCase()}` })
    } catch (error) {
      addNotification({ type: 'error', message: 'Export failed' })
    }
  }, [exportGraph, addNotification])

  // Effect for updating current view based on data state
  useEffect(() => {
    if (loading) {
      setCurrentView('loading')
    } else if (error) {
      setCurrentView('error')
    } else if (graphData && graphData.nodes?.length > 0) {
      setCurrentView('graph')
    } else if (searchQuery && !loading) {
      setCurrentView('empty')
    }
  }, [loading, error, graphData, searchQuery])

  // 🔧 FORCE 2D MODE ON DATA LOAD
  useEffect(() => {
    if (graphData && graphMode !== '2d') {
      setGraphMode('2d')
      console.log('🔧 Forced 2D mode for stable connections')
    }
  }, [graphData, graphMode])

  // Get filtered data for display
  const displayData = getFilteredData() || graphData

  return (
    <ErrorBoundary>
      <div className="app-container">
        {/* Top Navigation Bar */}
        <TopBar
          onSearch={handleSearch}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          searchInputRef={searchInputRef}
          loading={loading}
          graphMode={graphMode} // Will always be 2D
          onGraphModeChange={handleViewChange} // Will block 3D
          hasData={!!displayData?.nodes?.length}
          onToggleControls={() => setShowControls(!showControls)}
          onToggleInspector={() => setShowInspector(!showInspector)}
          showControls={showControls}
          showInspector={showInspector}
        />

        {/* Main Application Content */}
        <MainContent currentView={currentView}>
          <AnimatePresence mode="wait">
            {/* Search Interface - Empty State */}
            {currentView === 'empty' && (
              <SearchInterface
                key="search"
                onSearch={handleSearch}
                searchInputRef={searchInputRef}
                recentSearches={[]}
                popularTopics={[
                  'Machine Learning',
                  'Computer Vision', 
                  'Natural Language Processing',
                  'Healthcare AI',
                  'Climate Science',
                  'Quantum Computing',
                  'Deep Learning',
                  'Robotics'
                ]}
              />
            )}

            {/* Loading Screen */}
            {currentView === 'loading' && (
              <LoadingScreen
                key="loading"
                message="Discovering research connections..."
                progress={loading ? 45 : 0}
              />
            )}

            {/* Graph Visualization */}
            {currentView === 'graph' && displayData && (
              <GraphViewer
                key="graph"
                ref={graphContainerRef}
                data={displayData}
                mode="2d" // 🔧 HARDCODE 2D MODE
                config={graphConfig}
                selectedNode={selectedNode}
                hoveredNode={hoveredNode}
                onNodeInteraction={handleNodeInteraction}
                onConfigChange={updateGraphConfig}
              />
            )}

            {/* Error State */}
            {currentView === 'error' && (
              <motion.div
                key="error"
                className="error-state"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <div className="error-content">
                  <h3>Something went wrong</h3>
                  <p>{error?.message || 'Failed to load research data'}</p>
                  <div className="error-actions">
                    <button 
                      onClick={() => handleSearch(searchQuery)}
                      className="retry-btn"
                    >
                      Try Again
                    </button>
                    <button 
                      onClick={() => {
                        setCurrentView('empty')
                        setSearchQuery('')
                      }}
                      className="reset-btn"
                    >
                      Start Over
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </MainContent>

        {/* Side Panels */}
        <AnimatePresence>
          {/* Node Inspector Panel */}
          {showInspector && selectedNode && (
            <NodeInspector
              key="inspector"
              node={selectedNode}
              nodeDetails={nodeDetails}
              loading={loadingDetails}
              onClose={() => {
                setShowInspector(false)
                setSelectedNode(null)
              }}
              onRelatedNodeClick={handleNodeInteraction}
            />
          )}

          {/* Control Panel */}
          {showControls && currentView === 'graph' && (
            <ControlPanel
              key="controls"
              config={graphConfig}
              onConfigChange={updateGraphConfig}
              onResetView={resetView}
              onExport={handleExport}
              statistics={statistics}
              graphMode="2d" // 🔧 HARDCODE 2D
              onGraphModeChange={handleViewChange} // Will block 3D
            />
          )}
        </AnimatePresence>

        {/* Status Bar */}
        <AnimatePresence>
          {currentView === 'graph' && displayData && (
            <StatusBar
              key="status"
              nodeCount={displayData.nodes?.length || 0}
              linkCount={displayData.links?.length || 0}
              selectedNode={selectedNode}
              searchQuery={searchQuery}
              statistics={statistics}
            />
          )}
        </AnimatePresence>

        {/* Toast Notifications */}
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            className: 'custom-toast',
            style: {
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: '#fff'
            }
          }}
        />
      </div>
    </ErrorBoundary>
  )
}

function App() {
  return (
    <AppProvider>
      <AppCore />
    </AppProvider>
  )
}

export default App
