import React, { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster } from 'react-hot-toast'

// Layout Components
import TopBar from './components/layout/TopBar'
import MainContent from './components/layout/MainContent'
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
  const [currentView, setCurrentView] = useState('welcome')
  const [selectedNode, setSelectedNode] = useState(null)
  const [hoveredNode, setHoveredNode] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchFilters, setSearchFilters] = useState({
    sources: ['core', 'arxiv', 'pubmed'],
    dateRange: 'all',
    minCitations: 0,
    maxResults: 50,
    analysisDepth: 'detailed'
  })
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
    loadingDetails,
    clearData
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
    'cmd+shift+x': () => {
      clearData()
      setCurrentView('welcome')
      setSearchQuery('')
      addNotification({ type: 'info', message: 'Session cleared' })
    }
  })

  // Handlers
  const handleSearch = useCallback(async (query, filters = searchFilters) => {
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
          message: `Discovered ${result.nodes.length} papers with ${result.links?.length || 0} connections` 
        })
      } else {
        setCurrentView('welcome')
        addNotification({ type: 'warning', message: 'No research connections found. Try different keywords.' })
      }
    } catch (err) {
      setCurrentView('error')
      console.error('Search error:', err)
      addNotification({ type: 'error', message: 'Search failed. Please try again.' })
    }
  }, [searchPapers, addNotification, searchFilters])

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

  const handleExport = useCallback(async (format) => {
    try {
      await exportGraph(format)
      addNotification({ type: 'success', message: `Graph exported as ${format.toUpperCase()}` })
    } catch (error) {
      addNotification({ type: 'error', message: 'Export failed' })
    }
  }, [exportGraph, addNotification])

  const handleNewSearch = useCallback(() => {
    clearData()
    setCurrentView('welcome')
    setSearchQuery('')
    setSelectedNode(null)
    setShowInspector(false)
    setShowControls(false)
    searchInputRef.current?.focus()
  }, [clearData])

  // Effect for updating current view based on data state
  useEffect(() => {
    if (loading) {
      setCurrentView('loading')
    } else if (error) {
      setCurrentView('error')
    } else if (graphData && graphData.nodes?.length > 0) {
      setCurrentView('graph')
    }
  }, [loading, error, graphData])

  // Get filtered data for display
  const displayData = getFilteredData() || graphData

  return (
    <ErrorBoundary>
      <div className="app-container">
        {/* Fixed Top Bar - Professional Header */}
        <TopBar
          onSearch={handleSearch}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          searchInputRef={searchInputRef}
          searchFilters={searchFilters}
          onFiltersChange={setSearchFilters}
          loading={loading}
          hasData={!!displayData?.nodes?.length}
          onToggleControls={() => setShowControls(!showControls)}
          onToggleInspector={() => setShowInspector(!showInspector)}
          showControls={showControls}
          showInspector={showInspector}
          onNewSearch={handleNewSearch}
          statistics={statistics}
        />

        {/* Main Content Area - Properly Positioned */}
        <MainContent currentView={currentView}>
          <AnimatePresence mode="wait">
            {/* Welcome State */}
            {currentView === 'welcome' && (
              <motion.div
                key="welcome"
                className="welcome-state"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.6 }}
              >
                <div className="welcome-content">
                  <motion.div
                    className="welcome-hero"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                  >
                    <div className="hero-icon">
                      <div className="neural-network-large">
                        {[...Array(12)].map((_, i) => (
                          <motion.div
                            key={i}
                            className="neural-node"
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{
                              delay: 0.5 + i * 0.1,
                              duration: 0.5,
                              type: "spring"
                            }}
                          />
                        ))}
                      </div>
                    </div>
                    <h1 className="hero-title">
                      Explore Research
                      <span className="gradient-text">Connections</span>
                    </h1>
                    <p className="hero-description">
                      Discover hidden patterns and relationships in scientific literature 
                      using AI-powered knowledge graph visualization.
                    </p>
                    <motion.button
                      className="cta-button"
                      onClick={() => searchInputRef.current?.focus()}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1.0 }}
                    >
                      Start Exploring
                    </motion.button>
                  </motion.div>

                  {/* Quick Topic Suggestions */}
                  <motion.div
                    className="quick-topics"
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 1.2 }}
                  >
                    <h3>Popular Research Areas</h3>
                    <div className="topic-grid">
                      {[
                        'Machine Learning', 'Computer Vision', 'Natural Language Processing',
                        'Healthcare AI', 'Climate Science', 'Quantum Computing'
                      ].map((topic, index) => (
                        <motion.button
                          key={topic}
                          className="topic-card"
                          onClick={() => handleSearch(topic)}
                          whileHover={{ scale: 1.02, y: -2 }}
                          whileTap={{ scale: 0.98 }}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.4 + index * 0.1 }}
                        >
                          {topic}
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Loading Screen */}
            {currentView === 'loading' && (
              <LoadingScreen
                key="loading"
                message="Discovering research connections..."
                progress={45}
              />
            )}

            {/* Graph Visualization */}
            {currentView === 'graph' && displayData && (
              <GraphViewer
                key="graph"
                ref={graphContainerRef}
                data={displayData}
                mode="2d"
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
                  <h3>Research Discovery Failed</h3>
                  <p>{error?.message || 'Unable to discover research connections'}</p>
                  <div className="error-actions">
                    <button 
                      onClick={() => handleSearch(searchQuery)}
                      className="retry-btn"
                    >
                      Try Again
                    </button>
                    <button 
                      onClick={handleNewSearch}
                      className="reset-btn"
                    >
                      New Search
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </MainContent>

        {/* Side Panels - Properly Positioned */}
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
            />
          )}
        </AnimatePresence>

        {/* Bottom Status Bar - Separate Component */}
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

        {/* Toast Notifications - Top Layer */}
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            className: 'custom-toast',
            style: {
              background: 'rgba(15, 23, 42, 0.95)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              color: '#f1f5f9',
              borderRadius: '12px',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
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
