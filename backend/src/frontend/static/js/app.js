let cy = null;
let currentQuery = '';
let selectedPaperId = null;

// Initialize Cytoscape graph
function initializeGraph() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'color': '#ffffff',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px',
                    'font-size': '12px',
                    'font-weight': 'bold',
                    'text-outline-width': 2,
                    'text-outline-color': '#000000'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 'mapData(weight, 0, 1, 2, 8)',
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                    'target-arrow-shape': 'triangle',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'color': '#666',
                    'curve-style': 'bezier'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#FFA500',
                    'border-opacity': 1
                }
            }
        ],
        
        layout: {
            name: 'cose',
            animate: true,
            animationDuration: 1000,
            nodeRepulsion: 400000,
            nodeOverlap: 10,
            idealEdgeLength: 100,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
            initialTemp: 200,
            coolingFactor: 0.95,
            minTemp: 1.0
        }
    });
    
    // Add event listeners
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        const data = node.data();
        selectedPaperId = data.id;
        
        showPaperDetails(data);
        highlightConnections(node);
    });
    
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            hideSidebar();
            cy.elements().removeClass('highlighted faded');
        }
    });
    
    // Add hover effects
    cy.on('mouseover', 'node', function(evt) {
        const node = evt.target;
        node.style('transform', 'scale(1.1)');
        showTooltip(evt, node.data());
    });
    
    cy.on('mouseout', 'node', function(evt) {
        const node = evt.target;
        node.style('transform', 'scale(1.0)');
        hideTooltip();
    });
}

// Search for papers and build graph
async function searchPapers() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;
    
    currentQuery = query;
    showLoading(true);
    
    try {
        const response = await fetch(`/api/graph/search?query=${encodeURIComponent(query)}&max_papers=20`);
        const data = await response.json();
        
        if (data.status === 'success') {
            buildGraph(data.graph);
            showInsights(data.graph.insights);
        } else {
            showError('Failed to generate graph');
        }
    } catch (error) {
        console.error('Search failed:', error);
        showError('Search failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Build the graph visualization
function buildGraph(graphData) {
    if (!cy) return;
    
    // Clear existing graph
    cy.elements().remove();
    
    // Add nodes
    const nodes = graphData.nodes.map(node => ({
        data: {
            id: node.id,
            label: node.label,
            color: getThemeColor(node.group),
            size: Math.max(30, Math.min(80, node.size)),
            title: node.title,
            summary: node.summary,
            themes: node.themes,
            contributions: node.contributions,
            methodology: node.methodology,
            innovation_score: node.innovation_score
        }
    }));
    
    // Add edges
    const edges = graphData.edges.map(edge => ({
        data: {
            source: edge.source,
            target: edge.target,
            label: edge.label,
            weight: edge.weight,
            color: edge.color,
            explanation: edge.explanation
        }
    }));
    
    // Add to graph
    cy.add([...nodes, ...edges]);
    
    // Run layout
    cy.layout({
        name: 'cose',
        animate: true,
        animationDuration: 2000
    }).run();
    
    // Fit to viewport
    setTimeout(() => {
        cy.fit();
        cy.center();
    }, 2500);
}

// Show paper details in sidebar
function showPaperDetails(nodeData) {
    document.getElementById('paper-title').textContent = nodeData.title;
    document.getElementById('paper-authors').textContent = `Innovation Score: ${nodeData.innovation_score}`;
    document.getElementById('paper-summary').innerHTML = `<strong>Summary:</strong> ${nodeData.summary}`;
    
    // Show themes
    const themesHtml = nodeData.themes ? 
        `<strong>Key Themes:</strong> ${nodeData.themes.join(', ')}` : '';
    document.getElementById('paper-themes').innerHTML = themesHtml;
    
    // Show contributions
    const contributionsHtml = nodeData.contributions ? 
        `<strong>Main Contributions:</strong><ul>${nodeData.contributions.map(c => `<li>${c}</li>`).join('')}</ul>` : '';
    document.getElementById('paper-contributions').innerHTML = contributionsHtml;
    
    // Show sidebar
    document.getElementById('sidebar').classList.remove('hidden');
}

// Expand connections for selected paper
async function expandConnections() {
    if (!selectedPaperId) return;
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/graph/expand/${selectedPaperId}?depth=1`);
        const data = await response.json();
        
        if (data.status === 'success') {
            addExpansionToGraph(data.expansion);
        }
    } catch (error) {
        console.error('Expansion failed:', error);
        showError('Failed to expand connections');
    } finally {
        showLoading(false);
    }
}

// Add expansion nodes to existing graph
function addExpansionToGraph(expansion) {
    const newNodes = expansion.nodes.map(node => ({
        data: {
            id: node.id,
            label: node.label,
            color: getThemeColor(node.group),
            size: Math.max(25, node.size),
            title: node.title,
            authors: node.authors,
            similarity: node.similarity
        }
    }));
    
    const newEdges = expansion.edges.map(edge => ({
        data: {
            source: edge.source,
            target: edge.target,
            label: edge.label,
            weight: edge.weight,
            color: edge.color
        }
    }));
    
    cy.add([...newNodes, ...newEdges]);
    
    // Re-run layout for new nodes
    cy.layout({
        name: 'cose',
        animate: true,
        fit: false
    }).run();
}

// Highlight connected nodes
function highlightConnections(node) {
    cy.elements().addClass('faded');
    
    const connectedEdges = node.connectedEdges();
    const connectedNodes = connectedEdges.connectedNodes();
    
    node.removeClass('faded').addClass('highlighted');
    connectedNodes.removeClass('faded').addClass('highlighted');
    connectedEdges.removeClass('faded').addClass('highlighted');
}

// Show research insights
function showInsights(insights) {
    if (!insights) return;
    
    const themesHtml = insights.emerging_themes ? 
        `<h4>Emerging Themes:</h4><ul>${insights.emerging_themes.map(t => `<li>${t}</li>`).join('')}</ul>` : '';
    document.getElementById('emerging-themes').innerHTML = themesHtml;
    
    const gapsHtml = insights.research_gaps ? 
        `<h4>Research Gaps:</h4><ul>${insights.research_gaps.map(g => `<li>${g}</li>`).join('')}</ul>` : '';
    document.getElementById('research-gaps').innerHTML = gapsHtml;
    
    const directionsHtml = insights.future_directions ? 
        `<h4>Future Directions:</h4><ul>${insights.future_directions.map(d => `<li>${d}</li>`).join('')}</ul>` : '';
    document.getElementById('future-directions').innerHTML = directionsHtml;
}

// Utility functions
function getThemeColor(theme) {
    const colors = {
        'Machine Learning': '#4CAF50',
        'Deep Learning': '#2196F3',
        'Optimization': '#FF9800',
        'NLP': '#9C27B0',
        'Computer Vision': '#F44336',
        'General': '#757575'
    };
    return colors[theme] || '#757575';
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}

function showError(message) {
    alert(message); // Replace with better error UI
}

function hideSidebar() {
    document.getElementById('sidebar').classList.add('hidden');
    selectedPaperId = null;
}

// Initialize graph on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeGraph();
    
    // Allow Enter key for search
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchPapers();
        }
    });
});
