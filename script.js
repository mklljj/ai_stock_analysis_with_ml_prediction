// AI Stock Analyzer - Complete Client-Side Script

// Global variables to hold the chart instance and its full dataset
let priceChartInstance = null;
let fullChartData = null;

// Wait for the page to load before attaching event handlers
window.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById('analyzeBtn');
    if (btn) {
        btn.onclick = analyzeStock;
    }
    setupChartControls();
});

/**
 * Main function to orchestrate the stock analysis.
 */
async function analyzeStock() {
    const stockCode = document.getElementById('stockCode').value.trim().toUpperCase();
    const marketType = document.getElementById('marketType').value;
    
    if (!stockCode) {
        alert('Please enter a stock code!');
        return;
    }

    // Reset UI to loading state
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorDisplay').classList.add('hidden');
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    
    try {
        // 1. Fetch main analysis data (stock info, AI summary, etc.)
        const response = await fetch('http://localhost:8080/analyze_with_sentiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer my-secret-stock-api-key-2024'
            },
            body: JSON.stringify({
                stock_code: stockCode,
                market_type: marketType,
                include_sentiment: document.getElementById('includeSentiment').checked,
                include_predictions: document.getElementById('includePredictions').checked
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Server error');
        }
        
        const data = await response.json();
        
        // 2. Display the fetched text-based results
        showResults(data);
        
        // 3. Fetch and render the price chart data
        await fetchAndRenderChart(stockCode, marketType);
        
    } catch (error) {
        document.getElementById('errorDisplay').classList.remove('hidden');
        document.getElementById('errorMessage').textContent = error.message;
    } finally {
        // Restore UI after analysis is complete or fails
        document.getElementById('loader').classList.add('hidden');
        btn.disabled = false;
        btn.textContent = 'Analyze Stock';
    }
}

/**
 * Populates all text-based sections of the results area.
 * @param {object} data - The complete analysis data from the server.
 */
function showResults(data) {
    const stock = data.stock_data;
    
    // Stock Info Grid
    document.getElementById('ticker').textContent = stock.ticker || '-';
    document.getElementById('currentPrice').textContent = stock.current_price ? '$' + stock.current_price.toFixed(2) : '-';
    document.getElementById('rsi').textContent = stock.technical_indicators.RSI ? stock.technical_indicators.RSI.toFixed(2) : 'N/A';
    document.getElementById('volume').textContent = stock.volume_analysis.volume_ratio ? stock.volume_analysis.volume_ratio.toFixed(2) + 'x' : '-';
    document.getElementById('trend').textContent = stock.trend || '-';
    
    const change = stock.price_change;
    const pct = stock.price_change_percent;
    const priceChange = document.getElementById('priceChange');
    priceChange.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + ` (${pct.toFixed(2)}%)`;
    priceChange.className = 'value ' + (change >= 0 ? 'positive' : 'negative');

    // ML Predictions Section
    if (data.prediction_data && data.prediction_data.success) {
        showPredictions(data.prediction_data);
    } else {
        document.getElementById('predictionsSection').classList.add('hidden');
    }

    // Sentiment Analysis Section
    if (data.sentiment_data) {
        showSentiment(data.sentiment_data);
    } else {
        document.getElementById('sentimentSection').classList.add('hidden');
    }
    
    // AI Analysis Text
    document.getElementById('aiAnalysis').textContent = data.ai_analysis || 'No analysis available';

    // Model Info Footer
    if (data.model_info) {
        document.getElementById('modelProvider').textContent = data.model_info.provider || 'AI';
        document.getElementById('mlModels').textContent = `ML Models: ${data.model_info.ml_models.join(', ')}`;
    }
    
    // Make the results visible and scroll to them
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Populates the ML predictions section.
 * @param {object} pred - The prediction data object.
 */
function showPredictions(pred) {
    const section = document.getElementById('predictionsSection');
    
    document.getElementById('ensemblePrediction').textContent = pred.ensemble_prediction || '-';
    document.getElementById('ensembleDirection').textContent = pred.prediction_direction || '-';
    document.getElementById('predictionPeriod').textContent = pred.prediction_period || '-';
    
    const grid = document.getElementById('modelsGrid');
    grid.innerHTML = ''; // Clear previous results
    if (pred.predictions) {
        for (const [name, value] of Object.entries(pred.predictions)) {
            const card = document.createElement('div');
            card.className = 'model-card';
            card.innerHTML = `
                <div class="model-name">${name}</div>
                <div class="model-prediction">${value}</div>
            `;
            grid.appendChild(card);
        }
    }
    
    document.getElementById('trainingSamples').textContent = pred.training_samples || '-';
    document.getElementById('testSamples').textContent = pred.test_samples || '-';
    document.getElementById('featureCount').textContent = pred.feature_count || '-';
    
    section.classList.remove('hidden');
}

/**
 * Populates the sentiment analysis section.
 * @param {object} sentiment - The sentiment data object.
 */
function showSentiment(sentiment) {
    const section = document.getElementById('sentimentSection');
    if (!sentiment || typeof sentiment.combined_score === 'undefined') {
        section.classList.add('hidden');
        return;
    }

    const score = sentiment.combined_score;
    const label = sentiment.combined_label;

    document.getElementById('sentimentLabel').textContent = label;
    document.getElementById('sentimentScore').textContent = `${score} / 100`;
    
    const sentimentBar = document.getElementById('sentimentBar');
    const barWidth = (score + 100) / 2; // Normalize score from [-100, 100] to [0, 100]
    sentimentBar.style.width = `${barWidth}%`;
    
    sentimentBar.className = 'sentiment-bar-fill'; // Reset classes
    if (label === 'Positive') sentimentBar.classList.add('positive');
    else if (label === 'Negative') sentimentBar.classList.add('negative');
    else sentimentBar.classList.add('neutral');

    let sourcesText = `Overall Confidence: ${sentiment.confidence}\n\n--- Sources ---\n`;
    sentiment.sources.forEach(source => {
        sourcesText += `• ${source.source} (${source.weight}): ${source.sentiment} (${source.score}/100) based on ${source.count} items.\n`;
    });
    document.getElementById('sentimentAnalysis').textContent = sourcesText;
    
    section.classList.remove('hidden');
}

/**
 * Fetches data for the chart and triggers rendering.
 * @param {string} stockCode - The stock ticker.
 * @param {string} marketType - The market type (e.g., 'US').
 */
async function fetchAndRenderChart(stockCode, marketType) {
    try {
        const response = await fetch('http://localhost:8080/chart_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer my-secret-stock-api-key-2024'
            },
            body: JSON.stringify({ stock_code: stockCode, market_type: marketType })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Could not fetch chart data');
        }
        
        const chartData = await response.json();
        fullChartData = chartData; // Store complete dataset for filtering
        renderPriceChart(chartData);
        document.querySelector('.chart-btn[data-period="all"]').click(); // Activate 'All' button
        
    } catch (error) {
        console.error('Chart Error:', error);
        const chartContainer = document.querySelector('.chart-container');
        chartContainer.innerHTML = `<p style="text-align:center; padding: 2rem;">Could not load chart data.</p>`;
    }
}

/**
 * Renders the price chart using Chart.js.
 * @param {object} chartData - The data for plotting the chart.
 */
function renderPriceChart(chartData) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChartInstance) {
        priceChartInstance.destroy(); // Destroy old chart before creating a new one
    }
    
    priceChartInstance = new Chart(ctx, {
        type: 'bar', // Base type, datasets override this
        data: {
            labels: chartData.timestamps,
            datasets: [
                {
                    type: 'line',
                    label: 'Price',
                    data: chartData.prices,
                    borderColor: '#667eea',
                    backgroundColor: '#667eea20',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: true,
                    yAxisID: 'y',
                },
                {
                    type: 'line',
                    label: 'SMA 5',
                    data: chartData.sma5,
                    borderColor: '#f0ad4e',
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    yAxisID: 'y',
                },
                {
                    type: 'line',
                    label: 'SMA 20',
                    data: chartData.sma20,
                    borderColor: '#5bc0de',
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    yAxisID: 'y',
                },
                {
                    type: 'bar',
                    label: 'Volume',
                    data: chartData.volumes,
                    backgroundColor: '#cccccc80',
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 10 } },
                y: { type: 'linear', display: true, position: 'left' }, // Price Axis
                y1: { type: 'linear', display: false, position: 'right', grid: { drawOnChartArea: false } } // Volume Axis
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#fff',
                    titleColor: '#333',
                    bodyColor: '#666',
                    borderColor: '#ddd',
                    borderWidth: 1,
                }
            }
        }
    });
}

/**
 * Attaches click event listeners to the chart filter buttons.
 */
function setupChartControls() {
    const buttons = document.querySelectorAll('.chart-btn');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            buttons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            const period = button.getAttribute('data-period');
            updateChartPeriod(period);
        });
    });
}

/**
 * Filters the chart data based on the selected time period and redraws the chart.
 * @param {string} period - The time period to display ('1h', '3h', 'today', 'all').
 */
function updateChartPeriod(period) {
    if (!priceChartInstance || !fullChartData) return;

    const now = new Date();
    let startIndex = 0;
    
    if (period !== 'all') {
        let hoursToGoBack = 0;
        if (period === '1h') hoursToGoBack = 1;
        if (period === '3h') hoursToGoBack = 3;
        
        let targetTime;
        if (period === 'today') {
            targetTime = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        } else {
            targetTime = new Date(now.getTime() - hoursToGoBack * 60 * 60 * 1000);
        }
        
        startIndex = fullChartData.timestamps.findIndex(ts => new Date(ts) >= targetTime);
        if (startIndex === -1) startIndex = 0;
    }

    // Update chart data with sliced arrays
    priceChartInstance.data.labels = fullChartData.timestamps.slice(startIndex);
    priceChartInstance.data.datasets.forEach((dataset, index) => {
        const fullData = [
            fullChartData.prices, 
            fullChartData.sma5, 
            fullChartData.sma20, 
            fullChartData.volumes
        ];
        dataset.data = fullData[index].slice(startIndex);
    });
    
    priceChartInstance.update(); // Redraw the chart
}