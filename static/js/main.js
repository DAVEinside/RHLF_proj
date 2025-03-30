/**
 * Main JavaScript for Human-Aligned AI Content Generator
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeTemperatureSlider();
    initializeGenerateForm();
    initializeFeedbackForm();
});

/**
 * Initialize temperature slider
 */
function initializeTemperatureSlider() {
    const temperatureSlider = document.getElementById('temperatureSlider');
    const temperatureValue = document.getElementById('temperatureValue');
    
    if (temperatureSlider && temperatureValue) {
        temperatureSlider.addEventListener('input', function() {
            temperatureValue.textContent = this.value;
        });
    }
}

/**
 * Initialize content generation form
 */
function initializeGenerateForm() {
    const generateForm = document.getElementById('generateForm');
    
    if (generateForm) {
        generateForm.addEventListener('submit', function(event) {
            event.preventDefault();
            generateContent();
        });
    }
}

/**
 * Initialize feedback form
 */
function initializeFeedbackForm() {
    const feedbackForm = document.getElementById('feedbackForm');
    
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(event) {
            event.preventDefault();
            submitFeedback();
        });
    }
}

/**
 * Generate content from API
 */
function generateContent() {
    // Get form values
    const prompt = document.getElementById('promptInput').value;
    const contentType = document.getElementById('contentTypeSelect').value;
    const temperature = parseFloat(document.getElementById('temperatureSlider').value);
    
    // Validate prompt
    if (!prompt.trim()) {
        showAlert('Please enter a prompt.', 'danger');
        return;
    }
    
    // Show loading indicator and hide results
    toggleLoadingState(true);
    
    // Send request to API
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            content_type: contentType,
            temperature: temperature
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        toggleLoadingState(false);
        
        if (data.error) {
            showAlert('Error: ' + data.error, 'danger');
            return;
        }
        
        // Display content
        displayContent(data.content);
        
        // Store content ID for feedback
        document.getElementById('contentId').value = data.content_id || 'temp-id';
        
        // Display evaluation results
        if (data.evaluation) {
            displayEvaluation(data.evaluation);
        }
        
        // Show feedback section
        document.getElementById('feedbackSection').style.display = 'block';
    })
    .catch(error => {
        toggleLoadingState(false);
        showAlert('Error: ' + error.message, 'danger');
    });
}

/**
 * Submit feedback to API
 */
function submitFeedback() {
    // Get form values
    const contentId = document.getElementById('contentId').value;
    const ratingInput = document.querySelector('input[name="rating"]:checked');
    const rating = ratingInput ? parseInt(ratingInput.value) : 5;
    const feedbackText = document.getElementById('feedbackText').value;
    
    // Show loading for feedback
    const submitButton = document.querySelector('#feedbackForm button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';
    
    // Send feedback to API
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            content_id: contentId,
            rating: rating,
            feedback_text: feedbackText
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = originalText;
        
        if (data.error) {
            showAlert('Error: ' + data.error, 'danger');
            return;
        }
        
        // Show success message
        showAlert('Thank you for your feedback!', 'success');
        document.getElementById('feedbackForm').reset();
    })
    .catch(error => {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = originalText;
        
        showAlert('Error: ' + error.message, 'danger');
    });
}

/**
 * Toggle loading state of the application
 */
function toggleLoadingState(isLoading) {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const contentDisplay = document.getElementById('contentDisplay');
    const evaluationResult = document.getElementById('evaluationResult');
    const feedbackSection = document.getElementById('feedbackSection');
    
    if (isLoading) {
        // Show loading, hide results
        loadingIndicator.style.display = 'block';
        contentDisplay.style.display = 'none';
        evaluationResult.style.display = 'none';
        feedbackSection.style.display = 'none';
    } else {
        // Hide loading
        loadingIndicator.style.display = 'none';
    }
}

/**
 * Display generated content
 */
function displayContent(content) {
    const contentDisplay = document.getElementById('contentDisplay');
    contentDisplay.textContent = content;
    contentDisplay.style.display = 'block';
    
    // Scroll to content
    contentDisplay.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Display evaluation results
 */
function displayEvaluation(evaluation) {
    const evaluationResult = document.getElementById('evaluationResult');
    const overallScore = document.getElementById('overallScore');
    const metricScores = document.getElementById('metricScores');
    
    // Display overall score
    const overallValue = evaluation.overall_score.toFixed(2);
    let scoreClass = getScoreClass(evaluation.overall_score);
    overallScore.innerHTML = `<h4>Overall Score: <span class="${scoreClass}">${overallValue}/10</span></h4>`;
    
    // Clear previous metrics
    metricScores.innerHTML = '';
    
    // Display metric scores
    for (const [metric, data] of Object.entries(evaluation.metrics)) {
        const score = data.score.toFixed(2);
        const explanation = data.explanation || '';
        
        // Format metric name
        const metricName = formatMetricName(metric);
        
        // Determine score class
        scoreClass = getScoreClass(data.score);
        
        // Create metric score element
        const metricElement = document.createElement('div');
        metricElement.classList.add('mb-3');
        metricElement.innerHTML = `
            <div><strong>${metricName}:</strong> <span class="${scoreClass}">${score}/10</span></div>
            <div class="small text-muted">${explanation}</div>
        `;
        
        metricScores.appendChild(metricElement);
    }
    
    evaluationResult.style.display = 'block';
}

/**
 * Get CSS class based on score
 */
function getScoreClass(score) {
    if (score >= 7) {
        return 'score-high';
    } else if (score >= 5) {
        return 'score-medium';
    } else {
        return 'score-low';
    }
}

/**
 * Format metric name for display
 */
function formatMetricName(metric) {
    return metric
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    // Check if alert container exists, otherwise create it
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.maxWidth = '350px';
        alertContainer.style.zIndex = '1050';
        document.body.appendChild(alertContainer);
    }
    
    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.role = 'alert';
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to container
    alertContainer.appendChild(alertElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => alertElement.remove(), 150);
    }, 5000);
}