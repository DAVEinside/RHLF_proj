"""
Web interface for the Human-Aligned AI Content Generator.
"""
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import random
import json
import logging
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import config
from model.content_generator import ContentGenerator
from evaluation.evaluator import Evaluator
from rlhf.feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                static_folder="static",
                template_folder="templates")
    CORS(app)
    
    generator = ContentGenerator()
    evaluator = Evaluator()
    feedback_collector = FeedbackCollector()
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html', 
                              values=config.VALUES_AND_GUIDELINES,
                              metrics=config.EVALUATION_METRICS)
    
    @app.route('/api/generate', methods=['POST'])
    def generate_content():
        """Generate content based on prompt and parameters."""
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        temperature = data.get('temperature', config.MODEL_TEMPERATURE)
        max_tokens = data.get('max_tokens', config.MAX_TOKENS)
        
        try:
            logger.info(f"Generating content for prompt: {prompt[:50]}...")
            content = generator.generate(
                prompt, 
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Evaluate the content
            evaluation_results = evaluator.evaluate_content(content)
            
            return jsonify({
                'content': content,
                'evaluation': evaluation_results.to_dict()
            })
        
        except Exception as e:
            logger.exception(f"Error generating content: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        """Submit feedback for generated content."""
        data = request.json
        
        if not data or 'content_id' not in data or 'rating' not in data:
            return jsonify({'error': 'Content ID and rating are required'}), 400
        
        content_id = data['content_id']
        rating = data['rating']
        feedback_text = data.get('feedback_text', '')
        
        try:
            feedback_collector.collect_feedback(
                content_id=content_id,
                rating=rating,
                feedback_text=feedback_text
            )
            
            return jsonify({'success': True, 'message': 'Feedback recorded'})
        
        except Exception as e:
            logger.exception(f"Error recording feedback: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/values', methods=['GET'])
    def get_values():
        """Return the configured values and guidelines."""
        return jsonify(config.VALUES_AND_GUIDELINES)
    
    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        """Return the evaluation metrics."""
        return jsonify(config.EVALUATION_METRICS)
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        return jsonify({'error': 'Server error'}), 500
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        return jsonify({'error': 'Server error'}), 500
    
    @app.route('/dashboard')
    def dashboard():
        """Render the dashboard page."""
        # In a real implementation, this data would come from the database
        # For demonstration, we'll generate sample data
        
        # Generate dates for the last 14 days
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14)]
        dates.reverse()  # Oldest first
        
        # Sample data for content generations
        generation_counts = [random.randint(5, 30) for _ in range(14)]
        
        # Sample data for evaluation scores
        evaluation_scores = [round(random.uniform(6.5, 9.5), 2) for _ in range(14)]
        
        # Sample metrics averages
        metrics_averages = [
            round(random.uniform(7.5, 9.0), 2),  # Factual Accuracy
            round(random.uniform(7.0, 9.5), 2),  # Ethical Alignment
            round(random.uniform(7.2, 9.2), 2),  # Value Alignment
            round(random.uniform(8.0, 9.8), 2),  # Toxicity
            round(random.uniform(6.8, 8.5), 2),  # Bias
            round(random.uniform(7.5, 9.0), 2),  # Helpfulness
            round(random.uniform(7.8, 9.5), 2)   # Coherence
        ]
        
        # Sample feedback data
        content_types = ["blog post", "social media post", "email", "article", "product description"]
        feedback_data = []
        
        for i in range(10):
            feedback_data.append({
                "id": f"f-{100 + i}",
                "content_type": random.choice(content_types),
                "rating": random.randint(3, 5),
                "feedback_text": "Good content but could be more detailed about the subject.",
                "timestamp": (today - timedelta(days=random.randint(0, 7))).strftime('%Y-%m-%d %H:%M')
            })
        
        # Sample RLHF status data
        rlhf_data = {
            "reward_model": {
                "status": "Trained",
                "last_trained": (today - timedelta(days=3)).strftime('%Y-%m-%d'),
                "training_pairs": 342,
                "accuracy": 87.4
            },
            "fine_tuning": {
                "status": "Ready",
                "last_tuned": (today - timedelta(days=5)).strftime('%Y-%m-%d'),
                "training_samples": 253,
                "current_model": "human-aligned-model-v2"
            },
            "suggestions": [
                {
                    "content_type": "blog post",
                    "suggestion": "Include more concrete examples for climate change",
                    "impact_score": 0.89
                },
                {
                    "content_type": "social media",
                    "suggestion": "Improve factual accuracy about technology",
                    "impact_score": 0.76
                },
                {
                    "content_type": "article",
                    "suggestion": "Add more nuance to discussions of social issues",
                    "impact_score": 0.72
                },
                {
                    "content_type": "email",
                    "suggestion": "Use more direct language for calls to action",
                    "impact_score": 0.68
                },
                {
                    "content_type": "product description",
                    "suggestion": "Balance features with benefits more effectively",
                    "impact_score": 0.63
                }
            ]
        }
        
        # Sample metrics data
        metrics_data = {
            "content_generations": sum(generation_counts),
            "feedback_count": 153,
            "avg_rating": 4.3,
            "model_status": "healthy"
        }
        
        return render_template(
            'dashboard.html',
            metrics=metrics_data,
            feedback=feedback_data,
            rlhf=rlhf_data,
            generation_dates=dates,
            generation_counts=generation_counts,
            evaluation_dates=dates,
            evaluation_scores=evaluation_scores,
            metrics_averages=metrics_averages
        )
    
    return app
    

if __name__ == '__main__':
    # This is just for development. In production, use main.py.
    app = create_app()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)