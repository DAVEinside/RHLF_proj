"""
Reward model for RLHF that learns from human feedback.
"""

import logging
import os
import json
import numpy as np
import pickle
from typing import Dict, List, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from rlhf.feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)

class RewardModel:
    """
    A reward model that learns to predict content quality based on human feedback.
    This is a simplified implementation for demonstration purposes.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the reward model.
        
        Args:
            model_path: Path to save/load the model
        """
        self.model_path = model_path or config.REWARD_MODEL_PATH
        self.model = None
        self.feedback_collector = FeedbackCollector()
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(f"{self.model_path}.pkl"):
            self._load_model()
        else:
            self._initialize_model()
            
        logger.info(f"Initialized RewardModel (model={'loaded' if self.model else 'new'})")
    
    def _initialize_model(self):
        """Initialize a new model."""
        # Simple pipeline with TF-IDF and logistic regression
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=10000)),
            ('classifier', LogisticRegression(C=1, max_iter=1000))
        ])
    
    def _load_model(self):
        """Load the model from disk."""
        try:
            with open(f"{self.model_path}.pkl", 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded reward model from {self.model_path}.pkl")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._initialize_model()
    
    def _save_model(self):
        """Save the model to disk."""
        try:
            with open(f"{self.model_path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved reward model to {self.model_path}.pkl")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def train(self, 
             min_rating_diff: float = 1.0, 
             min_pairs: int = 50, 
             max_pairs: int = 1000) -> Dict[str, float]:
        """
        Train the reward model using feedback pairs.
        
        Args:
            min_rating_diff: Minimum rating difference to consider
            min_pairs: Minimum number of pairs needed to train
            max_pairs: Maximum number of pairs to use
            
        Returns:
            Dictionary of training metrics
        """
        # Get feedback pairs
        pairs = self.feedback_collector.get_feedback_pairs(
            min_rating_diff=min_rating_diff, 
            limit=max_pairs
        )
        
        if len(pairs) < min_pairs:
            logger.warning(f"Not enough feedback pairs for training: {len(pairs)} < {min_pairs}")
            return {
                "trained": False,
                "pairs_available": len(pairs),
                "min_pairs_required": min_pairs
            }
        
        # Prepare training data
        X = []
        y = []
        
        for pair in pairs:
            # Add preferred content as positive example
            X.append(pair['prompt'] + " " + pair['preferred']['content'])
            y.append(1)
            
            # Add dispreferred content as negative example
            X.append(pair['prompt'] + " " + pair['dispreferred']['content'])
            y.append(0)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "trained": True,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "pairs_used": len(pairs),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        logger.info(f"Trained reward model with {len(pairs)} pairs. Accuracy: {metrics['accuracy']:.4f}")
        
        # Save the model
        self._save_model()
        
        # Also save training metrics
        with open(f"{self.model_path}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def predict_reward(self, prompt: str, content: str) -> float:
        """
        Predict the reward (quality) score for the given content.
        
        Args:
            prompt: The prompt that generated the content
            content: The content to evaluate
            
        Returns:
            A reward score between 0 and 1
        """
        if not self.model:
            logger.warning("Reward model not trained yet")
            return 0.5  # Default neutral score
        
        combined_input = prompt + " " + content
        
        # Get probability of the positive class
        pred_proba = self.model.predict_proba([combined_input])[0]
        reward = pred_proba[1]  # Probability of class 1 (preferred)
        
        return float(reward)
    
    def compare_versions(self, 
                        prompt: str, 
                        content_versions: List[str]) -> List[Tuple[int, float]]:
        """
        Compare multiple versions of content and rank them.
        
        Args:
            prompt: The prompt that generated the content
            content_versions: List of content versions to compare
            
        Returns:
            List of (version_index, score) tuples, sorted by score
        """
        if not self.model:
            logger.warning("Reward model not trained yet, returning random rankings")
            scores = np.random.random(len(content_versions))
            rankings = [(i, float(score)) for i, score in enumerate(scores)]
            return sorted(rankings, key=lambda x: x[1], reverse=True)
        
        scores = []
        for version in content_versions:
            reward = self.predict_reward(prompt, version)
            scores.append(reward)
        
        # Create (index, score) pairs and sort by score
        rankings = [(i, float(score)) for i, score in enumerate(scores)]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get the training status and metrics.
        
        Returns:
            Dictionary with training status and metrics
        """
        metrics_path = f"{self.model_path}_metrics.json"
        
        if not os.path.exists(metrics_path):
            return {
                "trained": False,
                "message": "Reward model has not been trained yet."
            }
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            return {
                "trained": True,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error loading training metrics: {e}")
            return {
                "trained": False,
                "error": str(e)
            }