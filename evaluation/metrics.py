"""
Metrics for evaluating AI-generated content.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Callable
import json

import config
from model.llm_integration import LLMIntegration
from model.prompt_engineering import PromptEngineering

logger = logging.getLogger(__name__)

class EvaluationResult:
    """
    Container for evaluation results across multiple metrics.
    """
    
    def __init__(self):
        """Initialize an empty evaluation result."""
        self.metrics = {}
        self.overall_score = None
    
    def add_metric_result(self, 
                         metric_name: str, 
                         score: float, 
                         explanation: Optional[str] = None):
        """
        Add a metric result.
        
        Args:
            metric_name: Name of the metric
            score: Score for the metric (0-10)
            explanation: Optional explanation for the score
        """
        self.metrics[metric_name] = {
            "score": score,
            "explanation": explanation
        }
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall score as weighted average of metric scores.
        
        Args:
            weights: Optional dictionary of metric weights
            
        Returns:
            Overall score
        """
        if not self.metrics:
            return 0.0
            
        if not weights:
            # Equal weights by default
            weights = {metric: 1.0 for metric in self.metrics}
        
        # Normalize weights for metrics that exist
        valid_weights = {m: w for m, w in weights.items() if m in self.metrics}
        total_weight = sum(valid_weights.values())
        
        if total_weight == 0:
            return 0.0
            
        normalized_weights = {m: w / total_weight for m, w in valid_weights.items()}
        
        # Calculate weighted average
        weighted_sum = sum(
            self.metrics[metric]["score"] * normalized_weights.get(metric, 0)
            for metric in self.metrics
        )
        
        self.overall_score = weighted_sum
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert evaluation results to a dictionary.
        
        Returns:
            Dictionary representation of results
        """
        # Calculate overall score if not already done
        if self.overall_score is None:
            self.calculate_overall_score()
            
        return {
            "metrics": self.metrics,
            "overall_score": self.overall_score
        }
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the evaluation results.
        
        Returns:
            Summary string
        """
        # Calculate overall score if not already done
        if self.overall_score is None:
            self.calculate_overall_score()
        
        lines = ["Evaluation Results:"]
        lines.append(f"Overall Score: {self.overall_score:.2f}/10\n")
        lines.append("Individual Metrics:")
        
        # Sort metrics by score (descending)
        sorted_metrics = sorted(
            self.metrics.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )
        
        for metric_name, result in sorted_metrics:
            score = result["score"]
            
            # Format the metric name for display
            display_name = metric_name.replace('_', ' ').title()
            
            lines.append(f"- {display_name}: {score:.2f}/10")
            
            # Add explanation if available (indented)
            if result.get("explanation"):
                # Truncate long explanations
                explanation = result["explanation"]
                if len(explanation) > 100:
                    explanation = explanation[:97] + "..."
                lines.append(f"  {explanation}")
        
        return "\n".join(lines)


class MetricsEvaluator:
    """
    Evaluates content using various metrics.
    """
    
    def __init__(self):
        """Initialize the metrics evaluator."""
        self.llm = LLMIntegration()
        self.prompt_engineering = PromptEngineering()
        
        logger.info("Initialized MetricsEvaluator")
    
    def evaluate_content(self, 
                        content: str, 
                        metrics: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate content across specified metrics.
        
        Args:
            content: The content to evaluate
            metrics: Optional list of metrics to evaluate (defaults to config)
            
        Returns:
            Evaluation results
        """
        metrics = metrics or config.EVALUATION_METRICS
        logger.info(f"Evaluating content across {len(metrics)} metrics")
        
        result = EvaluationResult()
        
        for metric in metrics:
            logger.debug(f"Evaluating metric: {metric}")
            
            score, explanation = self._evaluate_metric(content, metric)
            result.add_metric_result(metric, score, explanation)
        
        # Calculate overall score
        result.calculate_overall_score()
        
        return result
    
    def _evaluate_metric(self, content: str, metric: str) -> Tuple[float, str]:
        """
        Evaluate content for a specific metric.
        
        Args:
            content: The content to evaluate
            metric: The metric to evaluate
            
        Returns:
            Tuple of (score, explanation)
        """
        # Create evaluation prompt
        eval_prompt = self.prompt_engineering.generate_evaluation_prompt(content, metric)
        
        # Get evaluation from LLM
        response = self.llm.generate_text(
            prompt=eval_prompt,
            system_prompt="You are an objective content evaluator. Provide accurate and fair assessments of content quality.",
            temperature=0.3  # Low temperature for more consistent evaluations
        )
        
        # Extract score and explanation
        score, explanation = self._parse_evaluation_response(response)
        
        return score, explanation
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, str]:
        """
        Parse the evaluation response to extract score and explanation.
        
        Args:
            response: The LLM response to the evaluation prompt
            
        Returns:
            Tuple of (score, explanation)
        """
        # Default values
        score = 5.0
        explanation = response
        
        # Try to extract numerical rating
        # Look for patterns like "Rating: 8/10" or "Score: 7.5" or "6 out of 10"
        rating_patterns = [
            r'(\d+(\.\d+)?)\s*\/\s*10',  # 8/10
            r'rating:?\s*(\d+(\.\d+)?)',  # Rating: 8 or Rating: 8.5
            r'score:?\s*(\d+(\.\d+)?)',   # Score: 8
            r'(\d+(\.\d+)?)\s*out of\s*10' # 8 out of 10
        ]
        
        for pattern in rating_patterns:
            matches = re.search(pattern, response.lower())
            if matches:
                try:
                    extracted_score = float(matches.group(1))
                    
                    # Validate score is in 0-10 range
                    if 0 <= extracted_score <= 10:
                        score = extracted_score
                        break
                except (ValueError, IndexError):
                    continue
        
        # Try to extract explanation
        # Look for sections that start with "Explanation:" or similar
        explanation_patterns = [
            r'explanation:?\s*(.*)',
            r'reasoning:?\s*(.*)',
            r'assessment:?\s*(.*)',
            r'evaluation:?\s*(.*)'
        ]
        
        for pattern in explanation_patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                extracted_explanation = matches.group(1).strip()
                if extracted_explanation:
                    explanation = extracted_explanation
                    break
        
        return score, explanation
    
    def analyze_metric_trends(self, 
                            results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Analyze trends across multiple evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Analysis of metric trends
        """
        metrics = set()
        for result in results:
            metrics.update(result.metrics.keys())
        
        trends = {}
        
        for metric in metrics:
            # Collect scores for this metric across all results
            scores = [
                result.metrics[metric]["score"]
                for result in results
                if metric in result.metrics
            ]
            
            if not scores:
                continue
                
            trends[metric] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
        
        return trends


# Register custom metrics (these would be implemented in a real system)
custom_metrics = {
    # Example custom metric implementations would go here
    # "custom_metric_name": custom_metric_function
}