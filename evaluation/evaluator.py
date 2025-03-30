"""
Evaluator for assessing AI-generated content.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import config
from evaluation.metrics import MetricsEvaluator, EvaluationResult

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Main evaluator that orchestrates content evaluation across multiple dimensions.
    """
    
    def __init__(self, cache_results: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            cache_results: Whether to cache evaluation results
        """
        self.metrics_evaluator = MetricsEvaluator()
        self.cache_results = cache_results
        
        # Cache for evaluation results
        self.evaluation_cache = {}
        
        # Create cache directory if needed
        if self.cache_results:
            os.makedirs("data/evaluations", exist_ok=True)
        
        logger.info("Initialized Evaluator")
    
    def evaluate_content(self, 
                        content: str, 
                        content_id: Optional[str] = None,
                        metrics: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate content across different metrics.
        
        Args:
            content: The content to evaluate
            content_id: Optional ID for the content
            metrics: Optional list of metrics to evaluate
            
        Returns:
            Evaluation results
        """
        # Generate a content hash for caching if no ID provided
        if not content_id:
            # Simple hash of content for caching purposes
            content_id = str(hash(content) % 10000000)
        
        # Check cache first
        if content_id in self.evaluation_cache:
            logger.info(f"Using cached evaluation for content {content_id}")
            return self.evaluation_cache[content_id]
        
        # Perform full evaluation
        logger.info(f"Evaluating content {content_id}")
        
        # Use default metrics from config if not specified
        metrics = metrics or config.EVALUATION_METRICS
        
        # Evaluate using metrics evaluator
        result = self.metrics_evaluator.evaluate_content(content, metrics)
        
        # Cache the result
        if self.cache_results:
            self.evaluation_cache[content_id] = result
            self._save_evaluation(content_id, content, result)
        
        return result
    
    def _save_evaluation(self, 
                        content_id: str, 
                        content: str, 
                        result: EvaluationResult):
        """
        Save evaluation results to disk.
        
        Args:
            content_id: ID of the content
            content: The evaluated content
            result: Evaluation results
        """
        try:
            # Create a record with content and results
            evaluation_record = {
                "content_id": content_id,
                "content": content,
                "result": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to JSON file
            file_path = f"data/evaluations/eval_{content_id}.json"
            with open(file_path, 'w') as f:
                json.dump(evaluation_record, f, indent=2)
                
            logger.debug(f"Saved evaluation for content {content_id}")
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
    
    def load_evaluation(self, content_id: str) -> Optional[EvaluationResult]:
        """
        Load a previously saved evaluation.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Evaluation results or None if not found
        """
        file_path = f"data/evaluations/eval_{content_id}.json"
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct EvaluationResult
            result = EvaluationResult()
            
            for metric_name, metric_data in data["result"]["metrics"].items():
                result.add_metric_result(
                    metric_name, 
                    metric_data["score"], 
                    metric_data.get("explanation")
                )
            
            result.overall_score = data["result"]["overall_score"]
            
            # Cache the result
            self.evaluation_cache[content_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading evaluation: {e}")
            return None
    
    def batch_evaluate(self, 
                      contents: List[str], 
                      metrics: Optional[List[str]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple content items.
        
        Args:
            contents: List of content items to evaluate
            metrics: Optional list of metrics to evaluate
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, content in enumerate(contents):
            logger.info(f"Batch evaluating content {i+1}/{len(contents)}")
            result = self.evaluate_content(content, metrics=metrics)
            results.append(result)
        
        return results
    
    def compare_versions(self, 
                        versions: List[str], 
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different versions of content.
        
        Args:
            versions: List of content versions to compare
            metrics: Optional list of metrics to evaluate
            
        Returns:
            Comparison results
        """
        # Evaluate each version
        results = self.batch_evaluate(versions, metrics)
        
        # Compare overall scores
        overall_scores = [result.overall_score for result in results]
        best_version_idx = overall_scores.index(max(overall_scores))
        
        # Compare by metric
        metric_comparisons = {}
        
        # Get all metrics used across versions
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            metric_scores = []
            
            for i, result in enumerate(results):
                if metric in result.metrics:
                    score = result.metrics[metric]["score"]
                else:
                    score = None
                    
                metric_scores.append((i, score))
            
            # Filter out None values
            valid_scores = [(i, s) for i, s in metric_scores if s is not None]
            
            if valid_scores:
                best_idx, best_score = max(valid_scores, key=lambda x: x[1])
                metric_comparisons[metric] = {
                    "best_version": best_idx,
                    "best_score": best_score,
                    "all_scores": [s for _, s in metric_scores]
                }
        
        return {
            "overall_scores": overall_scores,
            "best_version": best_version_idx,
            "metric_comparisons": metric_comparisons,
            "results": [r.to_dict() for r in results]
        }
    
    def get_improvement_suggestions(self, 
                                  result: EvaluationResult, 
                                  content: str) -> Dict[str, List[str]]:
        """
        Generate improvement suggestions based on evaluation results.
        
        Args:
            result: Evaluation results
            content: The evaluated content
            
        Returns:
            Dictionary of metrics and improvement suggestions
        """
        suggestions = {}
        
        # Focus on metrics with low scores
        low_scoring_metrics = [
            (metric, data["score"], data.get("explanation", ""))
            for metric, data in result.metrics.items()
            if data["score"] < 7.0  # Threshold for suggesting improvements
        ]
        
        # Sort by score ascending (lowest scores first)
        low_scoring_metrics.sort(key=lambda x: x[1])
        
        for metric, score, explanation in low_scoring_metrics:
            # Generate suggestions based on the metric and explanation
            metric_suggestions = self._generate_suggestions(
                metric, score, explanation, content
            )
            
            if metric_suggestions:
                suggestions[metric] = metric_suggestions
        
        return suggestions
    
    def _generate_suggestions(self, 
                            metric: str, 
                            score: float, 
                            explanation: str, 
                            content: str) -> List[str]:
        """
        Generate improvement suggestions for a specific metric.
        
        Args:
            metric: The metric name
            score: The metric score
            explanation: The evaluation explanation
            content: The content
            
        Returns:
            List of improvement suggestions
        """
        # In a production system, this would use a more sophisticated approach
        # For this example, we'll use pre-defined suggestions based on metrics
        
        # Common suggestions by metric
        suggestion_templates = {
            "factual_accuracy": [
                "Double-check factual claims for accuracy",
                "Cite sources for controversial or specific claims",
                "Avoid making overly broad generalizations",
                "Use more precise and specific language"
            ],
            
            "ethical_alignment": [
                "Consider different perspectives on ethical issues",
                "Ensure content respects autonomy and human dignity",
                "Avoid content that could cause harm or distress",
                "Make sure content is fair and equitable"
            ],
            
            "value_alignment": [
                "Align content more closely with human welfare",
                "Ensure content respects privacy and confidentiality",
                "Improve transparency about limitations and capabilities",
                "Consider diverse cultural perspectives"
            ],
            
            "toxicity": [
                "Remove potentially offensive or harmful language",
                "Use more inclusive and respectful terminology",
                "Reframe potentially divisive points in a constructive way",
                "Consider how content might affect vulnerable groups"
            ],
            
            "bias": [
                "Present multiple perspectives on the topic",
                "Check for and remove stereotypical representations",
                "Ensure balanced treatment of different groups",
                "Use inclusive language throughout"
            ],
            
            "helpfulness": [
                "Provide more actionable information",
                "Add specific examples to illustrate key points",
                "Structure content to be more accessible and scannable",
                "Consider what practical questions the reader might have"
            ],
            
            "coherence": [
                "Improve logical flow between paragraphs",
                "Strengthen the introduction and conclusion",
                "Use clearer transitions between ideas",
                "Ensure consistent terminology throughout"
            ]
        }
        
        # Return appropriate suggestions for the metric
        if metric in suggestion_templates:
            return suggestion_templates[metric]
        else:
            return ["Review and improve the content based on the evaluation"]