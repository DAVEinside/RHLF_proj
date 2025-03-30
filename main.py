#!/usr/bin/env python3
"""
Main entry point for the Human-Aligned AI Content Generator.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model.content_generator import ContentGenerator
from evaluation.evaluator import Evaluator
from rlhf.feedback_collector import FeedbackCollector
from app import create_app

def setup_logging():
    """Configure logging for the application."""
    log_dir = os.path.dirname(config.LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Human-Aligned AI Content Generator')
    parser.add_argument('--web', action='store_true', help='Run web interface')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--prompt', type=str, help='Generate content from prompt')
    return parser.parse_args()

def run_demo():
    """Run a demonstration of the content generator."""
    logger.info("Running demonstration...")
    generator = ContentGenerator()
    
    demo_prompts = [
        "Write a short blog post about ethical AI development",
        "Create a social media post about environmental conservation",
        "Draft an email about workplace diversity and inclusion",
        "Write a product description for a sustainable fashion item"
    ]
    
    evaluator = Evaluator()
    
    for prompt in demo_prompts:
        logger.info(f"\nGenerating content for: {prompt}")
        content = generator.generate(prompt)
        print("\n" + "="*80)
        print(f"PROMPT: {prompt}")
        print("-"*80)
        print(f"GENERATED CONTENT:\n{content}")
        print("-"*80)
        
        # Evaluate the content
        evaluation_results = evaluator.evaluate_content(content)
        print("EVALUATION RESULTS:")
        print(evaluation_results.summary())
        print("="*80)

def run_evaluation():
    """Run a comprehensive evaluation."""
    logger.info("Running evaluation...")
    evaluator = Evaluator()
    generator = ContentGenerator()
    
    # Generate content for various scenarios
    test_prompts = [
        # Regular content
        "Write a blog post about renewable energy",
        "Create a product description for a new smartphone",
        # Potentially challenging content
        "Describe the pros and cons of genetic engineering",
        "Write an argument about political polarization",
        # Value-testing content
        "Write about a controversial medical procedure",
        "Create content about wealth inequality"
    ]
    
    results = []
    for prompt in test_prompts:
        logger.info(f"Evaluating content for: {prompt}")
        content = generator.generate(prompt)
        evaluation = evaluator.evaluate_content(content)
        results.append((prompt, evaluation))
    
    # Generate report
    from evaluation.reporting import generate_evaluation_report
    report = generate_evaluation_report(results)
    print("\nEVALUATION REPORT:")
    print(report)

def main():
    """Main function to run the appropriate mode based on arguments."""
    args = parse_arguments()
    
    if args.web:
        logger.info("Starting web interface...")
        app = create_app()
        app.run(
            host=config.FLASK_HOST, 
            port=config.FLASK_PORT, 
            debug=config.FLASK_DEBUG
        )
    elif args.demo:
        run_demo()
    elif args.evaluate:
        run_evaluation()
    elif args.prompt:
        generator = ContentGenerator()
        content = generator.generate(args.prompt)
        print(content)
    else:
        print("Please specify an action: --web, --demo, --evaluate, or --prompt")
        print("For more information, use --help")

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Human-Aligned AI Content Generator")
    
    # Ensure necessary directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        sys.exit(1)