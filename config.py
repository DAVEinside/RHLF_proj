"""
Configuration settings for the Human-Aligned AI Content Generator.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # Options: "anthropic", "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# RLHF settings
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "data/feedback.db")
REWARD_MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "models/reward_model")
FINE_TUNED_MODEL_PATH = os.getenv("FINE_TUNED_MODEL_PATH", "models/fine_tuned")
FEEDBACK_THRESHOLD = float(os.getenv("FEEDBACK_THRESHOLD", "0.8"))

# Evaluation settings
EVALUATION_METRICS = [
    "factual_accuracy",
    "ethical_alignment",
    "value_alignment",
    "toxicity",
    "bias",
    "helpfulness",
    "coherence"
]

# Human values and ethical guidelines
VALUES_AND_GUIDELINES = {
    "human_welfare": "Prioritize human well-being and flourishing",
    "fairness": "Treat all individuals and groups with fairness and equity",
    "autonomy": "Respect human autonomy and decision-making capacity",
    "privacy": "Respect and protect personal privacy and data",
    "harm_prevention": "Avoid causing direct or indirect harm",
    "trustworthiness": "Be truthful, honest, and reliable",
    "transparency": "Be clear and understandable about limitations and capabilities",
    "diversity": "Respect diverse perspectives and cultural contexts"
}

# Web application settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/application.log")