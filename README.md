# Human-Aligned AI Content Generator

A Python-based system that generates content following specific human values and ethical guidelines using reinforcement learning from human feedback (RLHF).

## Overview

This project implements an LLM-based content generation system that:
- Generates content aligned with specific human values and ethical guidelines
- Utilizes RLHF techniques to continuously improve alignment with human preferences
- Includes a comprehensive evaluation framework to measure safety and utility

## Project Structure

```
human_aligned_ai/
├── main.py                  # Main entry point
├── app.py                   # Web interface/API
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
├── .env.example             # Template for environment variables
├── model/
│   ├── __init__.py
│   ├── llm_integration.py   # Integration with LLM APIs
│   ├── prompt_engineering.py # Prompt templates and engineering
│   └── content_generator.py # Core content generation logic
├── rlhf/
│   ├── __init__.py
│   ├── feedback_collector.py # Collects human feedback
│   ├── reward_model.py       # Trains reward model from feedback
│   └── fine_tuning.py        # Fine-tunes model using rewards
└── evaluation/
    ├── __init__.py
    ├── metrics.py           # Metrics definitions
    ├── evaluator.py         # Evaluation implementation
    └── reporting.py         # Evaluation reporting
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run the system:
   ```
   python main.py
   ```

## Features

- **Value-Aligned Content Generation**: Generate content that follows specific human values and ethical guidelines.
- **RLHF Implementation**: Continuously improve model responses based on human feedback.
- **Comprehensive Evaluation**: Measure the safety and utility of generated content across multiple dimensions.
- **Web Interface**: Interact with the system through a user-friendly web interface.

## Usage Examples

### Basic Content Generation

```python
from human_aligned_ai.model.content_generator import ContentGenerator

generator = ContentGenerator()
content = generator.generate("Write a blog post about sustainable technology")
print(content)
```

### Collecting Feedback

```python
from human_aligned_ai.rlhf.feedback_collector import FeedbackCollector

collector = FeedbackCollector()
collector.collect_feedback(content_id, rating=4, feedback_text="Good content but could be more nuanced")
```

### Running Evaluation

```python
from human_aligned_ai.evaluation.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate_content(content)
print(results.summary())
```

## License

MIT