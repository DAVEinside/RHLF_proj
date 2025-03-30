"""
Content generator module that integrates LLMs with prompt engineering.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List

import config
from model.llm_integration import LLMIntegration
from model.prompt_engineering import PromptEngineering

logger = logging.getLogger(__name__)

class ContentGenerator:
    """
    Generates aligned content using LLMs and prompt engineering.
    """
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the content generator.
        
        Args:
            llm_provider: Optional LLM provider to use (defaults to config)
        """
        self.llm = LLMIntegration(provider=llm_provider)
        self.prompt_engineering = PromptEngineering()
        logger.info(f"Initialized ContentGenerator with {self.llm.provider}")
        
        # Content history for tracking generated content
        self.content_history = {}
    
    def generate(self, 
                prompt: str, 
                content_type: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Generate content based on the given prompt.
        
        Args:
            prompt: The user prompt
            content_type: Optional type of content to generate
            context: Optional additional context for generation
            temperature: Optional temperature setting for the LLM
            max_tokens: Optional maximum tokens for the response
            
        Returns:
            The generated content
        """
        # Detect content type if not provided
        if not content_type:
            content_type = self.prompt_engineering.detect_content_type(prompt)
        
        # Create full context
        full_context = {"content_type": content_type}
        if context:
            full_context.update(context)
        
        # Create system prompt
        system_prompt = self.prompt_engineering.create_system_prompt(full_context)
        
        # Enhance user prompt
        enhanced_prompt = self.prompt_engineering.enhance_user_prompt(prompt, content_type)
        
        # Generate the content
        logger.info(f"Generating content for prompt: {prompt[:50]}...")
        content = self.llm.generate_text(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Store in content history
        content_id = str(uuid.uuid4())
        self.content_history[content_id] = {
            "prompt": prompt,
            "content": content,
            "content_type": content_type,
            "context": full_context
        }
        
        return content
    
    def get_content_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the content generation history.
        
        Returns:
            Dictionary of content IDs and their details
        """
        return self.content_history
    
    def generate_with_constraints(self,
                                 prompt: str,
                                 constraints: List[str],
                                 **kwargs) -> str:
        """
        Generate content with specific constraints.
        
        Args:
            prompt: The user prompt
            constraints: List of constraints to apply
            **kwargs: Additional arguments for generate()
            
        Returns:
            The generated content
        """
        constraints_text = "\n".join([f"- {constraint}" for constraint in constraints])
        enhanced_prompt = f"{prompt}\n\nGenerate content that follows these constraints:\n{constraints_text}"
        
        return self.generate(enhanced_prompt, **kwargs)
    
    def regenerate_content(self, 
                          content_id: str, 
                          feedback: Optional[str] = None,
                          temperature: Optional[float] = None) -> str:
        """
        Regenerate content based on feedback.
        
        Args:
            content_id: The ID of the content to regenerate
            feedback: Optional feedback to incorporate
            temperature: Optional temperature setting
            
        Returns:
            The regenerated content
        """
        if content_id not in self.content_history:
            raise ValueError(f"Content ID not found: {content_id}")
        
        entry = self.content_history[content_id]
        original_prompt = entry["prompt"]
        original_content = entry["content"]
        content_type = entry.get("content_type")
        
        if feedback:
            regenerate_prompt = f"""I previously generated the following content:

{original_content}

Based on this feedback: "{feedback}"

Please regenerate the content to address the feedback while maintaining alignment with human values and ethical guidelines."""
        else:
            regenerate_prompt = f"""I previously generated the following content:

{original_content}

Please regenerate this content with more creativity and improved quality while maintaining alignment with human values and ethical guidelines."""
        
        # Generate improved content
        improved_content = self.generate(
            prompt=regenerate_prompt,
            content_type=content_type,
            temperature=temperature or 0.8  # Slightly higher temp for creativity
        )
        
        # Store the regenerated content
        new_content_id = str(uuid.uuid4())
        self.content_history[new_content_id] = {
            "prompt": original_prompt,
            "content": improved_content,
            "content_type": content_type,
            "feedback": feedback,
            "original_content_id": content_id
        }
        
        return improved_content