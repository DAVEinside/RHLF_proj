"""
LLM integration module for connecting to various LLM providers.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List

import anthropic
import openai
import config

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Integration with LLM providers like Anthropic and OpenAI."""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM integration.
        
        Args:
            provider: The LLM provider to use. If None, use the one from config.
        """
        self.provider = provider or config.LLM_PROVIDER
        
        if self.provider == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env")
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = config.ANTHROPIC_MODEL
        
        elif self.provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env")
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = config.OPENAI_MODEL
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        logger.info(f"Initialized LLM integration with {self.provider} using model {self.model}")
    
    def generate_text(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = None,
                      max_tokens: int = None) -> str:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to provide context
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in the response
            
        Returns:
            The generated text response
        """
        # Use configured defaults if not specified
        temperature = temperature if temperature is not None else config.MODEL_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        
        logger.debug(f"Generating text with {self.provider}, temp={temperature}, max_tokens={max_tokens}")
        
        # Apply rate limiting and retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if self.provider == "anthropic":
                    return self._generate_anthropic(
                        prompt, system_prompt, temperature, max_tokens
                    )
                elif self.provider == "openai":
                    return self._generate_openai(
                        prompt, system_prompt, temperature, max_tokens
                    )
            
            except (anthropic.RateLimitError, openai.RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for rate limit: {e}")
                    raise
            
            except Exception as e:
                logger.error(f"Error generating text with {self.provider}: {e}")
                raise
    
    def _generate_anthropic(self, 
                           prompt: str, 
                           system_prompt: Optional[str], 
                           temperature: float, 
                           max_tokens: int) -> str:
        """Generate text using Anthropic's Claude."""
        system_prompt = system_prompt or "You are a helpful assistant."
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def _generate_openai(self, 
                        prompt: str, 
                        system_prompt: Optional[str], 
                        temperature: float, 
                        max_tokens: int) -> str:
        """Generate text using OpenAI's API."""
        system_prompt = system_prompt or "You are a helpful assistant."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content