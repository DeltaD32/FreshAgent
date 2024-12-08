from typing import Dict, Optional, List, Any, Tuple
import json
import aiohttp
from datetime import datetime
import time
from pydantic import BaseModel
import openai
from openai import AsyncOpenAI
import re
import logging
import subprocess
import asyncio
import platform
import os
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class LLMInterface:
    def __init__(self, model_name: str = "mistral", provider: str = "ollama", api_key: Optional[str] = None):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.base_url = "http://localhost:11434/api" if provider == "ollama" else "https://api.openai.com/v1"
        self._verify_model_name_sync()
        logger.info(f"LLM Interface initialized with {provider} provider and {model_name} model")

    def _verify_model_name_sync(self):
        """Verify that the model name is valid and available (synchronous version)."""
        if self.provider == "ollama":
            try:
                # First verify Ollama service is running
                if not self._check_ollama_service_sync():
                    raise ValueError("Ollama service is not running. Please start Ollama first.")

                # Get available models
                available_models = self._get_ollama_models_sync()
                
                # If model isn't available, try to pull it
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found locally. Attempting to pull...")
                    # Pull model using API
                    success = self._pull_model_sync()
                    if not success:
                        raise ValueError(f"Failed to pull model {self.model_name}")
                    logger.info(f"Successfully pulled model {self.model_name}")
                
                logger.info(f"Using Ollama model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error verifying Ollama model: {str(e)}")
                raise ValueError(f"Failed to verify Ollama model: {str(e)}")
        elif self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required")

    def _check_ollama_service_sync(self) -> bool:
        """Check if Ollama service is running (synchronous version)."""
        try:
            response = requests.get(f"{self.base_url}/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama service: {str(e)}")
            return False

    def _get_ollama_models_sync(self) -> List[str]:
        """Get list of available Ollama models (synchronous version)."""
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {str(e)}")
            return []

    def _pull_model_sync(self) -> bool:
        """Pull an Ollama model using the API (synchronous version)."""
        try:
            response = requests.post(
                f"{self.base_url}/pull",
                json={"name": self.model_name},
                stream=True
            )
            
            if response.status_code != 200:
                return False
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            logger.error(f"Error pulling model: {data['error']}")
                            return False
                        if "status" in data:
                            logger.debug(f"Pull status: {data['status']}")
                    except json.JSONDecodeError:
                        continue
            
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False

    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM (synchronous version)."""
        try:
            if self.provider == "ollama":
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                if system_prompt:
                    request_data["system"] = system_prompt

                response = requests.post(
                    f"{self.base_url}/generate",
                    json=request_data
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Ollama API error: {error_text}")
                    raise ValueError(f"Ollama API error: {error_text}")
                
                result = response.json()
                return result.get("response", "")

            elif self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages
                    }
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenAI API error: {error_text}")
                    raise ValueError(f"OpenAI API error: {error_text}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")

    def verify_connection_sync(self) -> bool:
        """Verify that the connection to the LLM provider is working (synchronous version)."""
        try:
            if self.provider == "ollama":
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": "test",
                        "stream": False
                    }
                )
                return response.status_code == 200
            elif self.provider == "openai":
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error verifying connection: {str(e)}")
            return False