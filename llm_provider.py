"""
This module abstracts the interaction with Large Language Model providers.
It allows for easily swapping between different LLMs (e.g., OpenAI, Gemini)
without changing the core application logic.
"""
import google.generativeai as genai
import ollama
import requests
from typing import Dict, Any
import time
from functools import wraps
from google.api_core import exceptions as google_exceptions

def gemini_retry_decorator(func):
    """
    A decorator to handle Gemini API rate limiting with exponential backoff.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 3
        delay = 5  # Initial delay in seconds
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            # Specifically catch ResourceExhausted, which is the error for rate limits
            except google_exceptions.ResourceExhausted as e:
                print(f"Gemini API rate limit hit. Retrying in {delay} seconds... (Attempt {i + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            except Exception as e:
                # Handle other potential exceptions during the API call
                print(f"An unexpected error occurred during the Gemini call: {e}")
                # Re-raise the exception to make the error visible
                raise
        
        # If all retries fail, return a clear error message
        print("All retries failed. Could not get a response from Gemini.")
        if "json" in func.__name__:
            return {
                "answer_found": False,
                "summary": "An error occurred after multiple retries with the language model.",
                "extracted_clauses": []
            }
        return "Error: Could not get a response from the language model after multiple retries."
    return wrapper

from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OllamaProvider(BaseLLMProvider):
    """
    A provider class to interact with a local Ollama server.
    """
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        """
        Initializes the Ollama provider.

        Args:
            model: The model to use (e.g., "llama3").
            host: The URL of the Ollama server.
        """
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Generates a natural language text response.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}")
            return f"Error: Could not get a response from Ollama. Is it running? Details: {e}"

class GeminiProvider(BaseLLMProvider):
    """
    A provider class to interact with the Google Gemini API.
    """
    def __init__(self, api_key: str):
        """
        Initializes the Gemini provider and configures the API key.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @gemini_retry_decorator
    def generate(self, prompt: str) -> str:
        """
        Generates a natural language text response.
        """
        response = self.model.generate_content(prompt)
        return response.text

import openai

def openai_retry_decorator(func):
    """
    A decorator to handle OpenAI API rate limiting with exponential backoff.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 3
        delay = 5  # Initial delay in seconds
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                print(f"OpenAI API rate limit hit. Retrying in {delay} seconds... (Attempt {i + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            except Exception as e:
                print(f"An unexpected error occurred during the OpenAI call: {e}")
                raise
        
        print("All retries failed. Could not get a response from OpenAI.")
        return "Error: Could not get a response from the language model after multiple retries."
    return wrapper

class OpenAIProvider(BaseLLMProvider):
    """
    A provider class to interact with the OpenAI API.
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAI provider and sets the API key.
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    @openai_retry_decorator
    def generate(self, prompt: str) -> str:
        """
        Generates a natural language text response using OpenAI's chat completions.
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            return f"Error: Could not get a response from OpenAI. Details: {e}"