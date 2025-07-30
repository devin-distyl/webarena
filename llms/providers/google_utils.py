"""Tools to generate from Google Gemini prompts.
Mirrors the structure of openai_utils.py for consistency."""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
from tqdm.asyncio import tqdm_asyncio

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (Exception,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def _configure_google_api():
    """Configure Google API with API key."""
    if not GOOGLE_AVAILABLE:
        raise ImportError(
            "Google GenerativeAI library not available. "
            "Install with: pip install google-generativeai"
        )
    
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError(
            "GOOGLE_API_KEY environment variable must be set when using Google API."
        )
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def _convert_openai_messages_to_gemini(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert OpenAI-style messages to Gemini format."""
    gemini_messages = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # Map OpenAI roles to Gemini roles
        if role == "system":
            # Gemini doesn't have a system role, prepend to first user message
            gemini_messages.append({
                "role": "user",
                "parts": [f"System: {content}\n\nUser: "]
            })
        elif role == "user":
            # If previous message was system, append to it
            if gemini_messages and "System:" in gemini_messages[-1]["parts"][0]:
                gemini_messages[-1]["parts"][0] += content
            else:
                gemini_messages.append({
                    "role": "user", 
                    "parts": [content]
                })
        elif role == "assistant":
            gemini_messages.append({
                "role": "model",
                "parts": [content]
            })
    
    return gemini_messages


async def _throttled_google_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    """Async Google chat completion with throttling."""
    async with limiter:
        for _ in range(3):
            try:
                _configure_google_api()
                
                # Create model instance
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        top_p=top_p,
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                # Convert messages format
                gemini_messages = _convert_openai_messages_to_gemini(messages)
                
                # For single message, use generate_content
                if len(gemini_messages) == 1:
                    response = await model_instance.generate_content_async(
                        gemini_messages[0]["parts"][0]
                    )
                else:
                    # For multi-turn conversation, use chat
                    chat = model_instance.start_chat(history=gemini_messages[:-1])
                    response = await chat.send_message_async(
                        gemini_messages[-1]["parts"][0]
                    )
                
                return {
                    "choices": [{
                        "message": {
                            "content": response.text if response.text else ""
                        }
                    }]
                }
                
            except Exception as e:
                logging.warning(f"Google API error: {e}")
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    logging.warning("Google API rate limit exceeded. Sleeping for 10 seconds.")
                    await asyncio.sleep(10)
                else:
                    break
                    
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_google_chat_completion(
    messages_list: list[list[dict[str, str]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 60,  # Google has lower rate limits than OpenAI
) -> list[str]:
    """Generate from Google Gemini Chat API.

    Args:
        messages_list: list of message list
        model: Model name (e.g., 'gemini-pro', 'gemini-2.5-pro')
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    _configure_google_api()

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_google_chat_completion_acreate(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_google_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Generate from Google Gemini Chat API.
    
    Args:
        messages: List of messages in OpenAI format
        model: Model name (e.g., 'gemini-pro', 'gemini-2.5-pro')
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        stop_token: Stop token (currently not supported by Gemini)
        
    Returns:
        Generated response string.
    """
    _configure_google_api()
    
    # Create model instance
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=[stop_token] if stop_token else None,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Convert messages format
    gemini_messages = _convert_openai_messages_to_gemini(messages)
    
    try:
        # For single message, use generate_content
        if len(gemini_messages) == 1:
            response = model_instance.generate_content(
                gemini_messages[0]["parts"][0]
            )
        else:
            # For multi-turn conversation, use chat
            chat = model_instance.start_chat(history=gemini_messages[:-1])
            response = chat.send_message(
                gemini_messages[-1]["parts"][0]
            )
        
        return response.text if response.text else ""
        
    except Exception as e:
        logging.error(f"Google API error: {e}")
        raise e


@retry_with_exponential_backoff
def generate_from_google_completion(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Generate from Google Gemini using completion-style prompt.
    
    Args:
        prompt: The prompt string
        model: Model name (e.g., 'gemini-pro', 'gemini-2.5-pro') 
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        stop_token: Stop token (currently not supported by Gemini)
        
    Returns:
        Generated response string.
    """
    _configure_google_api()
    
    # Create model instance
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=[stop_token] if stop_token else None,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    try:
        response = model_instance.generate_content(prompt)
        return response.text if response.text else ""
        
    except Exception as e:
        logging.error(f"Google API error: {e}")
        raise e


# Debug function for testing (mirrors OpenAI implementation)
@retry_with_exponential_backoff
def fake_generate_from_google_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Debug function that returns a fake response without calling the API."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError(
            "GOOGLE_API_KEY environment variable must be set when using Google API."
        )
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer