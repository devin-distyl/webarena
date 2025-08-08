"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import base64
import io
import logging
import os
import random
import time
from typing import Any, Union

import aiolimiter
import numpy as np
import numpy.typing as npt
import openai
# import openai.error
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from tqdm.asyncio import tqdm_asyncio


class AgentResponse(BaseModel):
    """Structured response model for GPT-5 agents with explicit reasoning and action"""
    reasoning: str
    action: str

def openai_llm(prompt: str, image: Union[str, npt.NDArray[np.uint8], None] = None, output_model: BaseModel | None = None, model: str = "gpt-4o") -> Union[str, BaseModel]:
    client = OpenAI()

    content = [{"type": "input_text", "text": prompt}]

    if image is not None:
        if isinstance(image, str):
            # Handle file path
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            # Handle numpy array (screenshot from browser)
            pil_image = Image.fromarray(image)
            byte_io = io.BytesIO()
            pil_image.save(byte_io, format="PNG")
            byte_io.seek(0)
            base64_image = base64.b64encode(byte_io.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{base64_image}"
        })

    input_messages = [
        {
            "role": "user",
            "type": "message",
            "content": content
        }
    ]

    if output_model is not None:
        # Use structured output with parse
        response = client.responses.parse(
            model=model,
            input=input_messages,
            text_format=output_model,
        )
        return response.output_parsed
    else:
        # Use regular text output
        response = client.responses.create(
            model=model,
            input=input_messages
        )
        return response.output_text

# def retry_with_exponential_backoff(  # type: ignore
#     func,
#     initial_delay: float = 1,
#     exponential_base: float = 2,
#     jitter: bool = True,
#     max_retries: int = 3,
#     errors: tuple[Any] = (openai.error.RateLimitError,),
# ):
#     """Retry a function with exponential backoff."""

#     def wrapper(*args, **kwargs):  # type: ignore
#         # Initialize variables
#         num_retries = 0
#         delay = initial_delay

#         # Loop until a successful response or max_retries is hit or an exception is raised
#         while True:
#             try:
#                 return func(*args, **kwargs)
#             # Retry on specified errors
#             except errors as e:
#                 # Increment retries
#                 num_retries += 1

#                 # Check if max retries has been reached
#                 if num_retries > max_retries:
#                     raise Exception(
#                         f"Maximum number of retries ({max_retries}) exceeded."
#                     )

#                 # Increment the delay
#                 delay *= exponential_base * (1 + jitter * random.random())
#                 print(f"Retrying in {delay} seconds.")
#                 # Sleep for the delay
#                 time.sleep(delay)

#             # Raise exceptions for any errors not specified
#             except Exception as e:
#                 raise e

#     return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.environ.get("OPENAI_ORGANIZATION", None)
        )
        
        for _ in range(3):
            try:
                # Handle different parameter names for different model families
                kwargs = {
                    "model": engine,
                    "prompt": prompt,
                    "temperature": temperature,
                }
                
                # GPT-5 models use different parameter names
                if engine.startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = max_tokens
                    # GPT-5 models support top_p
                    kwargs["top_p"] = top_p
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["top_p"] = top_p
                
                response = await client.completions.create(**kwargs)
                return {"choices": [{"text": response.choices[0].text}]}
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"text": ""}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


# @retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.environ.get("OPENAI_ORGANIZATION", None)
    )
    
    # Handle different parameter names for different model families
    kwargs = {
        "prompt": prompt,
        "model": engine,
        "temperature": temperature,
        "stop": [stop_token] if stop_token else None,
    }
    
    # GPT-5 models use different parameter names
    if engine.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
        # GPT-5 models support top_p
        kwargs["top_p"] = top_p
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["top_p"] = top_p
    
    response = client.completions.create(**kwargs)
    answer: str = response.choices[0].text
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.environ.get("OPENAI_ORGANIZATION", None)
        )
        
        for _ in range(3):
            try:
                # Handle different parameter names for different model families
                kwargs = {
                    "model": model,
                    "messages": messages,
                }
                
                # GPT-5 models use different parameter names and restrictions
                if model.startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = max_tokens
                    # GPT-5 nano only supports temperature=1 (default), so don't include it
                    if not model.startswith("gpt-5-nano"):
                        kwargs["temperature"] = temperature
                        # GPT-5 full model doesn't support top_p
                        if not model.startswith("gpt-5-2025"):
                            kwargs["top_p"] = top_p
                else:
                    kwargs["temperature"] = temperature
                    kwargs["max_tokens"] = max_tokens
                    kwargs["top_p"] = top_p
                
                response = await client.chat.completions.create(**kwargs)
                return {"choices": [{"message": {"content": response.choices[0].message.content}}]}
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
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


# @retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
    output_model: BaseModel | None = None,
) -> str | BaseModel:
    logger = logging.getLogger(__name__)

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    try:
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.environ.get("OPENAI_ORGANIZATION", None)
        )
        
        # Use structured outputs if output_model is provided
        if output_model:
            # Use structured outputs with beta API
            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": output_model,
                "stop": [stop_token] if stop_token else None,
            }
            
            # GPT-5 models use different parameter names and restrictions
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
                # GPT-5 nano only supports temperature=1 (default), so don't include it
                if not model.startswith("gpt-5-nano"):
                    kwargs["temperature"] = temperature
                    # GPT-5 full model doesn't support top_p
                    if not model.startswith("gpt-5-2025"):
                        kwargs["top_p"] = top_p
            
            try:
                response = client.beta.chat.completions.parse(**kwargs)
                parsed_response = response.choices[0].message.parsed
                
                
                if parsed_response:
                    return parsed_response
                else:
                    logger.warning(f"GPT-5 structured response was None, falling back to raw content")
                    raw_content = response.choices[0].message.content or ""
                    return raw_content
                    
            except Exception as e:
                logger.error(f"GPT-5 structured output failed: {e}, falling back to regular chat completion")
                # Fall through to regular completion below
        
        else:
            # Use standard chat completions for non-GPT-5 models
            kwargs = {
                "model": model,
                "messages": messages,
                "stop": [stop_token] if stop_token else None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            response = client.chat.completions.create(**kwargs)
            answer: str = response.choices[0].message.content
            
            
            return answer
    except Exception as e:
        logger.error(f"Error calling OpenAI chat completion: {e}")
        raise e


