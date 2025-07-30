from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # Handle new/unknown OpenAI models by mapping to known tokenizers
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fall back to GPT-4 tokenizer for unknown OpenAI models
                if any(gpt in model_name.lower() for gpt in ['gpt-4', 'gpt4']):
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                elif any(gpt in model_name.lower() for gpt in ['gpt-3.5', 'gpt3']):
                    self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # Default fallback for any unknown OpenAI model
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        elif provider == "google":
            # Google/Gemini doesn't provide direct tokenizer access
            # Use GPT-4 tokenizer as approximation since it's similar quality
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError(f"Provider {provider} not implemented")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
