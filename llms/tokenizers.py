from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # Handle new models that tiktoken doesn't know about yet
            # by falling back to a known encoding
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # For new GPT models, use cl100k_base encoding (GPT-4 family)
                if model_name.startswith("gpt-5") or model_name.startswith("gpt-4"):
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                # For older GPT models, use p50k_base encoding (GPT-3 family)
                elif model_name.startswith("gpt-3"):
                    self.tokenizer = tiktoken.get_encoding("p50k_base")
                else:
                    # Default fallback to cl100k_base for any other unknown OpenAI model
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
