import argparse
import json
import logging
from pathlib import Path
from typing import Any

import jinja2
import tiktoken
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
    openai_llm,
)
from llms.tokenizers import Tokenizer
from pydantic import BaseModel

class BrowserAgentOutput(BaseModel):
    action: str
    reasoning: str

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError

class DistylAgent(Agent):
    
    def __init__(self, lm_config: lm_config.LMConfig, tokenizer: Tokenizer) -> None:

        super().__init__()
        self.lm_config = lm_config
        self.tokenizer = tokenizer

    def next_action(self, trajectory: Trajectory, intent: str, meta_data: Any):
        logger = logging.getLogger(__name__)
        
        try:
            # Load the Jinja2 template
            with open(Path(__file__).parent / "prompts" / "main.j2", 'r') as f:
                template_content = f.read()
            
            # Render the template
            prompt = jinja2.Template(template_content).render(
                trajectory=trajectory,
                intent=intent,
                meta_data=meta_data or {}
            )
            
            logger.info(f"=== DISTYL AGENT PROMPT ===")
            logger.debug(f"Rendered prompt: {prompt}")
            
            # Extract image from latest observation in trajectory
            image_data = trajectory[-1].get("observation", {}).get("image") if trajectory and isinstance(trajectory[-1], dict) else None
            response = openai_llm(prompt, image_data, BrowserAgentOutput)
            
            logger.info(f"=== DISTYL AGENT RESPONSE ===")
            logger.info(f"Action: {response.action}")
            logger.info(f"Reasoning: {response.reasoning}")
            
            action = create_id_based_action(response.action)
            action["reasoning"] = response.reasoning
            return action
                
        except Exception as e:
            logger.error(f"Error in DistylAgent.next_action: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return create_none_action()
    
    def reset(self, test_config_file: str) -> None:
        pass