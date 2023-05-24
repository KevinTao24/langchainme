"""封装大语言模型的APIs"""

from typing import Dict, Type

from langchainme.llms.base import BaseLLM
from langchainme.llms.openai import OpenAI

__all__ = ["OpenAI"]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "openai": OpenAI,
}
