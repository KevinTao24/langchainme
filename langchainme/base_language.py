"""所有语言模型的抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Set

from pydantic import BaseModel
from langchainme.schema import BaseMessage, LLMResult, PromptValue, get_buffer_string

from langchainme.callbacks.manager import Callbacks


def _get_token_ids_default_method(text: str) -> List[int]:
    """将文本转换为一系列的 token ID."""
    # TODO: this method may not be exact.
    # TODO: this method may differ based on model (eg codex).
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)


class BaseLanguageModel(BaseModel, ABC):
    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """根据提示内容列表返回LLMResult类型的结果."""

    @abstractmethod
    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        """根据文本预测文本"""

    @abstractmethod
    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        """根据消息预测消息"""

    @classmethod
    def all_required_field_names(cls) -> Set:
        all_required_field_names = set()
        for field in cls.__fields__.values():
            all_required_field_names.add(field.name)
            if field.has_alias:
                all_required_field_names.add(field.alias)
        return all_required_field_names

    def get_token_ids(self, text: str) -> List[int]:
        """获取文本的令牌ID"""

        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """获得文本的令牌数量"""

        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """获取消息队列中的所有令牌的总数"""

        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])
