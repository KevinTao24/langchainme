"""公共Schema对象"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """将包含多个消息对象的列表转换成一个字符串"""

    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)


class BaseMessage(BaseModel):
    """消息对象"""

    content: str
    """消息内容"""

    additional_kwargs: dict = Field(default_factory=dict)
    """用于存储任何额外的关键字参数，默认字典为空"""

    @property
    @abstractmethod
    def type(self) -> str:
        """消息类型，用于序列化"""


class HumanMessage(BaseMessage):
    """人类发出的消息"""

    example: bool = False

    @property
    def type(self) -> str:
        return "human"


class AIMessage(BaseMessage):
    """AI发出的消息"""

    example: bool = False

    @property
    def type(self) -> str:
        return "ai"


class SystemMessage(BaseMessage):
    """系统发出的消息"""

    @property
    def type(self) -> str:
        return "system"


class ChatMessage(BaseMessage):
    """任意发言者的消息"""

    role: str

    @property
    def type(self) -> str:
        return "chat"


class Generation(BaseModel):
    """一次生成的输出结果."""

    text: str
    """生成的输出文本"""

    generation_info: Optional[Dict[str, Any]] = None
    """包含提供者的原始生成信息，可能包括完成的原因（例如OpenAI）"""


class LLMResult(BaseModel):
    """包含LLM（大型语言模型）结果的所有相关信息"""

    generations: List[List[Generation]]
    """一个列表，每个元素也是一个列表，包含了一次或多次生成的Generation对象。这是一个二维列表，因为每个输入可能有多个生成结果。"""

    llm_output: Optional[dict] = None
    """用于存储特定LLM提供者的输出"""


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """返回prompt字符串，表示提示."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """返回一个BaseMessage对象的列表，表示提示."""
