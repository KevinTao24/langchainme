"""定义处理和管理聊天消息、代理动作、生成结果的数据结构对象"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Extra, Field, root_validator


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
    content: str  # 消息对象
    additional_kwargs: dict = Field(default_factory=dict)  # 用于存储任何额外的关键字参数，默认字典为空

    @property
    @abstractmethod
    def type(self) -> str:
        """消息类型，用于序列化"""


class HumanMessage(BaseMessage):
    """人发送的消息"""

    example: bool = False  # 这个消息是否是一个示例消息

    @property
    def type(self) -> str:
        return "human"


class AIMessage(BaseMessage):
    """AI发送的消息"""

    example: bool = False  # 这个消息是否是一个示例消息

    @property
    def type(self) -> str:
        return "ai"


class SystemMessage(BaseMessage):
    """系统发送的消息"""

    @property
    def type(self) -> str:
        return "system"


class ChatMessage(BaseMessage):
    """跟聊天模型相关的一种特殊类型消息，发言者可以是任意角色"""

    role: str  # 发言者角色，比如用户user

    @property
    def type(self) -> str:
        return "chat"


class AgentAction(NamedTuple):
    """代理要执行的动作"""

    tool: str  # 表示要使用的工具名称
    tool_input: Union[str, dict]  # 表示工具的输入
    log: str  # 用于记录关于这个执行动作的日志信息


class AgentFinish(NamedTuple):
    """代理完成动作后的返回值."""

    return_values: dict  # 返回值，执行完动作后得到的结果
    log: str  # 记录关于这个完成动作的日志信息


class Generation(BaseModel):
    """表示一次生成的输出结果."""

    text: str  # 生成的文本
    generation_info: Optional[Dict[str, Any]] = None  # 包含提供者的原始生成信息，可能包含生成结束的原因等信息


class ChatGeneration(Generation):
    """表示一次聊天生成的输出结果"""

    text = ""
    message: BaseMessage

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class ChatResult(BaseModel):
    """包含了所有与聊天结果相关的信息."""

    generations: List[ChatGeneration]  # 一个ChatGeneration对象的列表，表示生成的内容
    llm_output: Optional[dict] = None  # 存储任意的LLM提供者特定输出


class LLMResult(BaseModel):
    """包含了所有与LLM结果相关的信息."""

    generations: List[List[Generation]]  # 一个列表的列表，每个输入可能有多个生成，表示生成的内容
    llm_output: Optional[dict] = None  # # 存储任意的LLM提供者特定输出


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """返回prompt字符串，表示提示."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """返回一个BaseMessage对象的列表，表示提示."""
