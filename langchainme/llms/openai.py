"""封装OpenAI APIs."""

import logging
import warnings
from typing import AbstractSet, Any, Dict, Literal, Optional, Tuple, Union

from pydantic import Extra, Field, root_validator

from langchainme.llms.base import BaseLLM
from langchainme.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class BaseOpenAI(BaseLLM):
    """封装OpenAI大语言模型."""

    client: Any
    """用于与 OpenAI API 进行交互的客户端对象"""
    model_name: str = Field("text-davinci-003", alias="model")
    """模型名称，默认为text-davinci-003"""
    temperature: float = 0.7
    """用于控制生成文本的随机性的参数"""
    max_tokens: int = 256
    """生成文本的最大令牌数，如果max_tokens被设置为-1，那么模型将尽可能多地生成令牌，但数量不会超过模型的最大上下文大小"""
    top_p: float = 1
    """每一步生成令牌时要考虑的总概率质量"""
    frequency_penalty: float = 0
    """用于控制生成文本中重复令牌的频率。如果为正，重复令牌的频率将会降低；如果为负，重复令牌的频率将会增加。默认值为0，表示不进行任何调整"""
    presence_penalty: float = 0
    """用于控制生成文本中令牌的出现次数。如果为正，令牌的出现次数将会降低；如果为负，令牌的出现次数将会增加。默认值为0，表示不进行任何调整。"""
    n: int = 1
    """对于每个提示，生成的完成数"""
    best_of: int = 1
    """服务器端生成的最佳完成数"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """保存任何未明确指定的模型参数"""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    """上面三个参数用于与OpenAI API进行交互的参数"""
    batch_size: int = 20
    """传递多个文档进行生成时使用的批处理大小"""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """请求OpenAI API的超时时长. 默认10分钟."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """调整特定令牌生成概率的参数"""
    max_retries: int = 6
    """生成时的最大重试次数"""
    streaming: bool = False
    """是否流式传输结果."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """允许的特殊令牌集"""
    disallowed_special: Union[Literal["all"], AbstractSet[str]] = "all"
    """不允许的特殊令牌集"""

    def __new__(cls, **data: Any) -> Union[OpenAIChat, BaseOpenAI]:  # type: ignore
        """实例化OpenAI对象"""

        model_name = data.get("model_name", "")
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            warnings.warn(
                "You are trying to use a chat model. This way of initializing it is "
                "no longer supported. Instead, please use: "
                "`from langchainme.chat_models import ChatOpenAI`"
            )
            return OpenAIChat(**data)
        return super().__new__(cls)

    class Config:
        """Pydantic模型的配置类"""

        extra = Extra.ignore
        """当模型接收到未定义的字段时，忽略这些未定义的字段"""
        allow_population_by_field_name = True
        """是否允许通过字段名来填充模型"""

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """根据入参构建额外的关键字字典"""

        all_required_field_names = cls.all_required_field_names()
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证OpenAI API Key和Python依赖包"""

        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        try:
            import openai

        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        openai.api_key = openai_api_key
        if openai_organization:
            openai.organization = openai_organization
        if openai_api_base:
            openai.api_base = openai_api_base
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values


class OpenAI:
    def __init__(self) -> None:
        pass


class OpenAIChat:
    def __init__(self) -> None:
        pass
