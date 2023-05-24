"""LLMs大语言模型的抽象基类"""

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import Extra, Field, root_validator, validator

import langchainme
from langchainme.base_language import BaseLanguageModel
from langchainme.callbacks.manager import (
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchainme.schema import (
    AIMessage,
    BaseMessage,
    LLMResult,
    PromptValue,
    get_buffer_string,
)


def _get_verbosity() -> bool:
    return langchainme.verbose


def get_prompts(
    params: Dict[str, Any], prompts: List[str]
) -> Tuple[Dict[int, List], str, List[int], List[str]]:
    """获取已经在缓存中的提示."""

    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    for i, prompt in enumerate(prompts):
        if langchainme.llm_cache is not None:
            cache_val = langchainme.llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


def update_cache(
    existing_prompts: Dict[int, List],
    llm_string: str,
    missing_prompt_idxs: List[int],
    new_results: LLMResult,
    prompts: List[str],
) -> Optional[dict]:
    """更新缓存并获取大语言模型LLM的输出."""

    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if langchainme.llm_cache is not None:
            langchainme.llm_cache.update(prompt, llm_string, result)
    llm_output = new_results.llm_output
    return llm_output


class BaseLLM(BaseLanguageModel, ABC):
    """接收一个提示prompt并返回一个字符串"""

    cache: Optional[bool] = None
    """是否要缓存结果"""
    verbose: bool = Field(default_factory=_get_verbosity)
    """是否要打印响应内容"""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """回调函数的集合用于在生成过程中执行特定的操作"""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """回调管理器用于管理回调函数的执行"""

    class Config:
        """Pydantic模型的配置类"""

        extra = Extra.forbid
        """当模型接收到未定义的字段时，禁止未定义的字段"""
        arbitrary_types_allowed = True
        """是否允许任意类型的属性"""

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """如果callback_manager字段被使用则发出一个弃用警告，并将callback_manager的值移动到callbacks字段."""

        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """如果verbose字段的值为None，则将其设置为全局设置的值（由_get_verbosity()方法获取）."""

        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """子类实现父类BaseLanguageModel的抽象方法"""

        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks)

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        """子类实现父类BaseLanguageModel的抽象方法"""

        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(text, stop=_stop)

    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        """子类实现父类BaseLanguageModel的抽象方法"""

        text = get_buffer_string(messages)
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self(text, stop=_stop)
        return AIMessage(content=content)

    @abstractmethod
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """根据提示列表prompts生成结果"""

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """运行LLM大语言模型并生成预测文本."""

        if not isinstance(prompts, list):
            raise ValueError(
                "Argument 'prompts' is expected to be of type List[str], received"
                f" argument of type {type(prompts)}."
            )
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        disregard_cache = self.cache is not None and not self.cache
        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        if langchainme.llm_cache is None or disregard_cache:
            # This happens when langchainme.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchainme.cache`."
                )
            run_manager = callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompts, invocation_params=params
            )
            try:
                output = (
                    self._generate(prompts, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._generate(prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                run_manager.on_llm_error(e)
                raise e
            run_manager.on_llm_end(output)
            return output
        if len(missing_prompts) > 0:
            run_manager = callback_manager.on_llm_start(
                {"name": self.__class__.__name__},
                missing_prompts,
                invocation_params=params,
            )
            try:
                new_results = (
                    self._generate(missing_prompts, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._generate(missing_prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                run_manager.on_llm_error(e)
                raise e
            run_manager.on_llm_end(new_results)
            llm_output = update_cache(
                existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
            )
        else:
            llm_output = {}
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output)
