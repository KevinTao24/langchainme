"""回调管理"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import UUID, uuid4

from langchainme.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    LLMManagerMixin,
    RunManagerMixin,
)
from langchainme.schema import LLMResult, get_buffer_string

logger = logging.getLogger(__name__)
Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]


def _handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """通用事件处理器，用于CallbackManager"""

    message_strings: Optional[List[str]] = None
    for handler in handlers:
        try:
            if ignore_condition_name is None or not getattr(
                handler, ignore_condition_name
            ):
                getattr(handler, event_name)(*args, **kwargs)
        except NotImplementedError as e:
            if event_name == "on_chat_model_start":
                if message_strings is None:
                    message_strings = [get_buffer_string(m) for m in args[1]]
                _handle_event(
                    [handler],
                    "on_llm_start",
                    "ignore_llm",
                    args[0],
                    message_strings,
                    *args[2:],
                    **kwargs,
                )
            else:
                logger.warning(f"Error in {event_name} callback: {e}")
        except Exception as e:
            logging.warning(f"Error in {event_name} callback: {e}")


BRM = TypeVar("BRM", bound="BaseRunManager")


class BaseRunManager(RunManagerMixin):
    """运行管理器的基类"""

    def __init__(
        self,
        run_id: UUID,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: List[BaseCallbackHandler],
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """初始化运行管理器"""

        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id

    @classmethod
    def get_noop_manager(cls: Type[BRM]) -> BRM:
        """获取一个不执行任何操作的管理器"""

        return cls(uuid4(), [], [])


class RunManager(BaseRunManager):
    """运行管理器"""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """当接收到文本时，这个方法被调用"""

        _handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """大语言模型LLM运行的回调管理器"""

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """当LLM生成新的令牌时，此回调函数被调用，这个函数只在启用流式处理时可用"""

        _handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """当LLM完成运行时，此回调函数被调用"""

        _handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """当LLM遇到错误时，此回调函数被调用"""

        _handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """回调管理器，处理来自langchainme的回调"""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: UUID,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """当大语言模型LLM开始运行时，这个方法会被调用"""

        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_llm_start",
            "ignore_llm",
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return
