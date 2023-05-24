from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchainme.schema import LLMResult


class LLMManagerMixin:
    """大语言模型管理混入类，定义了一些方法，在大语言模型LLM的运行过程中，当特定事件发生时这些方法被调用。这些方法通常被称为回调函数"""

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当LLM生成新的令牌时，此回调函数被调用，这个函数只在启用流式处理时可用"""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当LLM完成运行时，此回调函数被调用"""

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当LLM遇到错误时，此回调函数被调用"""


class CallbackManagerMixin:
    """回调管理混入类"""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当大语言模型LLM开始运行时，此回调函数被调用"""


class RunManagerMixin:
    """运行管理混入类"""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当有新的文本生成时被调用"""


class BaseCallbackHandler(LLMManagerMixin, CallbackManagerMixin, RunManagerMixin):
    """基础回调处理器"""

    @property
    def ignore_llm(self) -> bool:
        """是否忽略大语言模型LLM回调"""

        return False


class BaseCallbackManager(CallbackManagerMixin):
    """基础回调管理器，用来处理来自Langchainme的回调"""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """初始化回调管理器"""

        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers or []
        self.parent_run_id = parent_run_id

    @property
    def is_async(self) -> bool:
        """回调管理器是否是异步的"""

        return False

    def add_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """添加处理器，inherit表示是否将处理器添加到可继承的处理器列表中"""

        self.handlers.append(handler)
        if inherit:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """从处理器列表和可继承的处理器列表中删除处理器"""

        self.handlers.remove(handler)
        self.inheritable_handlers.remove(handler)

    def set_handlers(
        self, handlers: List[BaseCallbackHandler], inherit: bool = True
    ) -> None:
        """设置处理器列表，清空处理器列表和可继承的处理器列表，然后将新的处理器添加到这两个列表中"""

        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """设置处理器，新的处理器设置为唯一的处理器"""

        self.set_handlers([handler], inherit=inherit)
