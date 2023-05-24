"""缓存抽象基类，定义缓存的基本接口"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchainme.schema import Generation

RETURN_VAL_TYPE = List[Generation]


class BaseCache(ABC):
    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """根据给定的提示prompt和语言模型字符串llm_string在缓存中查找值"""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """根据给定的提示prompt和语言模型字符串llm_string在缓存中更新值"""

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """清除缓存，它可以接受任意数量的关键字参数，这些参数可以用于指定清除缓存的具体方式"""
