"""Embeddings抽象基类"""

from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[str]]:
        """对文档进行Embed，嵌入到一个向量空间中，每个文档被表示为一个浮点数列表"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """对一个查询进行Embed，嵌入到一个向量空间中，查询被表示为一个浮点数列表"""
