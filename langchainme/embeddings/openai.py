"""OpenAI Embedding 模型封装类"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchainme.embeddings.base import Embeddings
from langchainme.utils import get_from_dict_or_env


logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    from openai.error import (
        Timeout,
        APIError,
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
    )

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(Timeout)
            | retry_if_exception_type(APIError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(RateLimitError)
            | retry_if_exception_type(ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        return embeddings.client.create(**kwargs)

    return _embed_with_retry(**kwargs)


class OpenAIEmbeddings(BaseModel, Embeddings):
    """使用OpenAI的嵌入模型来获取文本的嵌入表示"""

    client: Any
    model: str = "text-embedding-ada-002"
    deployment: str = model
    openai_api_version: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    max_retries: int = 6
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    headers: Any = None

    class Config:
        """未在模型中定义的字段，将会抛出错误"""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境中是否存在OpenAI的API密钥和Python包。如果不存在，将会抛出错误"""

        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        openai_api_type = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        openai_proxy = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if openai_api_type in ("azure", "azure_ad", "azuread"):
            default_api_version = "2023-03-15-preview"
        else:
            default_api_version = ""
        openai_api_version = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            openai.api_key = openai_api_key
            if openai_organization:
                openai.organization = openai_organization
            if openai_api_base:
                openai.api_base = openai_api_base
            if openai_api_type:
                openai.api_version = openai_api_version
            if openai_api_type:
                openai.api_type = openai_api_type
            if openai_proxy:
                openai.proxy = {"http": openai_proxy, "https": openai_proxy}  # type: ignore[assignment]  # noqa: E501
            values["client"] = openai.Embedding
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 1000
    ) -> List[List[float]]:
        """用于获取一组文本的嵌入表示."""

        return self._get_len_safe_embeddings(
            texts, engine=self.deployment, chunk_size=chunk_size
        )

    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = 1000
    ) -> List[List[float]]:
        """处理长输入文本的嵌入，由于OpenAI嵌入模型对输入长度有限制，将长文本分割成较小的块来解决这个问题"""

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        encoding = tiktoken.model.encoding_for_model(self.model)
        for i, text in enumerate(texts):
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j : j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = embed_with_retry(
                self,
                input=tokens[i : i + _chunk_size],
                engine=self.deployment,
                request_timeout=self.request_timeout,
                headers=self.headers,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = embed_with_retry(
                    self,
                    input="",
                    engine=self.deployment,
                    request_timeout=self.request_timeout,
                    headers=self.headers,
                )["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """用于获取查询文本的嵌入表示"""

        embedding = self._embedding_func(text, engine=self.query_model_name)
        return embedding

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """调用OpenAI Embedding服务"""

        if len(text) > self.embedding_ctx_length:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            return embed_with_retry(
                self,
                input=[text],
                engine=engine,
                request_timeout=self.request_timeout,
                headers=self.headers,
            )["data"][0]["embedding"]
