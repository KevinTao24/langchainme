from typing import Optional

from langchainme.cache import BaseCache
from langchainme.llms import OpenAI

verbose = False
llm_cache: Optional[BaseCache] = None
