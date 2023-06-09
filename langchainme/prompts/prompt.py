"""表示一个语言模型的提示."""

from __future__ import annotations

from string import Formatter
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchainme.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    check_valid_template,
)


class PromptTemplate(BasePromptTemplate, BaseModel):
    """表示一个语言模型的提示."""

    input_variables: List[str]
    template: str
    template_format: str = "f-string"  # 表示提示模板的格式。选项是：'f-string', 'jinja2'
    validate_template: bool = True

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

    class Config:
        extra = Extra.forbid

    def format(self, **kwargs: Any) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        if values["validate_template"]:
            check_valid_template(
                values["template"], values["template_format"], values["input_variables"]
            )
        return values

    @classmethod
    def from_examples(
        cls,
        examples: List[str],
        suffix: str,
        input_variables: List[str],
        example_separator: str = "\n\n",
        prefix: str = "",
    ) -> PromptTemplate:
        """接受一个示例列表、一个后缀、一个输入变量列表、一个示例分隔符和一个前缀，然后创建一个提示。
        这个方法主要用于从示例动态创建提示"""

        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_file(
        cls, template_file: str, input_variables: List[str]
    ) -> PromptTemplate:
        """从文件中加载提示模板."""

        with open(template_file, "r") as f:
            template = f.read()
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        """从模板中加载提示模板"""

        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
        return cls(input_variables=list(sorted(input_variables)), template=template)


Prompt = PromptTemplate
