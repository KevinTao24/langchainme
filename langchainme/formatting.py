"""自定义Formatter子类，提供更严格的参数检查"""

from string import Formatter
from typing import Any, List, Mapping, Sequence, Union


class StrictFormatter(Formatter):
    """提供一个严格的字符串格式化工具，它可以检查是否有未使用的参数，并确保所有参数都作为关键字参数提供"""

    def check_unused_args(
        self,
        used_args: Sequence[Union[int, str]],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> None:
        """检查是否有未使用的关键字参数。如果有，它会抛出一个KeyError。"""

        extra = set(kwargs).difference(used_args)
        if extra:
            raise KeyError(extra)

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """确保没有位置参数被提供"""

        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )

        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: List[str]
    ) -> None:
        """验证格式字符串是否可以接受输入变量列表中的所有变量"""

        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        super().format(format_string, **dummy_inputs)


formatter = StrictFormatter()
