from typing import Callable, TypeVar, ParamSpec
from functools import partial

T = TypeVar("T")
P = ParamSpec("P")


def tag_factory(
    name: str,
    tag_type: str,
    attribute: str,
) -> Callable:
    if tag_type == "input":

        def input_tag(func: Callable[P, T]) -> Callable[P, T]:
            if hasattr(func, "input_tag"):
                func.input_tag.append(attribute)
            else:
                func.input_tag = [attribute]
            return func

        input_tag.__name__ = name
        return input_tag

    elif tag_type == "output":

        def output_tag(func: Callable[P, T]) -> Callable[P, T]:
            if hasattr(func, "output_tag"):
                func.output_tag.append(attribute)
            else:
                func.output_tag = [attribute]
            return func

        output_tag.__name__ = name
        return output_tag
    else:
        raise ValueError(
            f"tag_type must be one of ['input', 'output'] but {tag_type} was given"
        )


def variable_attribute_tag_factory(
    name: str,
    tag_type: str,
) -> Callable:
    def _tag_factory(attribute: str) -> Callable:
        return tag_factory(name, tag_type, attribute)

    return _tag_factory
