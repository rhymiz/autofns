import asyncio
import json
import logging
from typing import Any, Callable, Type, Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

_CLIENT_TYPE = Union[Type[AsyncOpenAI], Type[OpenAI]]
_DEFAULT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

_logger = logging.getLogger("autofns")


class AutoFNS:
    """
    A utility class that wraps an OpenAI client and automates the
    tool calls processes.
    """

    def __init__(
        self,
        model: str,
        /,
        fns_definitions: list[dict[str, Any]],
        fns_mapping: dict[str, Callable[..., Any]] | None = None,
        api_key: str | None = None,
        client: _CLIENT_TYPE = OpenAI,
        response_format: dict[str, str] | None = None,
    ):
        self.model = model
        self.fns_mapping = fns_mapping or {}
        self.fns_definitions = fns_definitions

        if api_key:
            self.client = client(api_key=api_key)
        else:
            # Client will try to get the API key from
            # the environment variable OPENAI_API_KEY
            self.client = client()

        self.response_format = response_format or _DEFAULT_RESPONSE_FORMAT

    def map_function(self, fn_name: str | None = None):
        """
        Decorator to map a function to AutoFNS.

        :param fn_name:
        :type fn_name:
        :return:
        :rtype:
        """

        def decorator(fn: Callable[..., Any]):
            fn_name_ = fn_name or fn.__name__

            fn_definition = next(
                (
                    fn_definition
                    for fn_definition in self.fns_definitions
                    if fn_definition["function"]["name"] == fn_name_
                ),
                None,
            )

            if fn_definition is None:
                raise ValueError(
                    f"Function '{fn_name_}' is not defined in fns_definitions."
                )

            self.fns_mapping[fn_name_] = fn

            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    def default_kwargs(self) -> dict[str, Any]:
        """
        Build the kwargs to pass to the API.

        :return: The kwargs to pass to the API.
        """

        kwargs = {
            "model": self.model,
            "tools": self.fns_mapping,
            "response_format": self.response_format,
        }
        return kwargs

    def create_completion(
        self,
        messages: list[Any],
        max_tokens: int | None = None,
        **kwargs,
    ):
        _kwargs = self.default_kwargs()
        _kwargs["messages"] = messages
        _kwargs["max_tokens"] = max_tokens
        _kwargs.update(kwargs)

        while True:
            response = self.client.completions.create(**kwargs)

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                break

            messages.append(response.choices[0].message)

            self.handle_tool_calls(tool_calls, messages)

        return response

    def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        messages: list[Any],
    ) -> list[Any]:
        """
        Handle the tool calls.

        :param tool_calls: The list of tool calls.
        :type tool_calls: list[ChatCompletionMessageToolCall]
        :param messages: The list of current messages.
        :type messages: list[Any]
        :return: The list of messages with the tool create_completion results.
        :rtype: list[Any]
        """

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            tool_fn = self.fns_mapping.get(fn_name)

            if tool_fn is None:
                return_value = f"Error: function '{fn_name}' is not defined."
                _logger.warning(f"Call for undefined function: {fn_name}")
            else:
                _logger.info(f"calling function '{fn_name}'")
                return_value = tool_fn(**json.loads(fn_args))

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": return_value,
                }
            )
        return messages


class AutoFNSAsync(AutoFNS):
    def __init__(
        self,
        model: str,
        /,
        fns_mapping: dict[str, Callable[..., Any]],
        fns_definitions: list[dict[str, Any]],
        api_key: str | None = None,
        client: _CLIENT_TYPE = AsyncOpenAI,
        response_format: dict[str, str] | None = None,
    ):
        super().__init__(
            model,
            fns_definitions,
            fns_mapping,
            api_key,
            client,
            response_format,
        )

    async def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        messages: list[Any],
    ) -> list[Any]:
        """
        Handle the tool calls.

        :param tool_calls: The list of tool calls.
        :type tool_calls: list[ChatCompletionMessageToolCall]
        :param messages: The list of current messages.
        :type messages: list[Any]
        :return: The list of messages with the tool create_completion results.
        :rtype: list[Any]
        """

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            tool_fn = self.fns_mapping.get(fn_name)

            if tool_fn is None:
                return_value = f"Error: function '{fn_name}' is not defined."
                _logger.warning(f"Call for undefined function: {fn_name}")
            else:
                _logger.info(f"calling function '{fn_name}'")
                if asyncio.iscoroutinefunction(tool_fn):
                    return_value = await tool_fn(**json.loads(fn_args))
                else:
                    return_value = tool_fn(**json.loads(fn_args))

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": return_value,
                }
            )
        return messages

    async def create_completion(
        self,
        messages: list[Any],
        max_tokens: int | None = None,
        **kwargs,
    ):
        _kwargs = self.default_kwargs()
        _kwargs["messages"] = messages
        _kwargs["max_tokens"] = max_tokens
        _kwargs.update(kwargs)

        while True:
            response = await self.client.completions.create(**kwargs)

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                break

            messages.append(response.choices[0].message)

            await self.handle_tool_calls(tool_calls, messages)

        return response


__all__ = ["AutoFNS", "AutoFNSAsync"]

fns = AutoFNS("davinci")


@fns.map_function
def my_function():
    pass
