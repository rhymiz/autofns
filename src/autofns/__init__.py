import asyncio
import json
import logging
from typing import Any, Callable

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

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
        response_format: dict[str, str] | None = None,
    ) -> None:
        """
        :param model: The model to use.
        :type model: str
        :param fns_definitions: The list of function definitions.
        :type fns_definitions: list[dict[str, Any]]
        :param fns_mapping: The mapping of function names to functions.
        :type fns_mapping: dict[str, Callable[..., Any]] | None
        :param api_key: The OpenAI API key.
        :type api_key: str | None
        :param response_format: The response format.
        :type response_format: dict[str, str] | None
        """

        self.model = model
        self.fns_mapping = fns_mapping or {}
        self.fns_definitions = fns_definitions
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.response_format = response_format or _DEFAULT_RESPONSE_FORMAT

    @property
    def _default_completion_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tools": self.fns_definitions,
            "response_format": self.response_format,
        }

    def _build_completion_kwargs(self, **kwargs) -> dict[str, Any]:
        completion_kwargs = self._default_completion_kwargs
        completion_kwargs.update(kwargs)
        return completion_kwargs

    def _process_tool_calls(
        self,
        calls: list[ChatCompletionMessageToolCall],
        messages: list[Any],
    ) -> list[Any]:
        """
        Iterate over the tool calls and call the corresponding functions.
        """

        for tool_call in calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            tool_fn = self.fns_mapping.get(fn_name)

            if tool_fn is None:
                return_value = f"Error: function '{fn_name}' is not defined."
                _logger.error(f"Call for undefined function: {fn_name}")
            else:
                _logger.debug(f"calling function '{fn_name}'")
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

    def create_completion(
        self,
        messages: list[Any],
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Submit a chat completion request and process tool calls, if any.

        :param messages: The list of messages.
        :type messages: list[Any]
        :param max_tokens: The max tokens to use.
        :type max_tokens: int | None
        :param kwargs: The additional kwargs to be passed to completions.create().
        :type kwargs: Any
        :return: The chat completion response.
        """

        completion_kwargs = self._build_completion_kwargs(
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        while True:
            response = self.client.chat.completions.create(**completion_kwargs)

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                break

            messages.append(response.choices[0].message)

            self._process_tool_calls(tool_calls, messages)

        return response


class AutoFNSAsync(AutoFNS):
    """
    A utility class that wraps an AsyncOpenAI client and automates the
    tool calls processes.
    """

    def __init__(
        self,
        model: str,
        /,
        fns_definitions: list[dict[str, Any]],
        fns_mapping: dict[str, Callable[..., Any]] | None = None,
        api_key: str | None = None,
        response_format: dict[str, str] | None = None,
    ) -> None:
        """
        :param model: The model to use.
        :type model: str
        :param fns_definitions: The list of function definitions.
        :type fns_definitions: list[dict[str, Any]]
        :param fns_mapping: The mapping of function names to functions.
        :type fns_mapping: dict[str, Callable[..., Any]] | None
        :param api_key: The OpenAI API key.
        :type api_key: str | None
        :param response_format: The response format.
        :type response_format: dict[str, str] | None
        """

        super().__init__(model, fns_definitions, fns_mapping, api_key, response_format)
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def _process_tool_calls(
        self,
        calls: list[ChatCompletionMessageToolCall],
        messages: list[Any],
    ) -> list[Any]:
        """
        Iterate over the tool calls and call the corresponding functions.
        """

        for tool_call in calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            tool_fn = self.fns_mapping.get(fn_name)

            if tool_fn is None:
                return_value = f"Error: function '{fn_name}' is not defined."
                _logger.error(f"Call for undefined function: {fn_name}")
            else:
                _logger.debug(f"calling function '{fn_name}'")
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
    ) -> ChatCompletion:
        """
        Submit a chat completion request and process tool calls, if any.

        :param messages: The list of messages.
        :type messages: list[Any]
        :param max_tokens: The max tokens to use.
        :type max_tokens: int | None
        :param kwargs: The additional kwargs to be passed to completions.create().
        :type kwargs: Any
        :return: The chat completion response.
        :rtype: ChatCompletion
        """

        completion_kwargs = self._build_completion_kwargs(
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        while True:
            response = await self.client.chat.completions.create(**completion_kwargs)

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                break

            messages.append(response.choices[0].message)

            await self._process_tool_calls(tool_calls, messages)

        return response


__all__ = ["AutoFNS", "AutoFNSAsync"]
