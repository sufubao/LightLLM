import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from lightllm.server import api_openai, build_prompt
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_models import ChatCompletionRequest


@pytest.fixture
def start_args(monkeypatch):
    args = SimpleNamespace(default_chat_template_kwargs=None, reasoning_parser="qwen3")
    monkeypatch.setattr(build_prompt, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: True)
    monkeypatch.setattr(build_prompt, "get_model_type_v1", lambda: "qwen3")
    return args


def chat_request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], **kwargs)


@pytest.mark.parametrize("flag", ["--default_chat_template_kwargs", "--default-chat-template-kwargs"])
def test_cli_accepts_json_object(flag):
    args = make_argument_parser().parse_args([flag, '{"preserve_thinking": true}'])
    assert args.default_chat_template_kwargs == {"preserve_thinking": True}


@pytest.mark.parametrize("value", ["[]", "null", "true", "42", '"text"', "{invalid"])
def test_cli_rejects_non_objects(value):
    with pytest.raises(SystemExit) as error:
        make_argument_parser().parse_args(["--default-chat-template-kwargs", value])
    assert error.value.code == 2


def test_request_rejects_non_object_kwargs():
    with pytest.raises(ValidationError):
        chat_request(chat_template_kwargs=["invalid"])


def test_absent_defaults_remain_compatible(start_args):
    assert make_argument_parser().parse_args([]).default_chat_template_kwargs is None
    del start_args.default_chat_template_kwargs
    assert build_prompt.get_effective_chat_template_kwargs(chat_request()) == {}


def test_request_overrides_defaults_without_mutating_inputs(start_args):
    start_args.default_chat_template_kwargs = {"preserve_thinking": True, "custom": {"value": 1}}
    request = chat_request(chat_template_kwargs={"preserve_thinking": False, "other": 2})
    defaults = deepcopy(start_args.default_chat_template_kwargs)
    request_kwargs = deepcopy(request.chat_template_kwargs)

    result = build_prompt.get_effective_chat_template_kwargs(request)

    assert result == {"preserve_thinking": False, "custom": {"value": 1}, "other": 2}
    assert start_args.default_chat_template_kwargs == defaults
    assert request.chat_template_kwargs == request_kwargs
    assert build_prompt.get_effective_chat_template_kwargs(chat_request()) == defaults


@pytest.mark.parametrize("parser", ["qwen3", "deepseek-v3"])
@pytest.mark.parametrize("key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize("enabled", [False, True])
def test_default_thinking_aliases_reach_parser_and_template(start_args, monkeypatch, parser, key, enabled):
    start_args.reasoning_parser = parser
    start_args.default_chat_template_kwargs = {key: enabled, "preserve_thinking": True}
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "prompt"
    monkeypatch.setattr(build_prompt, "tokenizer", tokenizer)
    request = chat_request()

    assert api_openai._is_force_thinking_mode(request) is enabled
    assert asyncio.run(build_prompt.build_prompt(request, None)) == "prompt"
    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["thinking"] is enabled
    assert kwargs["enable_thinking"] is enabled
    assert kwargs["preserve_thinking"] is True
    assert start_args.default_chat_template_kwargs == {key: enabled, "preserve_thinking": True}


@pytest.mark.parametrize("parser", ["qwen3", "deepseek-v3"])
@pytest.mark.parametrize("key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize("enabled", [False, True])
def test_request_thinking_overrides_defaults(start_args, parser, key, enabled):
    start_args.reasoning_parser = parser
    start_args.default_chat_template_kwargs = {"thinking": not enabled, "enable_thinking": not enabled}
    request = chat_request(chat_template_kwargs={key: enabled})
    assert api_openai._is_force_thinking_mode(request) is enabled


@pytest.mark.parametrize("effort, enabled", [("none", False), ("high", True)])
def test_request_reasoning_effort_overrides_default_thinking(start_args, monkeypatch, effort, enabled):
    start_args.default_chat_template_kwargs = {"thinking": not enabled}
    tokenizer = Mock()
    monkeypatch.setattr(build_prompt, "tokenizer", tokenizer)
    request = chat_request(reasoning_effort=effort)

    assert api_openai._is_force_thinking_mode(request) is enabled
    asyncio.run(build_prompt.build_prompt(request, None))
    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["reasoning_effort"] == effort
    assert kwargs["thinking"] is enabled
    assert kwargs["enable_thinking"] is enabled


@pytest.mark.parametrize("request_kwargs, expected", [(None, "high"), ({"reasoning_effort": "medium"}, "medium")])
def test_explicit_reasoning_effort_takes_precedence_over_defaults(start_args, monkeypatch, request_kwargs, expected):
    start_args.default_chat_template_kwargs = {"reasoning_effort": "low"}
    tokenizer = Mock()
    monkeypatch.setattr(build_prompt, "tokenizer", tokenizer)
    request = chat_request(reasoning_effort="high", chat_template_kwargs=request_kwargs)

    asyncio.run(build_prompt.build_prompt(request, None))

    assert tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"] == expected
