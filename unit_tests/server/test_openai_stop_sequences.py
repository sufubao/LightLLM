import asyncio
import json
from types import SimpleNamespace

import pytest

from lightllm.server import api_http, api_openai
from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest
from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs import sampling_params as sampling_params_module


STOP_SEQUENCE = "<END>"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeHttpServerManager:
    def __init__(self, chunks):
        self.tokenizer = FakeTokenizer()
        self.chunks = chunks

    def generate(self, prompt, sampling_params, multimodal_params, request):
        async def results():
            for index, (text, status) in enumerate(self.chunks):
                metadata = {
                    "prompt_tokens": 1,
                    "prompt_cache_len": 0,
                    "is_first_token": index == 0,
                }
                yield 8, text, metadata, FinishStatus(status)

        return results()


@pytest.fixture
def fake_generation(monkeypatch):
    manager = FakeHttpServerManager(
        [
            ("Hello <", FinishStatus.NO_FINISH),
            ("END", FinishStatus.NO_FINISH),
            (">", FinishStatus.FINISHED_STOP),
        ]
    )
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", manager)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=None))
    monkeypatch.setattr(
        sampling_params_module,
        "get_env_start_args",
        lambda: SimpleNamespace(enable_prompt_logprobs=False),
    )

    async def fake_build_prompt(request, tools):
        return "prompt"

    monkeypatch.setattr(api_openai, "build_prompt", fake_build_prompt)
    return manager


async def collect_sse(response):
    events = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def streamed_text(events, field):
    output = []
    for event in events:
        for choice in event.get("choices", []):
            if field == "delta":
                output.append(choice.get("delta", {}).get("content") or "")
            else:
                output.append(choice.get(field) or "")
    return "".join(output)


def test_stop_sequence_filter_handles_chunk_boundaries_and_partial_matches():
    stop_filter = api_openai._StopSequenceFilter([STOP_SEQUENCE])

    assert stop_filter.process("safe<") == "safe"
    assert stop_filter.process("END") == ""
    assert stop_filter.process(">must-not-leak", final=True) == ""

    partial_filter = api_openai._StopSequenceFilter([STOP_SEQUENCE])
    assert partial_filter.process("safe<EN") == "safe"
    assert partial_filter.process("", final=True) == "<EN"


def test_chat_completion_omits_stop_sequence(fake_generation):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stop=STOP_SEQUENCE,
    )

    response = asyncio.run(api_openai.chat_completions_impl(request, SimpleNamespace()))

    assert response.choices[0].message.content == "Hello "
    assert response.choices[0].finish_reason == "stop"


def test_streaming_chat_completion_omits_stop_sequence(fake_generation):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stop=STOP_SEQUENCE,
        stream=True,
    )

    response = asyncio.run(api_openai.chat_completions_impl(request, SimpleNamespace()))
    events = asyncio.run(collect_sse(response))

    assert streamed_text(events, "delta") == "Hello "
    assert any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))


def test_completion_omits_stop_sequence(fake_generation):
    request = CompletionRequest(model="test-model", prompt="hello", stop=STOP_SEQUENCE)

    response = asyncio.run(api_openai.completions_impl(request, SimpleNamespace()))

    assert response.choices[0].text == "Hello "
    assert response.choices[0].finish_reason == "stop"


def test_streaming_completion_omits_stop_sequence(fake_generation):
    request = CompletionRequest(model="test-model", prompt="hello", stop=STOP_SEQUENCE, stream=True)

    response = asyncio.run(api_openai.completions_impl(request, SimpleNamespace()))
    events = asyncio.run(collect_sse(response))

    assert streamed_text(events, "text") == "Hello "
    assert any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_stream_usage_chunk_is_opt_in(fake_generation, endpoint):
    if endpoint == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        impl = api_openai.chat_completions_impl
    else:
        request = CompletionRequest(model="test-model", prompt="hello", stream=True)
        impl = api_openai.completions_impl

    response = asyncio.run(impl(request, SimpleNamespace()))
    events = asyncio.run(collect_sse(response))

    assert all(event.get("choices") for event in events)
    assert all("usage" not in event for event in events)


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_stream_usage_chunk_is_emitted_when_requested(fake_generation, endpoint):
    stream_options = {"include_usage": True}
    if endpoint == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options=stream_options,
        )
        impl = api_openai.chat_completions_impl
    else:
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            stream=True,
            stream_options=stream_options,
        )
        impl = api_openai.completions_impl

    response = asyncio.run(impl(request, SimpleNamespace()))
    events = asyncio.run(collect_sse(response))

    assert all("usage" in event for event in events)
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["total_tokens"] == 4
