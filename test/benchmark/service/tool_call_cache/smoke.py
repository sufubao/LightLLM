"""End-to-end correctness probes; run under exp with the service manifest saved."""

import argparse
import json
import urllib.request


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18235")
    args = parser.parse_args()
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo the supplied text.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }
    messages = [{"role": "user", "content": "Call echo with text hello. Use the tool, then wait for its result."}]
    first = post(
        args.url + "/v1/chat/completions",
        {
            "model": "tool-cache-qwen4b",
            "messages": messages,
            "tools": [tool],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls", first
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "echo", first
    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": choice["message"]["tool_calls"][0]["id"], "content": "hello"})
    messages.append({"role": "user", "content": "Now give the final answer in one word. Do not call any tools."})
    second = post(
        args.url + "/v1/chat/completions",
        {
            "model": "tool-cache-qwen4b",
            "messages": messages,
            "tools": [tool],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert second["choices"][0]["finish_reason"] == "stop", second
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] >= first["usage"]["prompt_tokens"] - 16, second
    print(json.dumps({"tool_round": first, "final_round": second}, indent=2))


if __name__ == "__main__":
    main()
