from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lightllm.server import api_http


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_http.g_objs, "metric_client", SimpleNamespace(counter_inc=lambda *args: None))
    test_client = TestClient(api_http.app, raise_server_exceptions=False)
    yield test_client
    test_client.close()


def test_missing_required_parameter_uses_openai_error_envelope(client):
    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "message": "Missing required parameter: 'model'.",
            "type": "invalid_request_error",
            "param": "model",
            "code": 422,
        }
    }


def test_invalid_parameter_uses_openai_error_envelope(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}], "seed": -2},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "message": "Invalid value for 'seed': Input should be greater than or equal to -1",
            "type": "invalid_request_error",
            "param": "seed",
            "code": 422,
        }
    }
