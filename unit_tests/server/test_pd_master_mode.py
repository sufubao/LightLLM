import asyncio
import json
from types import SimpleNamespace

import pytest
from easydict import EasyDict

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver.pd_loop import _allocate_capacity_share, _update_pd_master_membership
from lightllm.server.httpserver_for_pd_master.admission import (
    AdmissionPolicy,
    AdmissionPriority,
    PDAdmissionController,
    SessionTracker,
)
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager


def test_auto_set_response_parsers_from_qwen35_model_config(tmp_path):
    from lightllm.utils.config_utils import auto_set_response_parsers

    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    args = StartArgs(model_dir=str(tmp_path))

    auto_set_response_parsers(args)

    assert args.tool_call_parser == "qwen3_coder"
    assert args.reasoning_parser == "qwen3"


def test_auto_set_response_parsers_preserves_explicit_values(tmp_path):
    from lightllm.utils.config_utils import auto_set_response_parsers

    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    args = StartArgs(
        model_dir=str(tmp_path),
        tool_call_parser="llama3",
        reasoning_parser="deepseek-r1",
    )

    auto_set_response_parsers(args)

    assert args.tool_call_parser == "llama3"
    assert args.reasoning_parser == "deepseek-r1"


def test_get_server_info_serializes_runtime_easydict(monkeypatch):
    from lightllm.server import api_http

    runtime_args = EasyDict(
        model_name="qwen35_0.8b",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    monkeypatch.setattr(api_http.g_objs, "args", runtime_args)

    assert api_http.get_server_info() == dict(runtime_args)


def test_get_server_info_serializes_start_args_dataclass(monkeypatch):
    from lightllm.server import api_http

    start_args = StartArgs(model_name="qwen35_0.8b")
    monkeypatch.setattr(api_http.g_objs, "args", start_args)

    server_info = api_http.get_server_info()
    assert server_info["model_name"] == "qwen35_0.8b"
    assert server_info["run_mode"] == "normal"


def test_pd_master_models_endpoint_has_created_timestamp(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", model_dir="/tmp/test-model", model_name="test-model")
    manager = SimpleNamespace(get_real_supported_max_req_total_len=lambda: 1024)
    monkeypatch.setattr(api_http, "init_tokenizer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_http.SamplingParams, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.CompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.ChatCompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http, "MetricClient", lambda _port: object())
    monkeypatch.setattr(api_http, "get_shm_port_args", lambda: SimpleNamespace(metric_port=1234))
    monkeypatch.setattr(api_http, "HttpServerManagerForPDMaster", lambda args: manager)
    monkeypatch.setattr(api_http, "get_unique_server_name", lambda: "test-server")
    monkeypatch.setattr(api_http.setproctitle, "setproctitle", lambda _title: None)
    monkeypatch.setattr(api_http.time, "time", lambda: 1234.9)

    global_objs = api_http.G_Objs()
    global_objs.set_args(args)
    monkeypatch.setattr(api_http, "g_objs", global_objs)

    response = asyncio.run(api_http.get_models(None))
    assert response.data[0].created == 1234


def test_elastic_pd_nodes_are_ready_with_at_least_one_node_of_each_role():
    manager = PDManager(StartArgs(pd_master_mode="elastic"))
    assert manager.is_pd_nodes_ready() is False

    manager.prefill_nodes = [object()]
    assert manager.is_pd_nodes_ready() is False

    manager.decode_nodes = [object()]
    assert manager.is_pd_nodes_ready() is True

    manager.prefill_nodes.append(object())
    manager.decode_nodes.append(object())
    assert manager.is_pd_nodes_ready() is True


def test_fixed_pd_nodes_are_ready_only_with_exact_node_counts():
    manager = PDManager(StartArgs(pd_master_mode="2p1d"))
    assert manager.is_pd_nodes_ready() is False

    manager.prefill_nodes = [object(), object()]
    manager.decode_nodes = [object()]
    assert manager.is_pd_nodes_ready() is True

    manager.decode_nodes.append(object())
    assert manager.is_pd_nodes_ready() is False


@pytest.mark.parametrize("mode", ["flex", "xpxd", "1p1"])
def test_malformed_pd_master_mode_is_not_ready(mode):
    manager = PDManager(StartArgs(pd_master_mode=mode))
    assert manager.is_pd_nodes_ready() is False


def test_pd_master_readiness_endpoint_uses_pd_node_readiness(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="elastic")
    pd_manager = PDManager(args)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", SimpleNamespace(pd_manager=pd_manager))

    response = api_http.readiness()
    assert response.status_code == 503
    assert json.loads(response.body) == {"status": "not ready"}

    pd_manager.prefill_nodes = [object()]
    pd_manager.decode_nodes = [object()]
    response = api_http.readiness()
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "ok"}


def test_pd_manager_checks_all_connected_node_health(monkeypatch):
    from lightllm.server.httpserver_for_pd_master import manager as manager_module

    requested_urls = []
    responses = {
        "http://10.0.0.1:8000/health": SimpleNamespace(status_code=200),
        "http://10.0.0.2:8000/health": SimpleNamespace(status_code=200),
    }

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 8, "trust_env": False}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def get(self, url):
            requested_urls.append(url)
            result = responses[url]
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(manager_module.httpx, "AsyncClient", FakeAsyncClient)

    manager = PDManager(StartArgs())
    manager.prefill_nodes = [SimpleNamespace(mode="prefill", client_ip_port="10.0.0.1:8000")]
    manager.decode_nodes = [SimpleNamespace(mode="decode", client_ip_port="10.0.0.2:8000")]

    assert asyncio.run(manager.check_pd_nodes_health()) is True
    assert set(requested_urls) == set(responses)

    responses["http://10.0.0.2:8000/health"] = SimpleNamespace(status_code=503)
    assert asyncio.run(manager.check_pd_nodes_health()) is False

    responses["http://10.0.0.2:8000/health"] = RuntimeError("connection failed")
    assert asyncio.run(manager.check_pd_nodes_health()) is False


def test_pd_manager_without_connected_nodes_is_healthy():
    manager = PDManager(StartArgs())
    assert asyncio.run(manager.check_pd_nodes_health()) is True


def test_pd_node_capacity_is_partitioned_without_overlap():
    master_ids = [30, 10, 20]
    shares = [_allocate_capacity_share(8, master_ids, node_id) for node_id in sorted(master_ids)]

    assert shares == [3, 3, 2]
    assert sum(shares) == 8
    assert _allocate_capacity_share(8, master_ids, 99) == 0


def test_pd_master_membership_change_advances_epoch_and_wakes_heartbeats(monkeypatch):
    async def run():
        manager = SimpleNamespace()
        timestamps = iter([100, 200])
        monkeypatch.setattr("lightllm.server.httpserver.pd_loop.time.time_ns", lambda: next(timestamps))

        _update_pd_master_membership(manager, {20: object(), 10: object()})
        assert manager.pd_master_ids == (10, 20)
        assert manager.pd_master_capacity_epoch == 100
        assert manager.pd_master_membership_changed.is_set()

        manager.pd_master_membership_changed.clear()
        _update_pd_master_membership(manager, {10: object(), 20: object()})
        assert manager.pd_master_membership_changed.is_set() is False

        _update_pd_master_membership(manager, {10: object()})
        assert manager.pd_master_capacity_epoch == 200
        assert manager.pd_master_membership_changed.is_set()

    asyncio.run(run())


def test_pd_manager_uses_latest_decode_capacity_lease_and_cache_telemetry():
    args = StartArgs()
    manager = PDManager(args)
    client_ip_port = "10.0.0.2:8000"
    manager.register_pd(
        {
            "node_id": 2,
            "client_ip_port": client_ip_port,
            "mode": "decode",
            "start_args": {
                "max_req_total_len": args.max_req_total_len,
                "running_max_req_size": 8,
            },
            "capacity_share": 3,
            "capacity_epoch": 100,
        },
        websocket=object(),
    )

    assert manager.get_decode_capacity() == 3

    manager.update_node_load_info(
        {
            "client_ip_port": client_ip_port,
            "total_token_usage_rate": 0.25,
            "capacity_share": 1,
            "capacity_epoch": 99,
        }
    )
    assert manager.get_decode_capacity() == 3

    manager.update_node_load_info(
        {
            "client_ip_port": client_ip_port,
            "total_token_usage_rate": 0.5,
            "capacity_share": 2,
            "capacity_epoch": 101,
            "radix_cache_total_tokens": 700,
            "radix_cache_refed_tokens": 200,
            "radix_cache_capacity_tokens": 1000,
        }
    )
    node = manager.decode_nodes[0]
    assert manager.get_decode_capacity() == 2
    assert node.run_status.total_token_usage_rate == 0.5
    assert node.run_status.radix_cache_total_tokens == 700
    assert node.run_status.radix_cache_refed_tokens == 200
    assert node.run_status.radix_cache_capacity_tokens == 1000


def test_prefill_cache_headroom_is_scaled_to_the_master_decode_lease():
    args = StartArgs()
    manager = PDManager(args)
    manager.register_pd(
        {
            "node_id": 1,
            "client_ip_port": "10.0.0.1:8000",
            "mode": "prefill",
            "start_args": {
                "max_req_total_len": args.max_req_total_len,
                "running_max_req_size": 8,
                "max_image_pixels": args.max_image_pixels,
                "disable_image_resize": args.disable_image_resize,
            },
        },
        websocket=object(),
    )
    manager.register_pd(
        {
            "node_id": 2,
            "client_ip_port": "10.0.0.2:8000",
            "mode": "decode",
            "start_args": {
                "max_req_total_len": args.max_req_total_len,
                "running_max_req_size": 8,
            },
            "capacity_share": 4,
        },
        websocket=object(),
    )
    manager.update_node_load_info(
        {
            "client_ip_port": "10.0.0.1:8000",
            "total_token_usage_rate": 0.25,
            "radix_cache_total_tokens": 800,
            "radix_cache_refed_tokens": 100,
            "radix_cache_capacity_tokens": 1000,
        }
    )

    snapshot = manager.get_prefill_cache_capacity()
    assert snapshot is not None
    assert snapshot.total_tokens == 350
    assert snapshot.capacity_tokens == 375
    assert snapshot.free_tokens == 25


def test_prefill_registration_preserves_existing_inflight_prompt_chars():
    args = StartArgs()
    manager = PDManager(args)

    def register_prefill(node_id, client_ip_port):
        manager.register_pd(
            {
                "node_id": node_id,
                "client_ip_port": client_ip_port,
                "mode": "prefill",
                "start_args": {
                    "max_req_total_len": args.max_req_total_len,
                    "max_image_pixels": args.max_image_pixels,
                    "disable_image_resize": args.disable_image_resize,
                },
            },
            websocket=object(),
        )

    register_prefill(1, "10.0.0.1:8000")
    manager.prefill_nodes[0].dispatched_prompt_chars = 1234
    manager.prefill_nodes[0].dispatched_req_num = 12

    register_prefill(2, "10.0.0.2:8000")

    assert [node.dispatched_prompt_chars for node in manager.prefill_nodes] == [1234, 0]
    assert [node.dispatched_req_num for node in manager.prefill_nodes] == [12, 0]


def test_prefill_reconnection_preserves_other_nodes_inflight_prompt_chars():
    args = StartArgs()
    manager = PDManager(args)

    def pd_info(node_id, client_ip_port):
        return {
            "node_id": node_id,
            "client_ip_port": client_ip_port,
            "mode": "prefill",
            "start_args": {
                "max_req_total_len": args.max_req_total_len,
                "max_image_pixels": args.max_image_pixels,
                "disable_image_resize": args.disable_image_resize,
            },
        }

    manager.register_pd(pd_info(1, "10.0.0.1:8000"), websocket=object())
    manager.register_pd(pd_info(2, "10.0.0.2:8000"), websocket=object())
    manager.prefill_nodes[0].dispatched_prompt_chars = 100
    manager.prefill_nodes[1].dispatched_prompt_chars = 200
    manager.prefill_nodes[0].dispatched_req_num = 1
    manager.prefill_nodes[1].dispatched_req_num = 2

    manager.register_pd(pd_info(3, "10.0.0.2:8000"), websocket=object())

    assert [node.client_ip_port for node in manager.prefill_nodes] == ["10.0.0.1:8000", "10.0.0.2:8000"]
    assert [node.dispatched_prompt_chars for node in manager.prefill_nodes] == [100, 0]
    assert [node.dispatched_req_num for node in manager.prefill_nodes] == [1, 0]


def test_pd_master_inference_health_matches_normal_node_semantics(monkeypatch):
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.time.time", lambda: 1000)
    manager = SimpleNamespace(
        health_timeout=200,
        latest_success_infer_time=900,
        running_request_count=1,
        req_id_to_out_inf={1: object()},
    )

    assert HttpServerManagerForPDMaster.is_healthy(manager) is True

    manager.latest_success_infer_time = 700
    assert HttpServerManagerForPDMaster.is_healthy(manager) is False

    manager.running_request_count = 0
    manager.req_id_to_out_inf.clear()
    assert HttpServerManagerForPDMaster.is_healthy(manager) is True


def test_pd_master_waits_before_dispatching_beyond_decode_capacity():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = StartArgs()
        manager.pd_manager = SimpleNamespace(
            selector=SimpleNamespace(estimate_prompt_cache_hit_rate=lambda _prompt: 0.0),
        )
        manager.admission_policy = AdmissionPolicy()
        manager.admission_controller = PDAdmissionController(lambda: 1, policy=manager.admission_policy)
        manager.session_tracker = SessionTracker(
            ttl_seconds=manager.admission_policy.active_session_ttl_seconds,
            max_sessions=manager.admission_policy.max_tracked_sessions,
        )
        manager.metric_client = SimpleNamespace(histogram_observe=lambda *_args: None)
        manager.running_request_count = 0
        manager.latest_success_infer_time = 0
        dispatched_prompts = []

        async def fake_generate(prompt, *_args):
            dispatched_prompts.append(prompt)
            yield prompt

        manager._generate = fake_generate
        first = manager.generate("first", None, None, None)
        assert await first.__anext__() == "first"

        second = manager.generate("second", None, None, None)
        second_result = asyncio.create_task(second.__anext__())
        await asyncio.sleep(0)
        assert dispatched_prompts == ["first"]
        assert manager.admission_controller.queued_request_count == 1

        await first.aclose()
        assert await second_result == "second"
        assert dispatched_prompts == ["first", "second"]
        await second.aclose()
        assert manager.admission_controller.active_slots == 0

    asyncio.run(run())


def test_pd_master_admission_classifies_session_cache_and_multi_choice_cost():
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.pd_manager = SimpleNamespace(
        selector=SimpleNamespace(estimate_prompt_cache_hit_rate=lambda _prompt: 0.75),
    )
    manager.admission_policy = AdmissionPolicy()
    manager.session_tracker = SessionTracker(ttl_seconds=60, max_sessions=10)
    sampling_params = SimpleNamespace(n=3)

    probable = manager._build_admission_request(
        "abcdefghij",
        sampling_params,
        session_key="session-a",
    )
    assert probable.priority == AdmissionPriority.PROBABLE_CACHE_HIT
    assert probable.decode_slots == 3
    assert probable.estimated_uncached_work == 9

    manager.session_tracker.mark_success("session-a")
    continuation = manager._build_admission_request(
        "abcdefghij",
        sampling_params,
        session_key="session-a",
    )
    assert continuation.priority == AdmissionPriority.CONTINUATION


def test_pd_master_restores_request_count_when_preload_fails():
    class FailingMultimodalParams:
        async def verify_and_preload(self, request):
            raise RuntimeError("preload failed")

    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.args = StartArgs(disable_pd_master_decode_capacity_limit=True)
    manager.running_request_count = 0

    async def consume_generate():
        async for _ in manager.generate("prompt", None, FailingMultimodalParams(), None):
            pass

    with pytest.raises(RuntimeError, match="preload failed"):
        asyncio.run(consume_generate())

    assert manager.running_request_count == 0


def test_pd_master_request_count_covers_async_generator_lifecycle():
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.args = StartArgs(disable_pd_master_decode_capacity_limit=True)
    manager.running_request_count = 0
    inner_generator_closed = False

    async def fake_generate(prompt, sampling_params, multimodal_params, request):
        nonlocal inner_generator_closed
        try:
            assert manager.running_request_count == 1
            yield "result"
        finally:
            inner_generator_closed = True

    manager._generate = fake_generate

    async def consume_one_result_and_close():
        results_generator = manager.generate(None, None, None, None)
        assert await results_generator.__anext__() == "result"
        assert manager.running_request_count == 1
        await results_generator.aclose()

    asyncio.run(consume_one_result_and_close())
    assert manager.running_request_count == 0
    assert inner_generator_closed is True


def test_fixed_pd_master_health_endpoint_combines_ready_and_health_status(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="2p1d")
    pd_manager = PDManager(args)
    pd_node_health_results = iter([True, False])
    pd_node_health_check_count = 0

    async def check_pd_nodes_health():
        nonlocal pd_node_health_check_count
        pd_node_health_check_count += 1
        return next(pd_node_health_results)

    pd_manager.check_pd_nodes_health = check_pd_nodes_health
    httpserver_manager = SimpleNamespace(pd_manager=pd_manager, is_healthy=lambda: True)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", httpserver_manager)

    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "message": "Error",
        "pd_master_mode": "2p1d",
        "inference_healthy": True,
        "pd_nodes_ready": False,
        "pd_nodes_healthy": False,
        "registered_prefill_nodes": 0,
        "registered_decode_nodes": 0,
    }
    assert pd_node_health_check_count == 0

    pd_manager.prefill_nodes = [object(), object()]
    pd_manager.decode_nodes = [object()]
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 200
    assert json.loads(response.body)["message"] == "Ok"
    assert json.loads(response.body)["pd_nodes_ready"] is True
    assert pd_node_health_check_count == 1

    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body)["pd_nodes_healthy"] is False
    assert pd_node_health_check_count == 2

    httpserver_manager.is_healthy = lambda: False
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body)["inference_healthy"] is False
    assert pd_node_health_check_count == 2


def test_elastic_pd_master_health_does_not_check_individual_pd_nodes(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="elastic")
    pd_manager = PDManager(args)
    pd_manager.prefill_nodes = [object()]
    pd_manager.decode_nodes = [object()]

    async def unexpected_pd_node_health_check():
        raise AssertionError("elastic mode must not check individual PD node health")

    pd_manager.check_pd_nodes_health = unexpected_pd_node_health_check
    httpserver_manager = SimpleNamespace(pd_manager=pd_manager, is_healthy=lambda: True)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", httpserver_manager)

    response = asyncio.run(api_http.healthcheck(None))
    response_body = json.loads(response.body)
    assert response.status_code == 200
    assert response_body["message"] == "Ok"
    assert "pd_nodes_healthy" not in response_body
