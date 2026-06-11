# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import collections
import time

import uvloop
import requests
import base64
import os
from io import BytesIO
import pickle
import setproctitle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
import uuid
from PIL import Image
import multiprocessing as mp
from typing import AsyncGenerator, Union
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, JSONResponse
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs.io_objs import ProfileControlReq
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score
from lightllm.utils.envs_utils import get_env_start_args, get_lightllm_websocket_max_message_size
from lightllm.utils.log_utils import init_logger
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from dataclasses import dataclass

from .api_openai import chat_completions_impl, completions_impl
from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelListResponse,
)
from .build_prompt import build_prompt, init_tokenizer

logger = init_logger(__name__)


@dataclass
class G_Objs:
    app: FastAPI = None
    metric_client: MetricClient = None
    args: StartArgs = None
    g_generate_func: Callable = None
    g_generate_stream_func: Callable = None
    httpserver_manager: Union[HttpServerManager, HttpServerManagerForPDMaster] = None
    shared_token_load: TokenLoad = None
    # OpenAI-compatible "created" timestamp for /v1/models.
    # Should be stable for the lifetime of this server process.
    model_created: int = None

    def set_args(self, args: StartArgs):
        self.args = args
        from .api_lightllm import lightllm_generate, lightllm_generate_stream
        from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl

        if args.use_tgi_api:
            self.g_generate_func = tgi_generate_impl
            self.g_generate_stream_func = tgi_generate_stream_impl
        else:
            self.g_generate_func = lightllm_generate
            self.g_generate_stream_func = lightllm_generate_stream

        setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::api_server")

        if args.run_mode == "pd_master":
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManagerForPDMaster(
                args=args,
            )
        else:
            init_tokenizer(args)  # for openai api
            SamplingParams.load_generation_cfg(args.model_dir)
            CompletionRequest.load_generation_cfg(args.model_dir)
            ChatCompletionRequest.load_generation_cfg(args.model_dir)
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManager(args=args)
            dp_size_in_node = max(1, args.dp // args.nnodes)  # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
            self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", dp_size_in_node)
            if self.model_created is None:
                self.model_created = int(time.time())


g_objs = G_Objs()

LIGHTLLM_PROFILE_DIR_ROOT = os.getenv("LIGHTLLM_TORCH_PROFILER_DIR", "/tmp/lightllm_profile")
_PROFILE_ALLOWED_ACTIVITIES = {"CPU", "GPU"}

app = FastAPI()
g_objs.app = app

_ACCESS_LOG_STATUS_COLORS = {2: "\033[32m", 3: "\033[36m", 4: "\033[33m", 5: "\033[31m"}
_ACCESS_LOG_STATUS_COLORS = {2: "\033[32m", 3: "\033[36m", 4: "\033[33m", 5: "\033[31m"}
_ACCESS_LOG_RESET = "\033[0m"


class _AccessLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["type"] == "http":
                status = status_holder["status"]
                msg = f"{scope['method']} {scope['path']} {status}"
                color = _ACCESS_LOG_STATUS_COLORS.get(status // 100, "")
                if color:
                    msg = color + msg + _ACCESS_LOG_RESET
                logger.info(msg)


app.add_middleware(_AccessLogMiddleware)


def create_error_response(
    status_code: HTTPStatus, message: str, err_type: str = None, param: str = None
) -> JSONResponse:
    if err_type is None:
        if status_code.value >= 500:
            err_type = "InternalServerError"
        elif status_code == HTTPStatus.NOT_FOUND:
            err_type = "NotFoundError"
        else:
            err_type = "BadRequestError"

    g_objs.metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "param": param, "code": status_code.value}},
        status_code=status_code.value,
    )


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    return {"model_name": g_objs.args.model_name}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    if g_objs.args.run_mode == "pd_master":
        return JSONResponse({"message": "Ok"}, status_code=200)

    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=503)
    from lightllm.utils.health_check import health_check

    is_healthy = health_check(g_objs.httpserver_manager.shm_req_manager)
    return JSONResponse(
        {"message": "Ok" if is_healthy else "Error"},
        status_code=200 if is_healthy else 503,
    )


@app.get("/token_load", summary="Get the current server's load of tokens")
async def token_load(request: Request):
    ans_dict = {
        # 当前使用 token 量，估计的负载
        "current_load": [
            float(g_objs.shared_token_load.get_current_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 朴素估计的负载，简单将当前请求的输入和输出长度想加得到,目前已未使用，其值与 dynamic_max_load 一样。
        "logical_max_load": [
            float(g_objs.shared_token_load.get_logical_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 动态估计的最大负载，考虑请求中途退出的情况的负载
        "dynamic_max_load": [
            float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
    }

    if g_objs.args.dp == 1:
        ans_dict = {k: v[0] for k, v in ans_dict.items()}

    return JSONResponse(ans_dict, status_code=200)


def _check_profiling_enabled():
    if not g_objs.args.enable_profiling:
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED, "profiling is not enabled, launch the server with --enable_profiling"
        )
    return None


@app.post("/start_profile", summary="Arm an on-demand torch.profiler capture on all worker ranks")
async def start_profile(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    try:
        body = await request.json()
    except Exception:
        body = {}

    targets = body.get("targets", ["worker"])
    if any(target != "worker" for target in targets):
        return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "only the 'worker' target is supported")

    activities = body.get("activities", ["CPU", "GPU"])
    if not activities or not set(activities).issubset(_PROFILE_ALLOWED_ACTIVITIES):
        return create_error_response(
            HTTPStatus.BAD_REQUEST, f"activities must be a non-empty subset of {sorted(_PROFILE_ALLOWED_ACTIVITIES)}"
        )

    num_steps = body.get("num_steps")
    start_step = body.get("start_step")
    for name, value in (("num_steps", num_steps), ("start_step", start_step)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            return create_error_response(HTTPStatus.BAD_REQUEST, f"{name} must be a positive integer")

    root_dir = os.path.realpath(LIGHTLLM_PROFILE_DIR_ROOT)
    output_dir = os.path.realpath(body.get("output_dir") or root_dir)
    if output_dir != root_dir and not output_dir.startswith(root_dir + os.sep):
        return create_error_response(HTTPStatus.BAD_REQUEST, f"output_dir must be under {root_dir}")

    profile_id = int(time.time() * 1000)
    profile_req = ProfileControlReq(
        action="start",
        profile_id=profile_id,
        targets=targets,
        output_dir=output_dir,
        num_steps=num_steps,
        start_step=start_step,
        activities=activities,
        with_stack=bool(body.get("with_stack", True)),
        record_shapes=bool(body.get("record_shapes", False)),
        profile_prefix=str(body.get("profile_prefix", "lightllm")),
    )
    await g_objs.httpserver_manager.send_profile_control(profile_req)
    # 202: 仅代表命令已入队, worker 实际状态请轮询 /profile_status。
    return JSONResponse({"status": "accepted", "profile_id": profile_id, "output_dir": output_dir}, status_code=202)


@app.post("/stop_profile", summary="Stop a running capture and flush traces")
async def stop_profile(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    profile_req = ProfileControlReq(action="stop", profile_id=0)
    await g_objs.httpserver_manager.send_profile_control(profile_req)
    return JSONResponse({"status": "accepted"}, status_code=202)


@app.get("/profile_status", summary="Per-rank profiler state")
async def profile_status(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    board = g_objs.httpserver_manager.profile_status_board
    return JSONResponse(
        {
            "workers": [board.get_slot(slot) for slot in range(board.num_worker_slots)],
            "router": board.get_slot(board.router_slot),
        },
        status_code=200,
    )


@app.post("/generate")
async def generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.error("%s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_stream_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.error("%s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await lightllm_get_score(request, g_objs.httpserver_manager)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        resp = await chat_completions_impl(request, raw_request)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        resp = await completions_impl(request, raw_request)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/v1/messages")
async def anthropic_messages(raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_anthropic import anthropic_messages_impl

    try:
        return await anthropic_messages_impl(raw_request)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)


@app.get("/v1/models", response_model=ModelListResponse)
async def get_models(raw_request: Request):
    model_name = g_objs.args.model_name
    max_model_len = g_objs.httpserver_manager.get_real_supported_max_req_total_len()

    if model_name == "default_model_name" and g_objs.args.model_dir:
        model_name = os.path.basename(g_objs.args.model_dir.rstrip("/"))

    return ModelListResponse(
        data=[
            ModelCard(
                id=model_name,
                created=g_objs.model_created,
                max_model_len=max_model_len,
                owned_by=g_objs.args.model_owner or "lightllm",
            )
        ]
    )


@app.get("/tokens")
@app.post("/tokens")
async def tokens(request: Request):
    try:
        request_dict = await request.json()
        prompt = request_dict.pop("text")
        sample_params_dict = request_dict.pop("parameters", {})

        sampling_params = SamplingParams()
        sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sample_params_dict)
        sampling_params.verify()

        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)
        await multimodal_params.verify_and_preload(request)
        return JSONResponse(
            {
                "ntokens": g_objs.httpserver_manager.tokens(
                    prompt, multimodal_params, sampling_params, sample_params_dict
                )
            },
            status_code=200,
        )
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await g_objs.metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


@app.websocket("/pd_register")
async def register_and_keep_alive(websocket: WebSocket):
    await websocket.accept()
    websocket._receive_bytes_max_size = get_lightllm_websocket_max_message_size()
    client_ip, client_port = websocket.client
    logger.info(f"Client connected from IP: {client_ip}, Port: {client_port}")
    regist_json = json.loads(await websocket.receive_text())
    logger.info(f"received regist_json {regist_json}")
    await g_objs.httpserver_manager.register_pd(regist_json, websocket)

    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_bytes()
            obj = pickle.loads(data)
            await g_objs.httpserver_manager.put_to_handle_queue(obj)

    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"client {regist_json} has error {str(e)}")
        logger.exception(str(e))
    finally:
        logger.error(f"client {regist_json} removed")
        await g_objs.httpserver_manager.remove_pd(regist_json)
    return


@app.websocket("/kv_move_status")
async def kv_move_status(websocket: WebSocket):
    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"kv_move_status Client connected from IP: {client_ip}, Port: {client_port}")
    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_bytes()
            upkv_status = pickle.loads(data)
            logger.info(f"received upkv_status {upkv_status} from {(client_ip, client_port)}")
            await g_objs.httpserver_manager.update_req_status(upkv_status)
    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"kv_move_status client {(client_ip, client_port)} has error {str(e)}")
        logger.exception(str(e))
    return


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)
    logger.info("Graceful shutdown completed.")
    return


@app.on_event("startup")
async def startup_event():
    logger.info("server start up")
    loop = asyncio.get_event_loop()
    g_objs.set_args(get_env_start_args())
    loop.create_task(g_objs.httpserver_manager.handle_loop())
    logger.info(f"server start up ok, loop use is {asyncio.get_event_loop()}")
    return
