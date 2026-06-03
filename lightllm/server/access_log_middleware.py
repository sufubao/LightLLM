from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# ANSI colors keyed by the leading digit of the HTTP status code.
_ACCESS_LOG_STATUS_COLORS = {2: "\033[32m", 3: "\033[36m", 4: "\033[33m", 5: "\033[31m"}
_ACCESS_LOG_RESET = "\033[0m"


class AccessLogMiddleware:
    """ASGI middleware that emits a compact, optionally colorized access log line
    per HTTP request. It replaces gunicorn's ``--access-logfile`` so that access
    logs share the same formatting/handlers as the rest of the lightllm logs."""

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


def add_access_log_middleware(app):
    """Attach the access-log middleware to a FastAPI/Starlette app."""
    app.add_middleware(AccessLogMiddleware)
