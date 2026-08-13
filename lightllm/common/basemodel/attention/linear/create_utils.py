import os

from lightllm.common.basemodel.attention.linear.gdn import (
    FlaLinearAttBackend,
    FlashQlaLinearAttBackend,
)
from lightllm.utils.backend_validator import validate
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

linear_att_backend_classes = {
    "flashqla": FlashQlaLinearAttBackend,
    "fla": FlaLinearAttBackend,
}


def get_linear_att_backend_class(model, priority_list=("flashqla", "fla")):
    if os.environ.get("FLA_FLASH_QLA", "1") == "0":
        priority_list = ("fla",)

    for backend_name in priority_list:
        backend_class = linear_att_backend_classes[backend_name]
        if backend_name == "fla":
            logger.info("Linear attention backend: fla.")
            return backend_class
        if validate(backend_name):
            logger.info(f"Linear attention backend: {backend_name} (validated).")
            return backend_class

    logger.warning("No linear attention backend validation succeeded, falling back to FLA.")
    return FlaLinearAttBackend
