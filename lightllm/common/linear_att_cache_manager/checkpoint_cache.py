"""Fixed-budget state storage used by both GPU- and CPU-KV prefix hits."""

from threading import RLock

import torch

from .checkpoints import CheckpointIndex, CheckpointPolicy
from .linear_att_buffer_manager import LinearAttCacheManager


class LinearAttCheckpointCache:
    def __init__(self, capacity, linear_config, policy: CheckpointPolicy):
        self.buffers = LinearAttCacheManager(size=capacity, linear_config=linear_config)
        self.index = CheckpointIndex(capacity)
        self.policy = policy
        # Forward and post-handle can run on different overlap threads.
        self.lock = RLock()

    def capture(self, req_manager, req, token_count, state_index=None):
        """Publish a canonical snapshot only after the device copy completes."""
        if token_count <= 0:
            return None
        tokens = req.get_input_token_ids()
        with self.lock:

            def write(slot):
                conv, ssm = req_manager.get_linear_att_state(req, state_index=state_index)
                dst_conv, dst_ssm = self.buffers.get_state_cache(slot)
                dst_conv.copy_(conv, non_blocking=True)
                dst_ssm.copy_(ssm, non_blocking=True)
                # Protect the immutable source and its metadata before another
                # thread or a following decode overwrites the working state.
                torch.cuda.current_stream().synchronize()

            return self.index.insert(tokens, token_count, write)

    def match(self, tokens, max_tokens):
        with self.lock:
            return self.index.match(tokens, max_tokens)

    def restore(self, tokens, max_tokens, req_manager, req):
        with self.lock:
            checkpoint = self.index.match(tokens, max_tokens)
            if checkpoint is None:
                return None
            conv, ssm = self.buffers.get_state_cache(checkpoint.slot)
            req_manager.restore_linear_att_state(req, conv, ssm)
            # A slot may be evicted immediately after leaving the lock.
            torch.cuda.current_stream().synchronize()
            return checkpoint

    def clear(self):
        with self.lock:
            self.index.clear()
