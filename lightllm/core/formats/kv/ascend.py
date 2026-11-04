from .dense import DenseKVWriter


class AscendKVWriter(DenseKVWriter):
    def __call__(self, *, token_indices, kv):
        heads, dim = self.k.shape[1:]
        if kv.shape != (token_indices.numel(), 2 * heads, dim) or kv.dtype != self.k.dtype:
            raise ValueError("KV writer input does not match the bound Ascend format")
        if kv.device != self.k.device or token_indices.device != self.k.device:
            raise ValueError("KV writer indices and payload must reside on the bound device")
        indices = token_indices.long()
        self.k.index_copy_(0, indices, kv[:, :heads])
        self.v.index_copy_(0, indices, kv[:, heads:])
