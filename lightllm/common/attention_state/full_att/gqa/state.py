from ..state import TokenAttentionState


class GQAStorageState(TokenAttentionState):
    @property
    def storage(self):
        return self.resource

    @property
    def k_storage(self):
        return self.storage.buffers.get("k")

    @k_storage.setter
    def k_storage(self, buffer):
        self.storage.buffers["k"] = buffer

    @property
    def v_storage(self):
        return self.storage.buffers.get("v")

    @v_storage.setter
    def v_storage(self, buffer):
        self.storage.buffers["v"] = buffer
