from .attention import DenseStateStore


class AscendStateStore(DenseStateStore):
    def __init__(self, declarations, format, manager):
        super().__init__(declarations, format, manager, kv_type="None")
        self.initialized = set()

    def views(self):
        views = super().views()
        # FIA warmup can read the padding page at the maximum captured KV length.
        for view in views.values():
            for tensor in view.fields:
                identity = tensor.data_ptr(), tensor.numel()
                if identity not in self.initialized:
                    tensor.zero_()
                    self.initialized.add(identity)
        return views
