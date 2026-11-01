import torch

from lightllm.common.attention_state.kernels.copy import copy_bounded_att_tensors
from lightllm.common.attention_state.transfer.views import bounded_att_buffer_copies


class SnapshotTransfer:
    @staticmethod
    def pages(fields, page_buffers, page_indices, layout, write):
        assert len(fields) == len(page_indices), "CPU page and slot counts differ"
        for copies in bounded_att_buffer_copies(
            page_fields=fields, page_buffers=page_buffers, layout=layout, page_indices=page_indices, write=write
        ):
            copy_bounded_att_tensors(copies=copies)

    @staticmethod
    def copy_fields(sources, destinations):
        assert len(sources) == len(destinations), "source and destination counts differ"
        if not sources:
            return
        names = tuple(sources[0])
        assert all(tuple(fields) == names for fields in (*sources, *destinations)), "snapshot fields differ"
        for name in names:
            copy_bounded_att_tensors(
                copies=[(src[name].tensor, dst[name].tensor) for src, dst in zip(sources, destinations)]
            )

    @staticmethod
    def external_views(page_buffers, page_index, layout, write):
        page = page_buffers[page_index].view(torch.uint8).view(-1)
        for region in layout.regions:
            if not write or region.write_enabled:
                yield page[region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
