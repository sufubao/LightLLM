import argparse

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeight
from lightllm.common.basemodel.layer_weights.startup_weight_load import (
    CAPTURE_SAFE_WEIGHT_SENTINEL,
    WeightStorageManifest,
    initialize_capture_safe_weights,
    start_checkpoint_prefetch,
)
from lightllm.server.api_cli import add_cli_args


class DummyWeight(BaseWeight):
    def __init__(self):
        self.weight = torch.empty(8)

    def load_hf_weights(self, weights):
        self.weight.copy_(weights["weight"])

    def _create_weight(self):
        return

    def verify_load(self):
        return True


class DummyLayer:
    def __init__(self):
        self.projection = DummyWeight()


def test_capture_weights_can_be_committed_in_place():
    layer = DummyLayer()
    initialize_capture_safe_weights(layer, [], device_type="cpu")
    assert torch.all(layer.projection.weight == CAPTURE_SAFE_WEIGHT_SENTINEL)

    manifest = WeightStorageManifest.capture(layer, [], device_type="cpu")
    layer.projection.load_hf_weights({"weight": torch.arange(8)})

    assert manifest.changed_names(layer, []) == ()
    assert manifest.unchanged_floating_names(CAPTURE_SAFE_WEIGHT_SENTINEL) == ()


def test_storage_manifest_rejects_tensor_replacement():
    layer = DummyLayer()
    manifest = WeightStorageManifest.capture(layer, [], device_type="cpu")
    layer.projection.weight = layer.projection.weight.clone()

    assert manifest.changed_names(layer, []) == ("pre_post.projection.weight",)


def test_checkpoint_prefetch_reads_local_safetensors(tmp_path):
    for index in range(3):
        (tmp_path / f"model-{index}.safetensors").write_bytes(b"checkpoint" * 1024)

    handle = start_checkpoint_prefetch(
        weight_dir=str(tmp_path),
        num_threads=2,
        rank_in_node=0,
        node_world_size=1,
    )
    handle.wait(timeout=5)

    assert handle.done
    assert handle.errors == ()


def test_checkpoint_prefetch_requires_safetensors(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="safetensors"):
        start_checkpoint_prefetch(str(tmp_path), num_threads=1, rank_in_node=0, node_world_size=1)


def test_startup_weight_load_cli_defaults_to_serial():
    args = add_cli_args(argparse.ArgumentParser()).parse_args([])

    assert args.startup_weight_load_mode == "serial"
    assert args.weight_loader_prefetch_num_threads == 4


def test_startup_weight_load_cli_accepts_overlap():
    args = add_cli_args(argparse.ArgumentParser()).parse_args(
        ["--startup_weight_load_mode", "overlap", "--weight_loader_prefetch_num_threads", "8"]
    )

    assert args.startup_weight_load_mode == "overlap"
    assert args.weight_loader_prefetch_num_threads == 8
