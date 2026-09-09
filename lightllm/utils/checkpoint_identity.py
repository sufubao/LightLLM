"""Deployment identities shared by checkpoint storage and PD transports."""

import hashlib
import json
from pathlib import Path

import yaml


def _model_identity(directory):
    path = Path(directory)
    with (path / "config.json").open() as source:
        config = json.load(source)
    # Follow load_hf_weights: safetensors take precedence, otherwise load .bin.
    # Immutable deployments preserve this metadata across P/D copies. This is
    # not a content checksum: replacing weights while preserving size/mtime
    # requires a new weight_version and restarting the affected services.
    files = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
    weights = []
    for file in files:
        metadata = file.stat()
        weights.append((file.name, metadata.st_size, metadata.st_mtime_ns))
    return dict(config=config, weights=weights)


def get_checkpoint_identity(args):
    """Return target fingerprint, auxiliary fingerprint and store namespace."""

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    quant_config = None
    if getattr(args, "quant_cfg", None) is not None:
        # Quantcfg reads YAML, including its JSON subset. Include parsed content
        # so equivalent configuration files can live at different paths.
        with Path(args.quant_cfg).open() as source:
            quant_config = yaml.safe_load(source)
    execution = dict(
        dtype=args.data_type,
        kv_type=args.llm_kv_type,
        quant_type=args.quant_type,
        quant_config=quant_config,
        expert_dtype=getattr(args, "expert_dtype", None),
        ssm_dtype=args.linear_att_ssm_data_type,
    )
    target = digest(
        dict(
            model=_model_identity(args.model_dir),
            version=args.weight_version,
            execution=execution,
        )
    )
    draft = ""
    if args.mtp_step:
        directories = args.mtp_draft_model_dir or [args.model_dir]
        if isinstance(directories, str):
            directories = [directories]
        draft = digest(
            dict(
                models=[_model_identity(directory) for directory in directories],
                mode=args.mtp_mode,
                version=args.weight_version,
                # init_mtp_draft_model inherits the target's dtype, quant_cfg,
                # quant_type and expert_dtype; they also identify draft state.
                execution=execution,
            )
        )
    return target, draft, f"{target}:{draft or 'target-only'}"
