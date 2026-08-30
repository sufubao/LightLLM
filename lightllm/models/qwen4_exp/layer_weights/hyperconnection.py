from lightllm.common.basemodel.layer_weights.meta_weights import (
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import (
    BaseWeightTpl,
)


class Qwen4ExpGatedResidualWeight(BaseWeightTpl):
    """Checkpoint-compatible weights for one Qwen4 gated residual boundary."""

    def __init__(
        self,
        *,
        prefix: str,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        data_type,
        use_combine: bool = True,
    ) -> None:
        super().__init__(tp_rank=0, tp_world_size=1, data_type=data_type)
        hyper_hidden_size = hidden_size * hc_count
        merged_out_dim = hc_lowrank + hc_count
        # Padding rows are zero and discarded after the projection, so this
        # does not change the model. For Qwen3.8's wide K=10240 projection,
        # N=448 was elementwise invariant at all 96 residual boundaries for
        # both observed Hopper kernel transitions (M=4/12 and M=19/57).
        padded_out_dim = 1 << (merged_out_dim - 1).bit_length()
        if hyper_hidden_size >= 4096 and merged_out_dim <= 448:
            padded_out_dim = 448
        self.padding_size = padded_out_dim - merged_out_dim if use_combine else 0
        self.hc_norm = RMSNormWeight(
            dim=hyper_hidden_size,
            weight_name=f"{prefix}.hc_norm.weight",
            data_type=data_type,
        )
        self.input_mix_weight_down_block_inject = (
            ROWMMWeight(
                in_dim=hyper_hidden_size,
                out_dims=[hc_lowrank, hc_count]
                + ([self.padding_size] if self.padding_size else []),
                weight_names=[
                    f"{prefix}.input_mix_weight_down.weight",
                    f"{prefix}.block_inject_weight.weight",
                ]
                + ([f"{prefix}.__padding__"] if self.padding_size else []),
                data_type=data_type,
                tp_rank=0,
                tp_world_size=1,
            )
            if use_combine
            else None
        )
        if self.padding_size:
            padding_weight = self.input_mix_weight_down_block_inject.mm_param_list[-1]
            padding_weight.weight.zero_()
            padding_weight.load_ok[0] = True
        self.input_mix_weight_down = (
            ROWMMWeight(
                in_dim=hyper_hidden_size,
                out_dims=[hc_lowrank],
                weight_names=f"{prefix}.input_mix_weight_down.weight",
                data_type=data_type,
                tp_rank=0,
                tp_world_size=1,
            )
            if not use_combine
            else None
        )
        self.input_mix_weight_up = ROWMMWeight(
            in_dim=hc_lowrank,
            out_dims=[hyper_hidden_size],
            weight_names=f"{prefix}.input_mix_weight_up.weight",
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )

    def _create_weight(self):
        # Child meta-weights own their allocations.
        return

    def load_hf_weights(self, weights):
        self.hc_norm.load_hf_weights(weights)
        if self.input_mix_weight_down_block_inject is not None:
            self.input_mix_weight_down_block_inject.load_hf_weights(weights)
        else:
            self.input_mix_weight_down.load_hf_weights(weights)
        self.input_mix_weight_up.load_hf_weights(weights)

    def verify_load(self) -> bool:
        down = self.input_mix_weight_down_block_inject or self.input_mix_weight_down
        children = [self.hc_norm, down, self.input_mix_weight_up]
        return all(child.verify_load() for child in children)
