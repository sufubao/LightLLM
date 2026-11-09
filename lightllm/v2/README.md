# LightLLM v2 execution runtime

`lightllm/v2/` 是一个与现有 LightLLM 路径并行的完整 execution runtime。它不修改 `lightllm/common`、`lightllm/models` 或 `lightllm/server` 的旧实现，调用者必须显式选择 v2 API。

## 运行入口

```python
from lightllm.v2.runtime import V2ExecutionSession

session = V2ExecutionSession.create(
    model_adapter=adapter,
    config=build_config,
    device=execution_context,
    request_capacity=max_requests,
    token_capacity=kv_capacity,
    page_size=page_size,
)
```

session 的生命周期固定为：

```text
prepare
  -> provider/planner 选择
  -> storage store 创建
  -> v2 AttentionStateManager 注册 native state
  -> storage allocation
  -> StateView / writer / runner bind
  -> execution
  -> close
```

## 目录职责

- `execution/contracts`：模型、设备、state、provider、transfer contract；
- `execution/formats`：KV、MLA、recurrent、weight 的物理格式；
- `execution/compute`：attention、linear、MoE backend；
- `execution/planner`：backend 选择、workspace 规划、绑定计划；
- `execution/storage`：GQA、量化 GQA、MLA、recurrent 和 window state store；
- `execution/runner`：batch metadata、graph、warmup、attention metadata runner；
- `execution/transfer`：transfer plan、completion 和 expert transfer；
- `attention_state`：v2 自己的 native state manager 和 allocation 生命周期；
- `context_parallel`：PCP slice assembly、DCP slot ownership、partial merge；
- `context_parallel_runtime.py`：显式 DP/context/TP process-group owner；
- `models/base`：只声明模型语义和 recipe，不直接创建 execution backend。

## 隔离保证

```text
lightllm/common/*       legacy runtime，保持 upstream/main 原样
lightllm/models/*       legacy model path，保持 upstream/main 原样
lightllm/server/*       legacy server path，保持 upstream/main 原样

lightllm/v2/*           v2 contracts + planner + providers + storage + runner
```

v2 内部不会导入 `lightllm.common.execution` 或 `lightllm.common.attention_state`；kernel 和基础 tensor 工具仍可复用旧实现，因为它们是底层设备依赖，不是 execution ownership 或状态契约。

PCP 和 DCP 已经作为 v2 contract 和可测试的 device-independent 运算接入，但不会隐式改变 launcher、默认 process group 或旧模型的 attention 行为。只有显式创建对应配置和 runtime group 后，调用方才会启用 context-parallel 路径。
