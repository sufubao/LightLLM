# Qwen3.8-Flash-Next v2

`Qwen38FlashNextModelAdapter` 将 Qwen3.8-Flash-Next 的模型语义注册到 v2 execution runtime：

- 按 `full_attention_interval` 验证周期性 GDN/full attention 布局；
- full-attention 层声明 packed MLA KV、indexer 以及 rotary position 辅助状态；
- linear-attention 层声明 Conv/SSM recurrent state；
- PLE 层声明请求级 lexical state，并交给同一个 v2 state manager 管理；
- MoE 层输出 v2 `MoeSpec`，供 planner 选择执行 provider。

模型适配器描述状态、格式、执行参数，并按 v2 recipe contract 延迟创建权重和层 recipe。调用方可以使用 `default_formats` 和 `requested_phases` 生成 `BuildConfig`，再通过 `V2ExecutionSession` 完成资源分配与绑定。

legacy `lightllm/models/qwen4_exp` 没有被复制到本目录。Qwen 专用的 lowering、组件和 kernel 都在本目录的 v2 namespace 中；它复用现有基础模型层和视觉组件作为低层依赖，不注册旧 model/server path。
