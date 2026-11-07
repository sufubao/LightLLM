# PCP 和 DCP

LightLLM 的上下文并行进程布局为 `DP × Context × TP`。`--tp` 仍表示进程总数，
每个 DP 副本中的权重 TP 度数为 `tp / (dp × context)`。启动时，
`--pcp_size` 控制 prefill context parallel，`--dcp_size` 控制 decode context
parallel；两者同时大于 1 时必须相等。

例如，4 卡、单 DP、2 路 PCP 和 2 路 TP 使用：

```bash
--tp 4 --dp 1 --pcp_size 2 --dcp_size 2 --disable_cudagraph
```

PCP 将展平的 prompt token 按连续范围分配给 context rank。每层写入本地 KV
后，context group 交换变长 KV 更新；各 rank 用完整请求 KV 做本地 query 的因果
attention，最后交换并按原始 token 顺序拼回 hidden。DCP 保留请求的逻辑 KV 位置；在
现有 allocator 仍为每个 rank 保留完整物理视图的兼容阶段，每个 context rank 只计算
自己拥有的 KV 区间，并交换局部输出和 log-sum-exp，使用
稳定的归一化合并得到完整 attention 输出。

当前执行路径只接受标准 dense GQA/MHA/MQA 和浮点 KV。它明确拒绝混合线性注意力、
量化 KV、MTP、MoE、CUDA Graph、microbatch overlap、prompt logprobs、DP prefill
balance 和 TPSP mix；这些路径需要各自的状态与图捕获协议，不能静默复用 PCP/DCP
的参考实现。

这条路径优先保证状态转移和结果正确性，使用显式 Torch attention 作为后端边界，
不代表生产吞吐或显存占用已经达到专用 fused kernel 的水平。单元验证覆盖 PCP KV
同步、输出还原、DCP LSE 合并，以及 2-rank Gloo process group；GPU、真实模型权重、
NCCL 多节点和性能结果仍需在目标部署环境单独验证。
