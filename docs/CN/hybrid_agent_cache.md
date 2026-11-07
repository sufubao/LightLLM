# Hybrid Attention 的 Agent 输出前缀缓存

Agent 请求经常把上一轮的 assistant 输出和工具结果原样带入下一轮。对 hybrid attention 模型而言，原有缓存只覆盖输入 prompt；上一轮生成的 KV 和 bounded attention 状态在请求释放时会被丢弃，下一轮必须重新执行这段前缀。

本改动把普通 decode 已经执行完成的输出页发布到现有 GPU radix cache：

- full-attention KV 仍由请求页管理器持有；每个 hash page 额外保存一个 bounded attention 快照。
- 大页保存完整恢复点；同一大页区间内只复用一个尾部小页槽位，跨过大页时先完成大页快照再释放旧小页。
- 大页槽位申请失败时停止该请求后续输出缓存，不越过缺失的恢复点发布更长前缀。
- 缓存 key 只对已经执行的 token 计算；刚采样但还没有进入下一次 forward 的 token 不参与 key。
- speculative decoding 暂不发布输出快照，避免把未确认候选状态写入可复用缓存；输入 prompt 的既有缓存路径不变。

释放请求前会等待最近一次快照事件，确保异步 bounded-state copy 完成后才归还 KV 和状态页。下一轮 agent 请求仍通过原有 hybrid hash manifest 命中 GPU radix cache，并恢复对应的大页或小页状态。

验证覆盖 hash 边界、大小页切换、恢复点不足时的停止策略、生成 hash 与旧输入 hash 的区分，以及请求释放前的事件等待。该改动没有宣称跨重启、跨 TP 配置或真实模型吞吐收益；这些需要对应部署和基准环境验证。
