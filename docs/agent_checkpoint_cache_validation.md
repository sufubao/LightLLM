# 精确前缀缓存：H100 验证记录

日期：2026-09-10。对照基线：upstream `1eb4810c`；候选为本 PR 的实现快照。所有 benchmark/eval 经 `exp -m` 执行，保留失败运行、源码快照、启动日志、原始请求与逐 token 概率。下面区分恢复正确性、冷算数值差异与性能，不能把它们合并成“全部测试通过”。

## 环境

- NVIDIA H100 80GB HBM3；driver 580.167.08；PyTorch 2.11.0+cu130。
- Qwen3.5-0.8B：普通模式、TP2/DP2 双微批、P/D 分离、`eagle_with_att` MTP step=3。
- Qwen3.5-27B：同一对 GPU 上依次运行 baseline/candidate，TP2/DP1，无 MTP，关闭 CUDA graph。
- 固定 GPU KV 容量 65536 tokens、running requests 32、请求上限 32768、chunk/batch token 预算 8192。基线小页 256、大页 256×32、小页状态槽 128。
- 27B 两边实际后端一致：FA3 prefill、FlashInfer decode、FlashQLA linear prefill、Triton linear decode，FlashInfer/SymmMem allreduce。
- 最终 27B CPU 预算 8192 MiB/TP rank；MTP 并发8与 CPU 淘汰专项使用 4096 MiB/rank。

早期 0.8B 普通对照曾发生 baseline FlashQLA 自动降级到 Triton，而 candidate 成功启用 FlashQLA；该组数据不用于归因性能收益。首次新 batch shape 的 JIT 编译也单独保留，不当作稳定服务性能。

## 正确性结果

| 路径 | 最终结果 |
|---|---|
| 任意输入边界 | 255/256/257/8191/8192/8193/12000 均精确命中；8193 最后一个输入 token 被调度为 decode 时仍保存输入终点 |
| 27B 保存后完整重放 | 47 组对照全部输出 IDs 一致；串行 logprob 逐位一致，并发最大绝对误差 0.000481 |
| MTP 保存后完整重放 | 20 组串行＋8 组并发，IDs/logprob 均逐位一致，覆盖 Agent、分支、EOS、token stop |
| 状态与 KV 实体 | 真实 accepted S@259、S@284 的 conv、SSM、全部 target KV，经目录保存及恢复后逐位一致 |
| 双微批 / 空 DP rank | 实际覆盖空侧与双非空 microbatch，60 请求无 HTTP 错误；12 组 seed→full-hit 的 IDs/logprob 逐位一致 |
| GPU 淘汰后 CPU 恢复 | 两轮各 10×8192 正常缓存请求施压；日志均证明 `GPU=0 CPU=6000`；完整恢复及追加17-token后重放逐位一致 |
| PD 输出回流 | 12000 输入生成16 tokens后发布 S@12015；P 下一轮恢复 `GPU=12000 CPU=15`，只发送一个尾页、复用一个基础页 |
| PD 首 token | P 完整 HEAD_ONLY、D 完整零 KV HEAD_ONLY、D 缓存 S@L 后输入 L+1 的本地 decode 均通过，首 token 不重复 |
| 内存 / 协议故障 | OOM admission、lease/flush/COW、乱序 PD 回调、混合开关降级、9种 TP 组合、随机 registry secret 与权限通过 |
| 无效捕获槽 | req0 token 索引故意填 INT_MAX，真实 GPU 选择/gather 对 0/1/3/4/7/8 候选全部安全，超4槽候选跳过 |

CPU 专项26请求、4组对照全部通过；追加17-token后的结果与冷算 IDs 相同，logprob 最大差 0.005179。

现有单元测试68项通过。`test_radix_cache.py::test_case10` 因测试未提供 mem_manager 在 `flush_cache()` 中失败；独立 upstream `1eb4810c` 源码复测同样失败，未修改这一既有测试。本 PR 不增加单元测试文件，新增的是可复用 HTTP 集成/benchmark 脚本。Black 与仓库配置的 flake8 检查通过；新增 Python 文件另通过完整 F/E9 检查，所有改动文件通过语法检查及 `git diff --check`。

## 验证中修复的问题

1. **不同计算历史的 KV 混用。** 同 tokens 分别经过 decode/prefill，KV 字节可能不同。旧候选恢复 S@284 时有 95776 个 target KV 元素来自另一历史，最大差 0.185547。加入 origins 后，这些 KV 和状态均与保存时逐位一致。CPU 页去重、GPU radix、P→D 和 D→P 都保留来源。
2. **8193 终点漏捕获。** 最后一个输入 token 被分到 decode；现在把该行同时识别为 prompt endpoint。
3. **HEAD_ONLY 输出缓冲生命周期。** 独立 pinned-buffer 命名空间避免覆盖尚待 CPU post 的普通批次。
4. **MTP 辅助状态。** 保存正确 conv 窗口/SSM row；私有 packed 尾槽重建并更新来源。
5. **分配失败及身份。** 可恢复 allocator OOM 仍参加 TP admission；失败拷贝 fence 后释放资源。身份包含有效量化配置、expert dtype 和实际权重格式；registry 使用独立随机 secret。

## 冷算差异不是零

“保存时现场续算”与“CPU 往返恢复后续算”在真实 S@259 的完整 logits 和最终 conv/SSM 上逐位一致。但把同一259-token前缀重新做完整 prefill，与此前 decode 形成的状态本来就不同：conv 最大差 0.125，SSM 最大差 0.006098。由这些状态续算时，greedy 首 token 可能不同。

因此保留了严格冷算对照的失败：最终27B的一个 Agent 续接输入，IDs一致而 logprob 差 0.030503，略高于预设0.03；DP/MTP另有不同 greedy token 的冷算比较。没有放宽阈值使脚本变绿。origins 修复保证数据来自同一计算历史，不能让不同算子、分块和 batch 形状天然浮点等价。

要求结果与完整冷 prefill 逐位相同的使用方，不能据本验证作出这种保证。

## 性能：完整命中有收益，并发有回退

27B 单请求、输出8 tokens、每输入长度3次 warm 请求的客户端中位值：

| 输入长度 | baseline TTFT | candidate TTFT | baseline TPOT | candidate TPOT |
|---:|---:|---:|---:|---:|
| 255 | 177.17 ms | 49.39 ms | 40.68 ms | 48.93 ms |
| 257 | 142.25 ms | 51.87 ms | 40.40 ms | 50.22 ms |
| 8193 | 129.15 ms | 59.43 ms | 39.83 ms | 48.37 ms |
| 12000 | 176.48 ms | 45.35 ms | 39.83 ms | 48.06 ms |

这些是短输出功能负载的观察值，样本少，并非饱和吞吐测试。完整命中减少 prefill，但同步快照发布会影响后续请求；连发中的较高尾延迟保留在原始数据中。

0.8B、MTP step3、8并发、预热后完整命中的均值：

| 指标 | baseline | candidate |
|---|---:|---:|
| TTFT | 83.43 ms | 100.87 ms |
| TPOT | 2.32 ms | 13.01 ms |
| 总延迟 | 99.96 ms | 192.09 ms |

因此本功能保持默认关闭，**不宣称全面性能提升**。CPU 同步搬运和发布仍需后续优化；尚未分离每项开销占比。已验证的 CPU 尾页复制微测中，Torch copy约1.19ms，NumPy约9.5ms，更换成 NumPy 反而更慢，未采用。

默认1GiB也放不下上述8路 MTP 的16个 prompt/output checkpoint：每个约130.64MiB（112MiB固定容量KV页＋状态/seed），合计约2.04GiB。扩大到4GiB后8/8命中，48请求/48对照通过；容量不足的失败记录没有删除。

## 复现与记录索引

```bash
exp -m "exact checkpoint HTTP regression" \
  python test/benchmark/agent_checkpoint_cache.py \
  --candidate-url http://127.0.0.1:PORT \
  --candidate-revision DEPLOYED_REVISION \
  --model-dir /models/Qwen3.5-27B \
  --output /path/to/new-run \
  --lengths 255,256,257,8191,8192,8193,12000 \
  --agent-input-len 12000 --max-new-tokens 8 \
  --repeats 3 --concurrency 8 --require-exact-hits
```

脚本同时保存 cold→seed/warm 与 seed→warm 两组比较。固定 `--run-id` 可复现相同输入；不同实验使用不同输出目录。MTP增加 `--require-mtp-activity`，同时保留实际 kernel/接受行验证。

关键远端 ledger ID 前缀：

| 内容 | ID |
|---|---|
| 27B baseline / 最终来源版 | `260910-004346` / `260910-013451` |
| 实际 KV 混用复现 / 来源修复字节断言 | `260910-011532` / `260910-012940` |
| 同一 accepted 状态的现场/CPU往返证明 | `260910-011845` |
| 最终 MTP 并发8 / 无效捕获槽 | `260910-013839` / `260910-014005` |
| 最终 PD 短/长/权限 | `260910-013249` / `260910-013344` / `260910-013444` |
| 最终双微批 / CPU-only | `260910-013149` / `260910-014148` |
| CPU复制微测 | `260910-014043` |

部署需统一更新使用新共享请求/PD结构的进程。测试按源码快照记录；最后的无效槽保护单独经过真实 GPU 验证，未将旧服务日志冒充为该行的部署验证。
