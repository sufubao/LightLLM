# H200 / Qwen3.5-27B：负载覆盖与正常入口验证

2026-10-13。固定原 PR `3af33c2f47f1738e17eb70a3c1e8f42e25ed65ed`、TP1、MTP4、BF16 SSM、L16、264 请求容量和 65536 KV token。所有输入加输出不超过 4096 token。没有通过增加资源容量或更改 MTP 配置获得收益。

本次包含 GPU2 的 12 组负载、72 轮交替配对（144 个 cohort），以及 GPU5 正常服务入口的独立整合验证。正常入口由 `python -m lightllm.server.api_server` 启动；候选额外启用 `--replayssm_projection_mode parallel --replayssm_fold_method derived`。

## 负载矩阵：同卡交替配对

每组、每臂预热两轮，再进行六轮正反顺序交替。greedy、关闭 thinking、禁用 prompt cache 查询、固定输出长度并 ignore_eos。相同 rep 使用相同提示。短输入使用四类技术解释模板及变化的示例编号；长输入使用固定 filler，不能代表所有生产内容。吞吐包含 prefill、请求提交和尾部完成，按总输出 token / 总 wall time 计算。95%区间为10000次配对 bootstrap，seed=417；没有删掉较慢轮次。

| 输入 / 输出 token | 并发 | 原 PR token/s | 候选 token/s | 增幅 | 95%区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| 短输入 / 384 | 1 | 140.8 | 170.4 | 21.01% | [18.85%, 23.11%] |
| 短输入 / 384 | 8 | 961.6 | 1059.4 | 10.17% | [5.97%, 14.51%] |
| 短输入 / 384 | 32 | 2551.0 | 2956.0 | 15.88% | [13.77%, 18.30%] |
| 短输入 / 384 | 52 | 2680.6 | 3144.7 | 17.31% | [15.00%, 19.74%] |
| 短输入 / 384 | 53 | 2437.2 | 3177.4 | 30.37% | [27.69%, 33.46%] |
| 短输入 / 384 | 64 | 3083.4 | 3669.0 | 18.99% | [17.51%, 20.16%] |
| 短输入 / 384 | 128 | 3655.0 | 4467.6 | 22.23% | [20.58%, 23.66%] |
| 短输入 / 128 | 32 | 2218.9 | 2465.0 | 11.09% | [5.53%, 14.76%] |
| 短输入 / 128 | 64 | 2585.1 | 2991.6 | 15.72% | [8.73%, 22.32%] |
| 短输入 / 128 | 128 | 3077.0 | 3639.8 | 18.29% | [13.34%, 22.48%] |
| 1024 / 384 | 32 | 1558.6 | 1721.5 | 10.46% | [8.53%, 12.00%] |
| 3072 / 384 | 16 | 705.8 | 741.3 | 5.04% | [3.40%, 7.16%] |

B128 / 输出384的主场景再次超过20%；其他负载不能套用这一数字。B52→B53跨过首轮draft的原图覆盖边界，组合收益从17.3%变为30.4%，与图扩展/accepted-tail投影机制一致，但该比较不是只切换一个组件的消融。输出缩短或长输入占比增加，端到端收益降低。

## 延迟

| 输入 / 输出 | 并发 | TTFT p50 原PR→候选 ms | TPOT p50 原PR→候选 ms |
| --- | ---: | ---: | ---: |
| 短 / 384 | 1 | 142.8 → 141.4 | 6.74 → 5.52 |
| 短 / 384 | 8 | 219.8 → 163.9 | 7.24 → 6.71 |
| 短 / 384 | 32 | 370.6 → 383.6 | 10.47 → 8.88 |
| 短 / 384 | 52 | 590.8 → 570.8 | 16.85 → 14.41 |
| 短 / 384 | 53 | 600.7 → 534.1 | 19.20 → 14.51 |
| 短 / 384 | 64 | 633.9 → 602.2 | 18.13 → 15.01 |
| 短 / 384 | 128 | 974.1 → 944.8 | 31.04 → 25.06 |
| 短 / 128 | 32 | 359.9 → 367.5 | 10.64 → 9.12 |
| 短 / 128 | 64 | 594.8 → 600.4 | 18.77 → 15.53 |
| 短 / 128 | 128 | 946.6 → 922.3 | 32.73 → 26.42 |
| 1024 / 384 | 32 | 2185.1 → 2218.5 | 13.15 → 11.31 |
| 3072 / 384 | 16 | 4070.9 → 4021.2 | 11.19 → 10.32 |

延迟是客户端观测值，按同一负载的请求合并取分位数；TPOT为每请求首尾token时间差除以后续token数。TTFT没有一致改善。逐轮数据保留在本机 `/tmp/replayssm-h200/coverage/`。

## 正常服务入口

正常入口两臂均预热两轮并正式运行八轮。GPU5原 PR **3667.2 token/s** → 工作树候选 **4446.3 token/s**，增幅 **21.25%**，独立bootstrap95%区间 **[18.96%, 23.66%]**。候选完成后停止服务，再在同一GPU5加载原PR；这是顺序独立运行，不是交替配对。区间反映轮次波动，不覆盖两次运行之间的系统漂移。GPU2的交替配对用于补充因果证据，不能混为同一个统计设计。
正常入口TTFT p50 原PR 913.2 → 候选 1031.2 ms；TPOT 31.21 → 24.88 ms。正常入口正式性能阶段没有其他模型客户端并发压测。

候选已经通过B1/B52/B53/B128的正常请求 smoke。HTTP 随机采样、top-k/top-p、非默认presence/frequency/repetition惩罚、EOS最短长度与无效token参数请求全部完成，logprob有限。Prefix checkpoint 两次实际命中1536 token，输出64 token与禁用查询的请求逐项一致。原始脚本中的 `cache_cold` 标签并非真正冷缓存：禁用查询的首次请求仍填充了缓存，以实际命中长度为准。

## 质量与数值权衡

同样的全量GSM8K1319题，greedy、关闭thinking、并发64、最大输出1536：正常入口候选 **1278/1319**，历史原 PR **1279/1319**；截断8/6。原PR独对3题、候选独对2题，879条完整token序列一致。准确率差 −0.0758个百分点，配对bootstrap95%区间 [−0.3791,+0.2274] 个百分点。这是与保存的历史质量运行比较，不是本次同卡交替质量实验，不能证明严格无损。此前实验入口 derived 为1281/1319；正常入口和实验入口1162条完整序列一致。

对4道有变化的题另做小并发诊断，1536上限1/4正确、截断2；3072上限2/4正确、截断1。原PR在同样小并发复测中，1536上限2/4正确、截断1，3072上限3/4正确、无截断。该诊断改变了并发，且是事后选择的困难题，只说明部分截断可以缓解；没有替换全量测试分数，也不能将差异全部归因于输出上限。

并行求解采用activation精度的Tensor Core投影；derived fold复用现有FP32 keys/deltas、以FP32矩阵累加合并历史。历史容量和精度未降低，但运算顺序与checkpoint舍入变化，不能保证与原PR逐位一致。默认仍为raw fold，derived显式启用；固定共享输入的长期测试覆盖 L16/8192 步及 L64/4096 步；这不等同于模型反馈下的质量保证。

## 保留的实现与清理范围

保留 parallel 求解、可选 derived fold、首轮 draft 图扩展与 accepted-tail 投影、LM head 布局、strided gate、融合 greedy、draft argmax 和默认 token 惩罚跳过。单独的 parallel 求解已测得约 11.4% 端到端提升；其余组件的组合进一步改善吞吐，不能把组合收益算作每个组件的独立收益。

删除批量 checkpoint、普通 decode prepare/accept 融合，以及早期特定形状的 H200 调参文件：前两项没有可靠的端到端收益，后者不用于已验证的 parallel 固定布局。恢复普通路径的 autotune 版本，保留 parallel/fold 配置隔离。旧实验原型、重复源码快照、日志、图表和过时报告移出仓库。

以上性能数字记录的是清理前的完整候选，不能冒充清理后的重新测量。删除的普通 decode 融合不用于 MTP4 verify；批量 checkpoint 对已测场景收益可忽略。清理后的验证针对状态、checkpoint、CUDA Graph、MTP 和采样回归。

## 清理后的验证

本地 H200 回归覆盖 676 个用例：首轮 652 通过，24 个 autotune 图隔离参数组合因旧测试桩缺少 `logits_row_stride` 而失败；补齐该字段后，autotune 与最终版采样的 74 项复测全部通过，没有未解决的失败。覆盖普通及 MTP 状态生命周期、checkpoint、GDN/KDA、CUDA Graph、parallel verify width 2/3/5/8/16、derived fold、模型输出裁剪和采样边界。计数包含复测重叠，不相加。

24 个修改/新增 Python 文件通过语法检查和仓库 flake8 规则，`git diff --check` 通过。10 月 13 日清理时未重跑端到端模型吞吐或 GSM8K；上述性能和质量数字均来自清理前实验。清理后的吞吐复测见下一节。日志位于本机 `/tmp/replayssm-h200/cleanup-regression.log` 与 `cleanup-recheck.log`。

## 2026-10-14：清理后吞吐复测

同一 H200 GPU5、Qwen3.5-27B、TP1、MTP4、BF16 SSM、L16、264 请求容量、65536 KV token。并发 128，每请求固定生成 384 token，沿用上一轮短输入技术解释模板，greedy、关闭 thinking、禁用 prompt cache 查询、ignore_eos。两边都通过正常 API 入口运行，各预热 2 轮、正式测量 12 轮。原 PR 仍为 `3af33c2f47f1738e17eb70a3c1e8f42e25ed65ed`；候选为清理后的代码快照，显式启用 `parallel + derived`。

| 指标 | 原 PR | 清理后候选 |
| --- | ---: | ---: |
| 端到端吞吐，token/s | 3663.12 | 4505.88 |
| TTFT p50，ms | 928.97 | 901.96 |
| TPOT p50，ms | 31.10 | 24.91 |
| 正式请求数 | 1536 | 1536 |
| 输出 token 数 | 589824 | 589824 |
| prompt cache 命中请求 | 0 | 0 |
| 性能窗最高温度，°C | 63 | 62 |
| 性能窗显存峰值，MiB | 104887 | 104139 |

**吞吐提升 23.01%，独立 bootstrap 95% 区间 [21.26%, 24.52%]**（10000 次重采样，seed=417）。按全部输出 token / 全部 cohort wall time 汇总，不删除慢轮次。原 PR 的单轮范围为 3572–3741 token/s，候选为 4285–4597 token/s。全部 1179648 个输出 token 的 logprob 有限，两边都没有 HW/SW 热降频。每 verify step 输出 token 数为 3.1833 / 3.1773，候选未通过提高接受率获得这一收益。

这是先基线、后候选的同卡顺序独立运行，**不是交替配对**。区间反映轮次波动，不覆盖两个运行阶段之间的系统漂移；本次 23.01% 与上轮 21.25% 的差异不能归因于代码清理本身。本次仅复测这一主场景，没有重新运行其他负载或 GSM8K。结果表明移除低收益路径后，该场景的吞吐提升仍保留。

两份源码均已冻结并保存 Python 文件哈希；候选快照与测试结束时的工作区一致。启动参数除代码路径、优化开关和端口外一致。候选首次因旧 NCCL 端口占用而未启动成功，改用空闲端口后完成全部测量；失败启动没有产生性能样本。测试服务均已停止。

原始逐轮响应、客户端、启动命令、源码哈希、硬件记录和统计脚本保存在本机 `/tmp/replayssm-h200/cleanup_retest_20261014/`；[紧凑统计结果](h200_retest_summary.json)保留逐轮吞吐与汇总值。

## 复现与原始记录

正常入口使用相同模型、MTP 配置和容量，候选额外加 `--replayssm_projection_mode parallel --replayssm_fold_method derived`：

```bash
LIGHTLLM_TRITON_AUTOTUNE_LEVEL=3 LOADWORKER=4 python -m lightllm.server.api_server \
  --model_dir /models/Qwen3.5-27B --tp 1 --host 127.0.0.1 --port 19131 --nccl_port 19132 \
  --disable_vision --disable_audio --ssm_state_mode replay --replayssm_cache_len 16 \
  --linear_att_ssm_data_type bfloat16 --llm_prefill_att_backend triton triton \
  --llm_decode_att_backend fa3 triton --graph_max_batch_size 1280 \
  --running_max_req_size 264 --max_req_total_len 4096 --max_total_token_num 65536 \
  --batch_max_tokens 4096 --chunked_prefill_size 512 --linear_att_hash_page_size 512 \
  --linear_att_page_block_num 2 --linear_att_cache_size 16 --mtp_mode eagle_with_att --mtp_step 4 \
  --replayssm_projection_mode parallel --replayssm_fold_method derived
```

通用 HTTP benchmark 保留在 `test/benchmark/kernels/benchmark_replayssm_service.py`，用于固定 1024-token 输入的跨版本对照；它的提示与上述短输入实验不同，不能直接复现同一吞吐数字。两臂必须使用相同参数和预热次数。

```bash
python test/benchmark/kernels/benchmark_replayssm_service.py \
  --model-dir /models/Qwen3.5-27B --url http://127.0.0.1:19131/generate_stream \
  --batch 32 --tokens 384 --warmups 2 --repeats 8 --output /tmp/replayssm-service.json
```

原始实验客户端、逐轮结果和启动命令在本机 `/tmp/replayssm-h200/{coverage,integrated,normal_baseline}/`。清理前代码及文档完整归档于 `/tmp/replayssm-h200/cleanup-before.tar.gz`。这些是本地实验记录，不是仓库依赖。

GPU2 矩阵性能窗最高 60°C，正常入口候选/原 PR 的 GPU5 最高 62/63°C；均无硬件或软件热降频。GPU2 部分测试与其他 GPU 的模型加载/质量测试共享主机。正常入口性能阶段没有其他模型客户端并发压测。正常入口显存峰值候选 104921 MiB、原 PR 104785 MiB；扩展图使候选多占约 136 MiB，容量配置相同不代表总显存完全相等。
