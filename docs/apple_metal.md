# Apple Metal：Qwen3.5-0.8B

通过标准 `lightllm.server.api_server` 启动。模型计算使用 PyTorch MPS（Apple
Metal），无 MLX 依赖。

## 安装

使用 Apple Silicon 原生 arm64 Python，建议 Python 3.12 或更新版本：

```bash
python3 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install -e '.[metal]'
```

Metal 依赖单独列在 `requirements-metal.txt`。CUDA 部署继续使用原有安装方式。
下载原始、未量化的模型权重：

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B
```

## 启动

```bash
python -m lightllm.server.api_server \
  --hardware_platform metal \
  --model_dir ./models/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 8000 \
  --tp 1 --dp 1 --data_type fp16 \
  --max_total_token_num 1024 \
  --max_req_total_len 512 \
  --running_max_req_size 4 \
  --batch_max_tokens 128 \
  --chunked_prefill_size 32 \
  --linear_att_hash_page_size 16 \
  --output_constraint_mode none
```

Metal 配置会选择纯文本模式、关闭 CUDA Graph，使用 FP32 recurrent state。
保持动态前缀缓存和 chunked prefill 开启；这是 Qwen3.5 混合注意力状态管理的必要条件。
`max_total_token_num` 必须显式设置，用于限制统一内存中的 KV 容量。增加请求数、
上下文长度和 `linear_att_cache_size` 都可能增加内存占用。

## 调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default_model_name",
    "messages": [{"role": "user", "content": "请从1数到5，用逗号分隔。"}],
    "temperature": 0,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

流式输出增加 `"stream": true`，接口沿用现有 OpenAI 兼容 API。

## 复用范围

- HTTP、tokenizer、router、`ChunkedPrefillQueue`、`ChunkedPrefillBackend`。
- `Qwen3_5TpPartModel`、原始权重加载、pre/transformer/post layer 推理。
- `Qwen3NextMemManager`、`KvCacheAllocator` 和请求到 KV 槽位索引。
- `ReqManagerForMamba` 的卷积/recurrent state、hybrid radix cache 的保存与恢复。
- 通用采样、SSE、结束判断、请求释放和断开连接处理。

Metal 实现提供设备运行时和算子，包括原生 causal prefill / paged decode attention、
GDN、卷积、归一化、RoPE、投影、KV 索引读写和 greedy 采样。
原生 kernel 使用 PyTorch 的 `torch.mps.compile_shader`；不支持的布局和参数保留
对应的 Torch 路径。没有独立的模型循环、KV 管理器或 server。

macOS 的共享内存名称和大小按系统限制处理；共享请求锁使用进程文件锁。
退出时仍由原 launcher 清理进程回收服务所属共享内存和锁文件。

## 当前范围

已接通 Qwen3.5-0.8B dense 原始权重的单设备纯文本推理。当前不支持量化、视觉、
MTP、PD 分离、多卡、CPU/disk KV offload、图捕获、微批次 overlap、结构化输出约束。
不支持的启动选项会提前报错。支持的 CPU→MPS 异步复制会持有私有 CPU 快照，
直到 GPU 完成复制；不支持的传输保留同步路径。
CUDA 数值算子保持原实现，本次 Mac 环境没有执行 CUDA 推理。

## 数值与服务验证

本机验证环境为 macOS 14.4 / arm64 / PyTorch 2.11.0，M3 Pro 14 GPU cores、18GiB。
原生投影使用 FP16 操作数和 FP32 累加；prefill attention 的 softmax 归一化使用 FP32，
PV 的概率拆为 FP16 主值及残差，以避免仅使用 FP16 概率引入的额外误差。

GPU 回归位于 `unit_tests/backends`，覆盖 CPU FP32 数值参考、分块尾部、偏移与跨步
视图、输入和缓存保持、调用方输出缓冲区、fallback 及异步传输生命周期：

```bash
python -m pip install pytest
python -m pytest -q unit_tests/backends
```

当前本机 408 项 GPU 回归通过。真实 HTTP 六组合、各三轮共 198 个性能请求完成，
输出与相同提示词的原基准逐字一致；基础任务 8/8、前缀缓存重复 3/3 一致，取消恢复通过。
当前版本另通过 8K 三个位置各冷/热一次的召回探针（6/6）；
这些探针用于检查状态与缓存正确性，不构成完整模型质量评测。

## 阶段性性能测量

2026-10-16 同机、同模型 FP16 流式 HTTP 对照，每组合三轮。输入 128 tokens、
输出 128 tokens；吞吐包括首字和排队，延迟为 p50。MLX 使用 queue128 接收队列，
推理参数保持不变；三个框架的六组合各 198 个性能请求全部成功。

| 指标 | LightLLM round25 | MLX | vLLM-Metal |
| --- | ---: | ---: | ---: |
| 单请求 tok/s | 62.54 | 62.15 | 59.29 |
| 2 并发总 tok/s | 114.94 | 112.68 | 112.25 |
| 4 并发总 tok/s | 206.34 | 208.15 | 199.36 |
| 8 并发总 tok/s | 315.91 | 350.07 | 323.31 |
| 16 并发总 tok/s | 468.78 | 385.95 | 364.27 |
| 单请求首字 ms | 138.25 | 96.14 | 74.01 |
| 单请求包间隔 ms | 15.19 | 15.28 | 16.10 |
| 16 并发包间隔 ms | 25.58 | 35.42 | 37.69 |
| 2K 输入、1 token 输出首字 ms | 956.77 | 699.84 | 708.72 |

分层异步提交和门控融合前，同样六组合的 round23 为单请求 47.06 tok/s、
包间 19.69ms、首字 159.12ms，16 并发 409.11 tok/s、包间 28.40ms，2K 首字
988.26ms。新版本在该实验中改善了单请求与并发吞吐，单请求吞吐已接近 MLX，
16 并发吞吐领先上述 MLX/vLLM 对照约 21%/29%；首字和 8 并发吞吐仍落后。
共享桌面存在 CPU 与交换活动，收益仍需通过完整矩阵和重复测量核实。新版完整
20 组合、三轮的 LightLLM、MLX、vLLM、llama.cpp、Ollama FP16 对照正在执行；
量化版本尚未刷新，不能据此宣布全面领先。

Metal 在普通 decode 的层边界提交非计时事件，使 GPU 执行与 CPU 编码重叠；
小批次每层、大批次每四层提交。该运行时钩子在 CUDA 为 `None`，保留原 CUDA 调用。
Prefill 的 GDN 门控融合为一个算子，仍使用 FP32 门控和原有 FP16 beta 舍入。

### 混合请求的后续优化

非整块投影支持 8..272 个 token，以原生 SIMD 矩阵的公开坐标信息屏蔽尾部写入，
直接写调用方的精确尺寸输出；完整块沿用原 shader。普通单 token dense attention
在混合批次中复用 paged decode，保留特殊控制和不支持输入的回退。

相同冷前缀的五轮流式 HTTP 干扰实验，在长响应输出第 8 个片段后插入短问题及 2K
提示词。以下均为五轮指标的中位数；“最长包间”先取每轮最长间隔再取中位数：

| 指标 ms | round21 mixed | round23 mixed | MLX queue128 |
| --- | ---: | ---: | ---: |
| 短请求首字 | 238.29 | 214.23 | 1041.13 |
| 2K 请求首字 | 1419.81 | 1247.10 | 1066.21 |
| 受干扰响应最长包间 | 170.20 | 151.01 | 729.16 |
| 受干扰响应包间 p95 | 25.44 | 23.81 | 19.03 |

每个版本 30 次尝试，29 次成功及 1 次主动取消；LightLLM 两版本的 15 个成对混合
请求输出逐字一致。round21 默认调度的最长包间中位数为 1019.69ms，但 2K 首字为
1080.92ms；当前 mixed 选项仍默认关闭。新优化在这一负载下改善了停顿和短请求，
2K 首字及 p95 包间仍落后 MLX。以上不替代全矩阵测量，也不构成全面领先结论。
