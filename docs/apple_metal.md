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

Metal 实现仅提供设备运行时和算子，包括 SDPA、GDN、归一化、RoPE、KV 索引
读写和采样辅助操作。没有独立的模型循环、KV 管理器或 server。

macOS 的共享内存名称和大小按系统限制处理；共享请求锁使用进程文件锁。
退出时仍由原 launcher 清理进程回收服务所属共享内存和锁文件。

## 当前范围

已接通 Qwen3.5-0.8B dense 原始权重的单设备纯文本推理。当前不支持量化、视觉、
MTP、PD 分离、多卡、CPU/disk KV offload、图捕获、微批次 overlap、结构化输出约束。
不支持的启动选项会提前报错。Metal 使用同步队列屏障保证正确性，尚未做吞吐优化。
CUDA 数值算子保持原实现，本次 Mac 环境没有执行 CUDA 推理。

实际运行记录：macOS 14.4 / arm64 / PyTorch 2.11.0，标准聊天接口返回
`1,2,3,4,5`；连续五次请求结果一致，后四次命中 16 个前缀 token；两个并发请求与带随机种子、惩罚参数
的 SSE 请求正常完成；长提示词完成多轮预填充，SSE 断开后请求槽位正常释放。
以上使用真实模型和 HTTP 请求，没有新增或运行测试套件。
