# GLM-5.3-Flash H100/H200 部署与验证

该配置使用 TP8、BF16 KV cache、32K batch token、1.12M token cache 和
CUDA Graph 128，面向 16K 输入、256 输出的高并发文本服务。

## 镜像

私有仓库标签：

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:v1.5.0-glm53-16k256-kpool-c3f39a82
```

不可变镜像：

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:76a968231ffb0a4c8996bc9a0fb981b88b3afbf25b7ad77a07fa07fccc04901d
```

本机镜像为 `lightllm-glm53:16k256-kpool`，镜像 ID 为
`sha256:d388173fbbc1fc2e8961785248331cecd08683af3c5f28ff67bc530d770d1409`。
同一不可变镜像已拉取到 H100 节点。

## H100 部署命令

镜像已内置服务参数；启动时显式开启 K-pool decode 快路径：

```bash
IMAGE="registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:76a968231ffb0a4c8996bc9a0fb981b88b3afbf25b7ad77a07fa07fccc04901d"

sudo docker pull "$IMAGE"
sudo docker run -d \
  --name glm53-lightllm-16k256-kpool \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --gpus all \
  -e LIGHTLLM_ENABLE_KPOOL_DECODE_FASTPATH=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576:1048576 \
  -v /home/devsft/models/GLM-5.3-Flash:/model:ro \
  -v /home/devsft/cache-glm53-lightllm:/root/.cache \
  -v /home/devsft/cache-glm53-triton:/root/.triton \
  -v /home/devsft/cache-glm53-deep-gemm:/root/.deep_gemm \
  "$IMAGE"
```

接口为 `http://127.0.0.1:8002/v1`。该快路径要求请求 prompt 长度按
16 token 对齐，并使用镜像内置的关闭 prompt cache、关闭 MTP 配置；其他负载可移除
`LIGHTLLM_ENABLE_KPOOL_DECODE_FASTPATH`，使用保守路径。

本机 H200 可直接运行：

```bash
LIGHTLLM_GLM53_MODEL_DIR=/nvme/sufubao/models/GLM-5.3-Flash \
LIGHTLLM_GLM53_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-lightllm-h200 \
LIGHTLLM_GLM53_TRITON_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-triton-h200 \
LIGHTLLM_GLM53_DEEP_GEMM_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-deep-gemm-h200 \
tools/run_glm53_h100_container.sh
```

## 16K/256 性能

LightLLM 与 vLLM 在同一台空闲的 8×H100 80GB 节点顺序测试，使用同一 FP8
checkpoint、TP8、BF16 KV cache、32K batch token、关闭 prompt/prefix cache、
seed 42、temperature 0、无限请求速率和一次不计入结果的 warmup。每档请求数等于
并发数，每个请求输入 16,384 token、输出 256 token。

| 并发 | LightLLM 总吞吐 tok/s | vLLM 总吞吐 tok/s | LightLLM / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 5,373.86 | 6,542.89 | 82.13% |
| 8 | 16,444.11 | 17,635.33 | 93.25% |
| 16 | 20,669.61 | 25,246.99 | 81.87% |
| 64 | 25,614.05 | 31,485.12 | 81.35% |
| 128 | 25,588.22 | 27,988.09 | 91.43% |
| 256 | 25,623.93 | 31,206.94 | 82.11% |

结果文件位于
`/nvme/sufubao/m39-home/results/glm53_h100_16k_256_kpool_decode_ab/` 和
`/nvme/sufubao/m39-home/results/glm53_h100_16k_256_vllm_opt/`；所有启动、性能与
精度实验均由 `exp` 归档到 `~/experiments/runs/`。

## 精度

| 验证 | 结果 |
| --- | --- |
| GSM8K 固定前 100 题、5-shot、greedy | **99/100**；100/100 完成；无截断 |
| 精确 16,384-token needle | **PASS**；API 与本地 tokenizer 均计数 16,384；找回 `ZEBRA-4821` |

相关实验记录为 `260901-020939` 和 `260901-021013`。
