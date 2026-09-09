# 通过 PD master 采集多 P、多 D 指标

PD master 自动从注册表获取 P/D 节点的地址，调用方无需提供 P/D 的 IP、端口或节点 ID。

| 接口 | 返回内容 |
| --- | --- |
| `/metrics` | master 自身指标，行为不变 |
| `/pd/metrics` | 所有已注册 P/D 节点的指标 |
| `/pd/metrics?role=prefill` | 所有已注册 P 节点的指标 |
| `/pd/metrics?role=decode` | 所有已注册 D 节点的指标 |

```bash
curl 'http://MASTER:8000/pd/metrics'
```

新接口仅在 `pd_master` 模式可用，其他模式返回 404；非法 role 返回 422。
每次请求使用注册表快照，因此节点注册、移除会自动反映到下一次采集中。
master 必须能访问节点注册的 HTTP 地址。

## 指标语义与故障处理

各节点的指标解析后按指标族合并，保留原有样本值和标签，不对数值求和。
每个样本增加 `pd_role="prefill|decode"` 和 `pd_node="节点注册地址"`，例如：

```text
lightllm_num_running_reqs{pd_role="prefill",pd_node="10.0.0.1:8000",model_name="example"} 4
lightllm_num_running_reqs{pd_role="prefill",pd_node="10.0.0.2:8000",model_name="example"} 6
```

接口同时返回：

- `lightllm_pd_scrape_success{pd_role,pd_node}`：该节点采集成功为 1，失败为 0。
- `lightllm_pd_registered_nodes{pd_role}`：本次采集范围内各角色的注册节点数，无节点时为 0。

单次请求最多并发采集 16 个节点，每个节点的异步采集任务设有 5 秒超时（包含等待并发名额的时间）。
不跟随重定向，不缓存指标。网络错误、超时、HTTP 错误、指标解析失败、重复样本、
保留标签或指标类型冲突均会标记对应节点失败，不输出该节点的业务指标。
`pd_role`、`pd_node` 和上述监控指标名称由 master 保留。
节点指标过多时，解析和序列化仍会产生额外 CPU 耗时。

部分或全部节点失败时，接口仍返回 HTTP 200，以便 Prometheus 保存各节点的失败状态。
因此 `up` 只表示 master 汇集接口可采集；P/D 的采集告警应使用
`lightllm_pd_scrape_success == 0`。节点从注册表移除后不再输出其样本，
应结合注册节点数与部署期望副本数检查缺失节点。

## Prometheus 与 Grafana

只需要配置 master 的地址。以下配置同时采集 master 自身与所有 P/D：

```yaml
scrape_configs:
  - job_name: lightllm-master
    static_configs:
      - targets: ['MASTER:8000']
        labels:
          cluster: lightllm-pd
          pd_role: master

  - job_name: lightllm-pd
    metrics_path: /pd/metrics
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets: ['MASTER:8000']
        labels:
          cluster: lightllm-pd
```

Grafana 查询 Prometheus，按 `cluster`、`pd_role`、`pd_node` 筛选。
该配置下 `instance` 表示 master 采集入口，P/D 节点身份使用 `pd_node`。
例如 `sum by (pd_role) (lightllm_num_running_reqs{job="lightllm-pd"})`
查看各角色的运行请求数。
端到端请求量与延迟继续取 master 指标，避免将 master、P、D 的统计重复相加。

如需只采集 P 或 D，在 P/D job 中添加 `params: {role: [prefill]}` 或
`params: {role: [decode]}`。不要同时采集全量接口和角色筛选接口，避免重复统计。
同一组 P/D 只选择一个 master 采集入口；扩缩容不需要更新 Prometheus 的节点地址配置。
