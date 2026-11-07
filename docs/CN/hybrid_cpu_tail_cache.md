# Hybrid Attention 的 CPU 尾页复用

Hybrid attention 的 bounded state 以较小的 hash page 形成快照，而 CPU/disk cache 通常按多个 hash page 组成一个大页。上一轮 agent 对话结束在大页中间时，下一轮普通匹配只查询完整 CPU 页，历史碎尾页因此无法复用。

本改动在首个缺失的 CPU 页内，从长到短查找历史 hybrid hash：

- CPU 模式查询候选尾页；disk 模式还覆盖首个缺失磁盘块，避免只查询当前完整页而漏掉磁盘块末尾。
- 命中后更新请求的 cache manifest 和真实前缀长度，恢复流程仍走原有 CPU/disk 搬运路径。
- 保留超时边界；找不到尾页时继续原有完整页匹配，不改变请求正确性。
- `disable_prompt_cache` 和 `disable_linear_att_small_page_cpu_cache` 继续作为硬开关。

这样，agent 下一轮较长请求可以从上一轮生成前缀的历史碎尾页继续恢复，减少重复 prefill 和 bounded-state 重建。
