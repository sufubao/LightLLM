# 面向 Agent 的精确前缀检查点缓存设计

状态：已实现第一版，默认关闭。本文第 1–10 节保留目标架构与后续设计，**不表示每项建议均已落地**。以下实现范围及限制优先于后文。原代码基线：`cache-optim @ d7c6ef10`（PR #1558）。

## 本次实现

```bash
--enable_exact_prefix_cache \
--exact_prefix_cache_mb 1024 \
--exact_prefix_cache_entries 128 \
--exact_prefix_cache_page_size 8192 \
--exact_prefix_cache_capture_slots 4
```

新模式使用普通 GPU token radix 和独立 CPU 检查点目录，不与旧 `--enable_cpu_cache` / `--enable_disk_cache` 同开。旧小页/大页参数不再决定新检查点的长度，也不再额外切分 prefill。关闭新开关恢复旧路径。

| 已实现 | 代码入口 |
|---|---|
| 压缩 token 前缀目录、任意精确终点、完整 CPU KV 覆盖、独立 conv/SSM/seed | `dynamic_prompt/checkpoint_cache.py` |
| 固定容量纯 KV 页、同计算来源整页共享、不可变尾页 COW、LRU、lease、flush generation | 同上 |
| batch 长度绑定、GPU 带 mask 的有界捕获、CPU stop 最终判定、全 TP admission | `model_infer/exact_prefix_cache.py` |
| CPU 恢复仅加载 GPU radix 缺失区间，GPU KV 淘汰不删除 CPU 状态目录 | 同上 |
| 双 infer 线程各自 ticket；microbatch overlap 使用四组 staging；伙伴计算时跳过过时的空闲休眠 | chunked prefill / DP backend |
| 独立 hidden 副本、批量 scheduler HEAD_ONLY、当前请求重新采样、独立 pinned 输出缓冲 | `basemodel.py` / `base_backend.py` |
| 精确 MTP conv 窗口与 SSM row、canonical 恢复、Qwen draft 私有尾槽重建 | `linear_att.py` / `proposers/exact_resume.py` |
| D→P CPU 缺页传输、全 TP 发布、首 token owner、D 完整命中零 KV 控制任务 | `pd/checkpoint_transport.py` / PD master |
| 原生 HTTP 边界、Agent、分支、stop、并发回归脚本 | `test/benchmark/agent_checkpoint_cache.py` |

`exact_prefix_cache_mb` 是**每 TP rank 的缓存数据预算**，包括 KV 页物理容量、状态/seed、未发布数据及尚有 lease 的已淘汰数据。Python 元数据、HTTP 序列化、TP 转换和网络队列的临时内存另计；条目数、捕获槽及传输队列另有上限。GPU 自动容量 profiling 预留 staging 和一页 gather 临时空间。

启用时在启动阶段一次性预留该预算的 pinned CPU arena，请求处理中只划分和回收窗口，避免 `cudaHostAlloc` 停顿。每个窗口具有独立 tensor storage，序列化一页不会携带整座 arena；最后一个 tensor/view 释放后才归还窗口。finalizer 只向队列投递回收记录，分配时在锁内合并空闲区，避免 GC 重入锁或修改正在遍历的空闲表。已关闭 lease 和已消费的 pending 不再持有缓存实体。

捕获所需 CPU flags 按 staging slot 使用独立 pinned buffer，一次异步上传；已知输出长度且忽略 EOS、没有 token stop 时，直接使用 prompt/终止位置 mask。完整命中的 LM head、当前采样和 MTP 私有尾槽修复按批次执行，再统一等待。普通模式固定 MTP 的首次 decode 只验证有效 target 行，随后由正常 proposer 产生下一轮候选，避免验证尚未初始化的 draft 位置。启用该模式时，目标模型的 GDN、FA3 FP/MLA 始终使用可变行布局，保持 CUDA Graph 捕获和重放一致，SSM 的每请求物理槽跨度不变。捕获选择和请求起始位置 kernel 将实际行数作为非特化运行时参数，仅保留 block 大小的编译版本，避免新旧请求混合后按每个精确行数重新编译。启动时用空 mask 预热本 rank 容量范围内的有限 block 版本；不读写请求状态和 KV。offload 的最后一页 KV 与 state/seed 共用完成等待，多页之间仍释放上一页 gather 临时空间；异常路径同样等待已提交的读写，再归还窗口。

普通模式在 CPU post 决定接受/停止位置后冻结 tokens、origins 和状态视图，交给独立 CUDA stream 的后台线程写入 CPU 缓存。任务数量由 staging 槽约束；在全 TP 都完成复制之前，请求保留 GPU KV 和索引，不能释放或暂停，但可以继续计算后缀。worker 使用独立 Gloo 组，只准备尚不可见的数据；调度阶段取完成 epoch 的全 TP 交集后统一发布，再释放引用和 staging，并允许 worker 处理下一任务。目录发布与恢复因此保持串行，避免相同 tokens/长度在不同 TP rank 命中不同版本。clear 取消旧 generation 的发布，不提前释放在途资源。两个 infer 线程仍遵守原有全部握手，保证 post 发布登记与下一次分类轮询的顺序。若全 TP 的目录已经有同一精确前缀及所需 seed，worker 持 lease 共同确认后沿用既有 KV/state/seed/origins，跳过重复搬运；任一 rank 缺失仍执行原有准备流程。复用也经过完成队列和引用释放，不修改当前请求的计算来源。PD 保留原有同步发布和传输协议。

普通模式的 `_pre_post_handle` 只推进实际接受行；当输出长度已达 `max_new_tokens` 时，下一次分类不再安排多余 forward，仍保留原有握手和正常 post/finish/free 顺序。

捕获自然 prefill chunk 终点，以及 GPU token/EOS/长度提示选出的停止候选；CPU post 再核验实际接受与 stop 边界。普通 decode 生成 N 个 token，默认最多保存 `prompt_len + N - 1` 的状态：最后刚采样的 token 尚未计算。迟到的外部 abort 或字符串停止不保证留下精确末态，只保留此前合法检查点。

检查点额外保存每个 KV token 的计算来源 `origins`。同 token 前缀分别通过 decode/prefill 计算，可能留下不同的浮点 KV；不能把一份状态和另一条计算历史的 KV 混用。CPU 目录先验证实际 tokens，GPU radix 再按 origins 查找；CPU 页键同时包含 tokens 与 origins，COW 只复制两者都一致的前缀。每批新计算区间产生新的跨 TP 一致来源，恢复/PD 传输继承已有来源；普通 decode 不逐步复制整条前缀来源向量。

MTP CPU 页键还记录 successor 或终端标记，因为 draft slot i 可能依赖 token[i+1]。恢复末槽到请求私有 KV，再用 H@L 和本次 token[L] 重建 draft 尾部，并为这个 packed 槽赋新来源；捕获还冻结 packed 尾槽，防止下一批 proposal 提前改写它。DP 的单批次 draft 修复不能重放双微批 CUDA Graph，采用已有的普通执行路径；正常 DP 双微批仍使用图重放。

### 当前限制

- **CPU onload 和 PD 发布仍有同步屏障，CPU cache 锁会跨页拷贝等待。** 普通模式 offload 在后台完成，活动 decode 不等待该复制；新请求的目录查找仍可能等待 worker 持有的锁。尚未实现 `LOAD_WAIT`、HEAD_ONLY 与 onload 并行，不能称为全异步 CPU cache。
- 尾页采用 COW，没有原地追加、fragment arena；许多短分支可能浪费 CPU 容量和复制带宽。checkpoint 选择为 LRU/有界候选，没有成本模型或每会话优先级。
- namespace 覆盖模型配置、实际加载的 safetensors（没有时为 `.bin`）文件名/大小/mtime、dtype、解析后的量化配置、expert dtype 及 draft 配置，不包含模型或量化文件路径。部署期间权重必须不可变，跨节点复制需保留权重元信息；这里不扫描权重内容计算摘要，若替换权重却保留大小和 mtime，必须更新 `weight_version` 并重启服务。启动拒绝在线 RL 更新。多模态、请求 prompt logprobs/routed experts 回退重算。
- 完整命中要求模型具有末位置 hidden adapter。MTP 仅支持带 adapter 的单层 Qwen3.5 `vanilla_with_att` / `eagle_with_att`；其他 proposer 不使用新检查点。启动拒绝 MTP+expert parallelism、diverse mode、legacy DP cache fetch 组合。
- 没有 seal-only forward、严格字符串 stop 历史环或后台末态重建；也没有 disk 持久化。
- D→P 使用 CPU HTTP 传输与 canonical head 布局转换，尚非 RDMA/共享内存。相同 MTP layout 可传输；D MTP→P target-only 可剥离 draft 并重建页键，反方向拒绝。P→D 同时传递每段 KV 的计算来源，D→P 缺页协商保留这些来源。D 暂只复用完整输入或差一个 token 的检查点；其他部分命中放弃，接收 P 的全部输入 KV 和状态，避免混合两种计算历史。跨节点回流和 P 基页去重要求两端开启新模式；仅 D 开启时为收到的 KV 分配私有来源，仍可在 D 本地缓存。下一轮路由不保证返回原 P；owner 亲和及 D 短后缀 prefill 留待后续。
- 浮点 decode 状态续 prefill 与完整冷 prefill 的计算路径可能不同，甚至使接近的 greedy 候选改变顺序。来源隔离保证 KV/state 配套，不保证不同算子/分块/批次形状之间逐位相同。验证分别检查保存时运行态的恢复等价性与冷算差异，不将后者直接计为数据传输损坏。

性能实验必须用 `exp -m`，记录实际 attention backend、代码/补丁、GPU、TP/DP 和容量参数。自动后端降级、首次 JIT 和客户端输出长度都可能混淆对比；命中率不是吞吐结论。

目标是在 LightLLM 的 token KV 寻址、CPU 页搬运、PD 分离和 CPU/GPU 折叠流水线上，支持任意已计算长度的混合模型检查点。正确性必须无条件成立；命中率与额外开销有明确策略，不承诺所有结束条件都能零代价保存精确状态。

## 1. 核心决策

采用 **精确前缀目录 + 独立状态存储 + 固定容量 KV 页 + 异步捕获事务 + 显式恢复计划**。

- KV 搬运页大小与状态检查点长度解耦；8192 是物理容量/传输组织参数，不是状态对齐条件。
- 只在选定位置保存检查点，不保存每个 token 的历史状态；保存位置可以是任意整数长度。
- 检查点绑定同一实际 token 前缀的 KV、线性状态、可选输出信息和辅助模型恢复信息。
- 捕获、CPU 停止判定、跨 rank 就绪、发布、淘汰是不同阶段。
- GPU/CPU 状态缓存不再由 GPU radix 节点的生死直接决定。
- 完整命中使用显式 `HEAD_ONLY` 执行；不伪装成 0-token prefill。
- D 输出必须显式发布到下一轮可访问的 cache owner；会话亲和不能代替正确的前缀验证。

“任意长度”表示：在长度 L 实际得到的合法状态可以存储、索引和恢复。它不表示能从 S@12000 倒推出未保存过的 S@11000。

## 2. 当前流程中不可忽略的约束

| 当前实现 | 对新设计的约束 |
|---|---|
| `InferReq.linear_att_cache_len` 来自输入 hash 数；chunk 在大页及输入尾部边界停下 | 改为独立捕获策略，默认不为哈希尾边界额外拆 chunk |
| `_pre_post_handle()` 先推进请求长度，`notify_forward()` 后才执行 CPU `_post_handle()` | 不能在 finished 回调里读取“当前 req 长度和当前 state”当作上一批状态 |
| CUDA graph 输出缓冲会重用 | Python tensor 引用不能冻结 hidden/state |
| MTP SSM 有多个候选槽，conv 有接受位置对应的滑动窗口 | 要按精确接受行导出，不可直接套用 prefill 的 canonical 槽 |
| Qwen3.5 MTP 会原地归一化 target collector hidden | 输出信息需要独立捕获，且必须声明 hidden 的具体格式 |
| 部分 proposer 的 draft KV 使用左移输入，尾部包含本次采样 token | target 与 draft 的可复用边界可能不同 |
| CPU onload 目前有逐请求 synchronize、rank barrier，FA3 有额外保护 | 全异步是后续工程目标，不能简单删同步 |
| D backend 禁用 CPU cache | D→cache-owner 是新增路径，不是打开一个现成开关 |
| PD 首 token 协议依赖 `input_len - 1` 和最后一个 KV task | 精确完整命中需要独立控制消息和明确的首 token 产生方 |
| tool arguments、reasoning 和 chat template 会重新渲染 | 新请求必须按实际 token 前缀验证，不能仅凭 session ID 恢复 |

## 3. 数据模型与正确性不变量

### 3.1 前缀身份

逻辑身份为：

```text
PrefixKey = namespace + execution_fingerprint + token_count + prefix_digest
```

`execution_fingerprint` 需要覆盖会改变被复用计算结果的模型版本、adapter、位置/模型配置、KV/state 格式，以及多模态 embedding 身份等。采样 temperature 不属于 target KV 的身份；draft 对采样 token 的依赖由辅助恢复描述单独表达。

使用服务端确定的 namespace；客户端 continuation/session ID 只能作为查找提示。不可根据一个不经验证的客户端 handle 直接加载其他请求的内容。

### 3.2 检查点记录

```text
CheckpointDescriptor
  id, generation, namespace, execution_fingerprint
  prefix_handle, exact_len, prefix_digest
  state_handle + state_layout_version
  kv_manifest_handle
  output_seed_handle? + seed_format
  auxiliary_resume_descriptor?
  availability_by_location
  retention_metadata
```

`prefix_handle` 引用不可变 token 序列/压缩前缀树；manifest 使用结构共享的范围链，而不是为每个检查点复制整条 token 索引表。

物理句柄必须带 generation，例如 `(pool_id, slot_id, generation)`，防止 slot 复用后旧异步回调或目录指向新数据。进程内指针不能作为跨进程/跨节点句柄。

### 3.3 五条不变量

1. `S@L`、`KV[0,L)`、`OutputSeed@L` 必须属于同一个实际执行前缀；不能用较晚状态冒充较早截断位置。
2. 候选命中 L 必须存在完整 KV 覆盖 `[0,L)`。各层/各 TP 分片都要满足；检查点目录存在不等于数据仍完整。
3. 完整输入命中 `L == input_len`，除 KV/state 外还需要足够的输出恢复信息；否则选更早检查点。
4. READY 只表示相应位置的读路径已满足依赖。某副本正在 offload，不得被宣称为 CPU READY；可独立保留已可用的 GPU 路径。
5. 发布后的可见字节不可覆盖；所有异步读写结束前，源/目标资源不得被回收。

### 3.4 三个不同的长度

显式记录 `computed_len`、`verified_len`、`visible_end`。普通采样刚生成的最后一个 token 还没有 KV/state；MTP 已验证的一串 token 又可能在中间因 EOS/stop 截断。

检查点选择来自 batch 的精确行与状态选择器，不能仅依赖已预推进的 `req.cur_kv_len`。恢复下一轮前，再用新请求实际 token LCP 确认该检查点是否可用。

用 0-based token 下标说明 off-by-one：本批开始已计算 B 个 token，请求内第 r 个 verify 输入行处理 `token[B+r]`，产生 `S@(B+r+1)` 和该位置 hidden；该行采样的是下一位置 `token[B+r+1]`。刚采样出来的 token 不因“已输出”而自动拥有 KV/state。

普通 decode 生成 N 个 token 时，若没有额外超前计算，最新可保存状态是 `S@(prompt_len+N-1)`。下一轮从这里补算最后一个 token 即可。若业务确实要求包含它的状态，可在后台预算内做一次不向用户继续采样的 seal-only forward；默认不为省下一轮一个 token 强制增加本轮计算。

## 4. 索引、KV 页和状态存储

### 4.1 逻辑前缀索引独立存在

保留现有普通 token radix 的按 token 比较/拆边能力，另建不由 GPU KV 淘汰销毁的 checkpoint 前缀目录。不能仅在现有 GPU `TreeNode` 上挂字段，因为它的生命周期、ref 和 split 仍由 GPU KV 管理。

目录使用压缩边，不为每个 token 建 Python 节点；检查点可挂在任意边界。查找是一次前缀遍历加沿途候选检查，不扫描整个状态池、也不为全池每一种长度重复 hash 整个输入。

现有固定粒度 hash 可以作为前缀块加速索引。精确检查点的 key 可以通过增量 prefix digest 或复用分块 hash 加尾段计算；hash 页不再是可保存长度的限制。插入/匹配需遵守同一格式，不能混用不兼容 hash 方案。

元数据目录按 cache owner 分片；GPU 本地只做自己的快速匹配和租约获取。不要在各推理进程各自维护无法一致回收的“全局” Python 树。

### 4.2 固定容量 CPU KV 页与任意有效范围

CPU 页只存 KV，暂以 8192 token 为物理容量。状态不嵌在 KV 页内。每个 manifest 的页视图保存：

```text
page_handle, logical_token_start, physical_offset, valid_token_count
```

完整页可以继续使用现有前缀 hash 进行去重；尾页通过 checkpoint 的精确前缀身份关联，不能仅用更长完整页的 hash 来寻找较短历史尾部。

同一分支由 12000 延伸到 12317 时：

- 若获得尾页唯一追加写租约，且新增 token 与已存在内容无冲突，可只写未提交区域。旧检查点仍使用固定的旧 `valid_token_count`。
- 所有 rank 新增区域写完后，再发布更长页视图；旧读者不观察未提交区域。
- 若另一个分支写入不同 token，使用新尾页/写时复制，禁止覆盖旧视图可见字节。
- 空页初始化、padding 和复用也不能触碰正在被引用的有效区域。

首版允许尾页 COW，记录其字节开销；不要一开始引入复杂子页压缩。若 Agent 短分支造成大量尾页复制或内部空洞，再以测量结果决定是否增加 fragment arena。

页大小独立调节，不绑 snapshot 数量，也不绑 `chunked_prefill_size`。不同大小影响元数据数量、尾页浪费和搬运效率，不能先验保证 8192 对所有负载最佳。

### 4.3 状态池统一管理

使用一个逻辑 StateStore，取消大小页两个配额。其物理实现包含长期 CPU 锁页槽，以及有界 GPU capture staging；二者预算、用途和指标必须显式区分。

CPU 状态已在相同可长期持有的状态存储时，提交只做所有权转移，不再拷贝进 CPU KV 页。不同 owner 的状态仍需显式搬运。

CPU 共享内存需要使用现有 `CpuCacheCreator` 一类的显式映射/注册机制，不能把普通进程私有 pinned tensor 的 Python 引用当作跨服务共享。TP layout 要有版本，跨 TP 配置采用 canonical 全局 head 布局或显式转换。

### 4.4 查找输出是一个恢复计划

```text
ResumePlan
  checkpoint_id + exact_len
  target_kv_gpu_refs
  missing_kv_spans_by_owner
  state_source
  output_seed_source?
  auxiliary_restore_plan
  leases
  execution_kind = PREFILL_SUFFIX | HEAD_ONLY
```

先用 token LCP 限制候选，再验证 KV 覆盖和实际 capability。候选需获取租约，generation 和 READY 二次验证失败时释放并回退。

最长命中不一定最快。冷 CPU 的更长检查点与 GPU 上较短检查点，比较“加载缺口 + 状态恢复 + 剩余 prefill”的预计成本；无可靠测量时使用保守阈值，不引入无法解释的复杂评分。

## 5. 捕获事务与流水线折叠

### 5.1 新增本批不可变 CaptureTicket

```text
CaptureTicket
  request_id + request_slot_generation
  batch_epoch + microbatch_id
  token_prefix_handle + exact_candidate_lengths
  state_selectors / accepted_rows
  owned_state_staging + owned_output_seed
  KV source leases
  capture_event, copy_events, rank_completion
  decision = PENDING | RETAIN | DISCARD
```

生命周期：`RESERVED → FROZEN → TRANSFERRING → READY`，任何中间阶段可进入取消流程。取消是“不再发布”，不是立即释放仍被 DMA/kernel 使用的内存。

### 5.2 时序

```text
计算流，批次 t
  prepare：固定本批 token 范围和状态行
  target forward
  在会被 graph replay / draft norm 覆盖之前捕获所需 output seed
  sample / verify
  选定状态行，条件 gather 到 ticket 独占 GPU staging
  record capture_event
  允许后续批次写运行态

搬运流
  wait_event(capture_event)
  批量 state D2H、KV gather/offload
  record copy_done

CPU post / cache owner
  根据接受和停止结果决定 RETAIN / DISCARD
  event.query + 全 TP 分片完成确认
  短元数据事务发布对应位置 READY
  释放 ticket 的临时引用
```

CPU 不必阻塞等待 freeze 完成来放行下一批，但 GPU 的执行依赖必须保证下一次状态写入晚于 freeze。跨流只等待 forward_done 不能保护源状态；下一批仍可能同时改写它。独占 staging 把慢 D2H 移出主计算依赖链，但稀疏 gather 本身仍有成本。

两个 infer 线程/microbatch 必须各自携带 ticket，不能用一个全局 last_hidden。同一 TP/模型分片组使用同一个逻辑 capture plan；不同 DP 组可以处理不同请求，但须正确参与 collective 协议，包括空 batch 和 padding。CUDA graph 路径使用预分配固定容量 staging/描述数组和有效 mask，不在 replay 中动态分配。

### 5.3 不每个 decode token 都做快照

默认捕获策略：

| 位置 | 策略 |
|---|---|
| 精确 prompt 终点 | 预知，优先保存 state + output seed |
| 自然 chunk 终点 | 根据间隔预算保留一部分，不额外按页拆 chunk |
| EOS、max_new_tokens、可设备判定的 token stop | sample/verify 后条件捕获，CPU 后处理最终批准 |
| MTP 接受段中间遇 stop | 根据 token/state 的实际位置关系选择已冻结候选行；adapter 不具备该能力时回退 |
| CPU/外部晚到的字符串 stop、abort | 默认回退较早有效检查点，可在后台预算内重建结束检查点 |

设备 stop mask 必须与 ignore_eos、min_tokens、token stop 序列等真实规则等价，不能仅判断 `token_id == eos`。CPU 仍是可见输出边界的最终判定者。

若 stop token 是第 r 行刚采样出来的，row r 的状态仍在它之前；只有下一行确实处理了该 token 且对应前缀有效，才有包含它的状态。若删除多 token stop 序列后，目标边界早于本批保存的版本窗口，也只能回退，不能用当前 MTP bank 冒充历史状态。

独立 detokenizer 的字符串停止可能延迟多批；两个状态版本不能覆盖任意延迟。若产品要求严格保留每个此类终点，必须选择：

- 有界状态版本环 + 有界确认窗口，窗口满时回压；或
- 从较早检查点重算尾段，支付重建代价。

不承诺同时获得“所有停止点精确保存、无额外显存、无额外拷贝、从不阻塞”。默认优先不破坏推理流水线和正确性，缓存 admission 失败只损失本次缓存。

### 5.4 释放与取消

state/seed 已冻结到独立槽后，原请求运行态不必一直等 CPU/network 传完；满足原执行依赖即可回收请求槽。KV transfer 任务持有独立 KV lease、不可变 token prefix/manifest，不能继续依赖已被重用的 Req 对象。

网络失败、超时或取消仅使缓存事务失败。真正的 buffer 回收要等相关设备操作结束；租约 deadline 不能直接释放仍在传输的内存。

## 6. MTP、完整命中与输出信息

### 6.1 状态导出接口必须按模型实现

定义 `ModelResumeAdapter`，职责为：

```text
select_capture_candidates(batch_metadata, verify_result, device_stop_mask)
freeze_state(exact_len, state_selector, destination)
finalize_capture(ticket, cpu_stop_result)
restore_state(checkpoint, request)
capture_output_seed(final_hidden, exact_len, destination)
plan_auxiliary_resume(checkpoint, newly_sampled_tokens)
```

选择候选与 freeze 位于允许下一批覆盖之前；CPU `finalize_capture` 只能批准已冻结候选、丢弃或回退，不能再读取当前运行态补造较早状态。

Qwen3.5 MTP 的 SSM 槽是 `req_idx * (mtp_step + 1) + accepted_row`；这里的 row 是请求内 `b_mtp_index`，不是压缩 batch 的全局行号。conv 使用该 row 对应的窗口。恢复归一化到 canonical 槽，并初始化 MTP 状态索引。其他线性模型不得假定相同布局。

不能把所有已计算候选当作已提交 token，也不能把最大 accepted row 当作被 stop 截断后的终点。若所需状态行已经覆盖，回退，不伪造状态。

### 6.2 OutputSeed

首版定义明确格式：`target_final_hidden_before_final_norm`，并记录模型、dtype、TP/DP layout 和精确长度。捕获发生在 target 最终层数据就绪、请求行顺序恢复后，且早于后续复用/原地修改。

`mtp_collector.spec_hidden` 不是通用 LM-head 输入：普通模式没有它，一些模式是中间层拼接，Qwen MTP 还会原地 norm。因此 seed 必须独立拥有内存，不能保存 collector 别名。

完整命中 `L == input_len`：

```text
seed 到 GPU → HEAD_ONLY（final norm + LM head + vocab gather）
            → 本次请求的 sampling/约束/计数更新 → 当前请求首 token

缺失 KV/state onload ───────────────────┐
首 token + auxiliary bootstrap ─────────┴→ DECODE_READY → decode
```

不重放旧采样结果，也不缓存经过旧请求 temperature/penalty 处理的 logits。不同 sampling seed/约束可以共享模型输出信息，但应按本次请求重新执行处理。

HEAD_ONLY 本身不读取 KV/state，因而 seed 与采样上下文就绪后，可以和大块 KV onload 折叠；下一步 decode 才等待完整恢复。工程上仍由正常调度器安排 LM-head 的 TP collective，不能在任意后台线程发起并打乱通信次序。若首 token 提前返回而 KV 尚未加载完，必须同时记录首个 decode 的等待，不能只用变好看的 TTFT 掩盖后续停顿。

`prompt_logprobs` 需要整个输入的分布，一个终点 hidden 不足够；缺少对应缓存时保留重算路径。其他需要逐 token 输出的接口按 capability 同样处理。

### 6.3 Draft 的有效边界单独表达

当前 Vanilla/EAGLE draft 填充可能把输入左移，并在尾部使用本次采样的首 token。新请求重新采样后，target KV/state 可以相同，但旧 draft 尾槽可能不再有效。

辅助描述必须含模式、有效长度、依赖 token/特征和恢复版本。adapter 根据新采样 token 重建受影响 draft 尾部，写请求独占槽；多级 draft 逐级传播依赖，不能拿 target 的 L 当作全部辅助层的 L。

没有 adapter 的模式不启用该检查点恢复能力，走原正确路径。可以以后增加显式 target-only bootstrap，但不能假设只跑一个 target token 就自动重建任意 proposer 的完整历史上下文。

## 7. Agent 与 PD 的完整流程

### 7.1 标准 Agent API

标准 chat/tool 请求继续正常 render→tokenize。匹配验证的是本次实际 token 前缀：工具 JSON 规范化、reasoning 是否回传、模板和 stop token 都可能让它不同于上次 raw output。

保留精确 prompt 终点和输出终点，并按预算保留少量中间检查点。只有末态时，Agent 模板在更早位置分叉就无法恢复；需要回退到沿途已有状态。

可选 continuation hint 只用于定位 owner 和 prefix；当前 Responses API 的 `previous_response_id` 并没有现成状态化能力。原生 token continuation 应作为独立 API 能力设计，不能把 hint 解释为跳过验证的授权。

工具等待期间可以异步导出检查点、预取下一轮需要的副本。并行工具分支和 retry 保留不可变共同前缀，各自生成独立后缀。

### 7.2 P 和 D 之间的目录与数据

当前独立服务有独立 CPU shared-memory id 和目录，即便同机也不会自动共享。建议显式设置 cache owner：

- 首版仍由 P 侧管理 CPU cache；D 增加受预算控制的 checkpoint exporter。
- P 完成输入后发布其 KV 基础 manifest，P→D 控制信息携带可验证的 base handle。
- D 结束时导出 state + 新增 KV + 可选 seed/auxiliary 描述，目的 owner 检查已有基础页。
- 基础页仍在，只传增量；基础页已淘汰则补传缺失范围，或放弃本次 export。不能永远假设 P 的输入页还在。
- 完成发布后，再将 location/generation 告知路由目录。下一轮优先可直接使用的 owner，同时考虑排队和传输成本。

长期 Agent 会话不持有全前缀硬租约。lease 用于实际查找/传输事务；工具等待阶段主要依靠普通保留策略和短期预算，防止挂起会话锁死缓存容量。

P TP2、D TP4 等布局不能直接交换 rank-local state 槽；复用已有 PD 全局 head 打包/拆分概念，但 decode 导出必须先按 committed state selector 归一化，不能照搬 prefill 固定槽导出。

### 7.3 PD 控制协议显式化

新增/扩展协商信息：

```text
target_prefix_len
kv_ready_len / state_ready_len
output_seed_capability / auxiliary_capability
missing_data_plan
first_token_owner = P | D
transfer_epoch
```

`RESUME_READY` 和 `FIRST_TOKEN` 独立于数据任务，允许 0 KV 字节、仅 state、仅 seed 等计划。按 request generation + epoch 保证首 token exactly-once，收到重复通知不得再次递增输出计数。

这替换当前 `ready_kv_len == input_len-1` 的隐式判断，也消除“必须有最后一个 KV task 才能附带首 token”的耦合。

### 7.4 短后缀在 D 续算属于另一个调度扩展

这里的“流水线折叠”首先指现有 CPU/GPU、双 infer 线程和 microbatch overlap。若还希望 Agent 短后缀直接在持有状态的 D 续算，可作为后续 locality-aware 路由优化。

不能立即把所有 Agent prefill 发到 D。D 需要显式接收 suffix-prefill 的入口、token 预算和 admission；长工具结果仍可走 P。比较 D 短 prefill 对其他请求 TPOT 的影响与 P/D 搬运代价，再定阈值。

### 7.5 一个完整的 12000-token 例子

先取普通 decode，假设下一轮模板完整保留上轮 token 前缀，且停止后没有额外超前计算：

```text
第一轮 P：输入 12000，chunk 预算 8192
  forward [0,8192)       → 可按策略捕获 S@8192
  forward [8192,12000)   → 捕获 S@12000 + H@12000
  KV 与 state 独立保存；CPU KV 视图为 8192 + 3808
  不为 hash 对齐额外计算一个 224-token chunk

第一轮 D：生成 317 个 token
  最后一枚刚采样，冻结的已计算前缀为 12316
  导出 S@12316 + 新增 KV[12000,12316)
  owner 验证基础 KV 后，发布 checkpoint@12316

工具运行后：新输入 = 原 12000 + 输出 317 + 新增 400
  实际 token LCP 验证通过，命中 checkpoint@12316
  恢复 KV[0,12316) + S@12316
  只 prefill 剩余 401 个 token，再继续生成

另一个请求：输入恰好还是原来的 12000 token
  命中 checkpoint@12000
  读取 H@12000，HEAD_ONLY → 新请求重新采样首 token
```

`H@L` 表示处理前 L 个 token 后，最后位置的 target hidden。它解决完整输入命中后的首 token 生成；对于后面还要追加工具结果的请求，正常 suffix prefill 就会产生新的 logits，并不需要先对旧 H 再采样。

CPU 尾页可在满足唯一追加写租约时由有效 3808 延伸到 4124；发生分支则使用独立尾页。若模板重渲染使前缀在更早位置变化，以实际 LCP 回退，不保证这个示例一定能命中 12316。

## 8. CPU onload/offload 的效率约束

### 8.1 搬运描述符与批处理

以 `(source handles/token indices, destination page, offset, valid_count, layout)` 描述范围，多请求、多页合并提交。KV 根据现有 token 索引 gather，状态从独立池按 batch 搬运。

- onload 只加载 GPU 缺失的 KV 范围和选中终点的一份 state，缺 seed 时才加载 seed；不沿途加载所有 state。
- offload 只写新缺失范围，不重复发送已经 READY 的共同前缀；尾部 COW 单独计量。
- 一次传输可以跨页，也可以只涉及尾部有效区；不为逻辑 token 逐个启动 kernel。
- CPU lock 只保护元数据预留、租约和发布，不持锁跨 CUDA、网络或磁盘等待。
- 全 TP 完成信号必须属于同一个 ticket/epoch，不可某一 rank 完成就发布整份状态。

### 8.2 折叠加载与调度

新请求先得到 `ResumePlan`，进入 `LOAD_WAIT`，异步发起缺口 onload；只有消费该结果的请求等待 load event，其他已 READY 请求可继续计算。

依赖应细化为 `HEAD_READY` 与 `DECODE_READY`：完整命中时前者只需 output seed、请求采样上下文及调度通信条件，后者还需 KV/state 和 auxiliary 恢复完成。suffix prefill 则必须先满足该计算真正需要的 KV/state 依赖。首版可以统一等待全部完成保正确，再单独启用 HEAD_ONLY 与 onload 的折叠。

不能只删掉当前 `synchronize()`：需要新增请求 ready 状态、调度过滤、源/目标 lifetime 和跨 rank 完成协议。FA3 等现有特殊同步保护先保留，资源级事件依赖验证通过后再替换。

增加独立 staging、pending-copy、pending-export 队列的字节上限。工具等待期间可以 offload，但不能让长 D2H 队列饿死新请求的 H2D onload；优先级/限流按目标硬件实际 copy engine 和链路竞争测试决定。

### 8.3 所有内存预算可见

启动日志至少报告：

```text
GPU KV capacity
GPU live state bytes
GPU checkpoint staging bytes
CPU KV pinned bytes
CPU checkpoint state bytes
CPU output-seed bytes
pending transfer pinned bytes
```

GPU staging 和可选版本环必须在 KV 容量 profiling 前预留。取消“只配置小页槽，另有隐式大页池”的容量表达；按字节预算/实际单槽大小显示可用数量。

跨 TP 总内存和每 rank 内存分别展示，避免把单 rank 状态槽或 CPU 整份状态统计混为一谈。

## 9. 淘汰、过载和故障

- GPU KV 淘汰不直接释放 state。检查点是否仍可用取决于另一层是否有完整 KV。
- state 淘汰不直接释放共享 KV。KV 可以继续支持其他检查点或纯全注意力缓存。
- 普通目录/manifest 引用不等于永久 pin 所有物理页；generation 验证及无完整覆盖时的回退是必要的。
- 页覆盖变化更新目录可用性；惰性校验允许短期陈旧提示，但取得使用租约前必须再次验证。
- 只优先保留 Agent 最近检查点也不够：共享系统前缀、中间稳定前缀可能更有价值。首版沿用易解释的 LRU，加受限的最近回合优先级，避免每会话无限保留。
- staging 满、CPU 池满、export 失败：放弃新缓存或回退旧检查点，不能让普通请求因为缓存优化而永久等待。
- cache flush 使用 epoch/generation 失效新查找，再排空或取消旧事务；不得立即释放仍被设备读取的 slot。
- 若启用 disk cache，新增纯 KV/state 格式必须版本化；首版未实现 state 持久化时不承诺任意 checkpoint 的跨重启恢复。旧打包页不得按新格式解释。

## 10. 工程拆分与验收

### 10.1 建议模块职责

| 模块/入口 | 改造职责 |
|---|---|
| `infer_batch.py` | 删除固定尾边界作为唯一恢复点的假设；引入 checkpoint/plan handles；分离请求释放与 transfer leases |
| 新 `CheckpointDirectory` | 精确前缀索引、版本句柄、租约、可用性，不访问 CUDA |
| 新 `StateStore` / `CaptureManager` | 有界 state/seed/staging 池、ticket、event 轮询、跨 rank 发布 |
| `req_manager/linear_att.py` | committed state export/restore adapter；明确 MTP row/window |
| `post_layer_infer` / `ModelOutput` | 专用 output seed 捕获，不复用 spec_hidden 别名 |
| `chunked_prefill/impl.py` / `InferReqUpdatePack` | batch epoch、精确长度、capture ticket 和停止结果的绑定 |
| `multi_level_kv_cache.py` / CPU client | 纯 KV 有效范围搬运、LOAD_WAIT、manifest、独立 state 加载 |
| PD master + P/D backend | 显式恢复/首 token 协议、D export、owner 路由提示 |
| 各 proposer | draft frontier、依赖和尾槽重建 adapter |

接口可按项目风格合并到现有类，以上是职责边界，不要求机械地新增同样数量的类。

### 10.2 分阶段落地

1. **先做生命周期和模型 adapter**：精确长度、ticket、state selector、取消/回收。普通 decode 与 Qwen MTP 分别验收，再启用对应能力。
2. **单服务、GPU KV 常驻的精确检查点**：token 前缀索引独立；state 仍可放 CPU 独立池，保留自然 chunk 末尾和精确输入尾部；增加 OutputSeed/HEAD_ONLY，删除仅为 hash 尾边界引起的切分。
3. **CPU 解耦与范围搬运**：纯 KV 页、独立 state、尾页视图和 lease。初版可保留现有同步安全屏障，随后独立验证 LOAD_WAIT/event pipeline。
4. **PD 精确恢复协议**：0-byte 控制任务、首 token owner、跨 TP state/auxiliary 转换。
5. **D 输出导出与 Agent 路由**：基础 manifest 验证、增量传输、工具等待期 export，标准 chat 仍验证 token LCP。
6. **依据瓶颈扩展**：严格字符串 stop 版本环、D 短后缀 prefill、CPU 尾片压缩。每项单独证明收益和成本。

新旧缓存模式通过配置与协议版本隔离，启动时验证模型 adapter 和服务间能力。不能让某个 unsupported proposer 在运行时悄悄使用不完整的恢复信息。回滚时释放新模式事务并冷启动对应缓存，不复用不兼容的共享内存布局。

### 10.3 必须证明的行为

正确性覆盖：任意长度/大页前后 1 token、完整命中重新采样、不同约束、prompt_logprobs 回退、MTP 部分拒绝和接受段中间 stop、工具序列化改变前缀、并行分支、GPU-only/CPU-only/混合覆盖、尾页追加与 COW、不同 TP、0-byte PD、重复 FIRST_TOKEN、下一批覆盖 state、graph replay 覆盖 hidden、slot ABA、拷贝时淘汰、flush/abort、队列满和延迟 rank。

验收不仅看 token 命中率：

- 冷请求 TTFT：12000 输入在 chunk=8192 且无其他调度约束时应为 2 次 target prefill，而非为 11776 再拆一次。
- 真正少算的 token，以及来自 D 输出的复用 token 数。
- Agent 下一轮 token LCP、可用 checkpoint 长度、模板变化造成的回退长度。
- 完整命中 HEAD_ONLY 次数和 MTP auxiliary 重建成本。
- TTFT/TPOT p50/p95/p99，其他并发请求的尾延迟。
- checkpoint 捕获次数、D2D/D2H 字节、对计算流的阻塞时间、状态/staging 峰值。
- KV 增量传输、基础页补传、尾页 COW、CPU 内部空洞。
- 因状态缺失、KV 缺口、未 READY、capability 不足而回退的次数。

任何 benchmark/eval 都按仓库要求使用 `exp -m ...` 记录；本设计稿没有执行新的性能实验，也不声称上述方案已有收益数据。

## 11. 代码依据

- `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py`：forward、预更新、notify_forward、post_handle 的次序。
- `lightllm/server/router/model_infer/mode_backend/overlap_events.py`：双 infer 线程握手。
- `lightllm/server/router/model_infer/infer_batch.py`：固定 `linear_att_cache_len`、当前快照复制、请求释放、`InferReqUpdatePack`。
- `lightllm/server/router/model_infer/mtp_speculative/utils.py`、`lightllm/common/basemodel/triton_kernel/mtp_utils.py`：接受状态索引。
- `lightllm/common/basemodel/attention/linear/gdn.py`、`lightllm/common/basemodel/triton_kernel/linear_att/causal_conv1d_mtp.py`：SSM 候选行和 conv 窗口。
- `lightllm/models/llama/layer_infer/post_layer_infer.py`：最后位置、final norm 和 LM head。
- `lightllm/models/qwen3_5_mtp/layer_infer/pre_layer_infer.py`：target hidden 原地 norm。
- `lightllm/server/router/model_infer/mtp_speculative/proposers/vanilla_with_att.py`、`eagle_with_att.py`：draft 输入左移和采样 token 依赖。
- `lightllm/server/router/model_infer/mode_backend/multi_level_kv_cache.py`：现有同步、prompt-only offload 和 READY 发布。
- `lightllm/server/router/model_infer/mode_backend/pd/decode_node_impl/decode_impl.py`：禁用 CPU cache、最后一个输入 token 假设。
- `lightllm/server/router/model_infer/mode_backend/pd/prefill_node_impl/prefill_impl.py`：KV/state 传输和首 token 附着。
- `lightllm/server/httpserver_for_pd_master/manager.py`、`pd_selector/pd_selector.py`：PD 握手、token 编码和路由。
- `lightllm/server/build_prompt.py`、`api_openai.py`、`api_responses.py`：Agent 模板/工具重渲染与无状态 Responses 接口。
