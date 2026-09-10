# 精确前缀缓存：H100 验证记录

日期：2026-09-10。基线为 upstream `1eb4810c78c6ea908b69c20c3c037b79979e1cd6`，候选按独立源码快照记录。最终R12在原MTP负载上消除了此前的大停顿，TTFT降低28.24%、总延迟降低20.88%，但**TPOT仍回退12.97%，尚未追平基线**。所有 benchmark/eval 经 `exp -m` 执行，保留失败运行、源码、启动日志、原始请求及逐 token 概率。**本功能默认关闭；恢复正确性、冷算差异和性能分别报告，不宣称全面提升或整体全绿。**

## 环境与验证边界

- H100 80GB HBM3，driver 580.167.08，PyTorch 2.11.0+cu130。
- Qwen3.5-0.8B：普通模式、DP2双微批、P/D分离，以及 `eagle_with_att` MTP step=3。DP组CLI `--tp 2 --dp 2`表示两个总rank、每个DP副本TP1，不能解释为每副本TP2；各组配置见启动工件。
- Qwen3.5-27B：同一对 GPU 依次运行 baseline/candidate，TP2/DP1、MTP0、关闭 CUDA graph。两侧实际后端相同：Fa3 full-attention prefill、FlashInfer decode、FlashQLA linear prefill、Triton linear decode。
- GPU KV 容量 65536 tokens，running requests 32，请求上限32768，chunk/batch预算8192；小页256、大页256×32、旧小页状态槽128。27B CPU预算8192 MiB/TP rank；MTP并发8和DP/CPU专项4096 MiB/rank；PD专项1024 MiB/rank。
- R8上传tar SHA256：`fc329aa845f507fea9249707771c75276051aea8d200f84742b9e91e328f8da1`。R10：`46e489e9e9b85cf15a518708c27bc1eeced583e16c7b3cbf18fdfc389cc52466`。
- 最终R12上传tar SHA256：`0c900118b9cbd83399f761cc8fb1052d4ef32c320b404b8728dce7d1cd31b6d9`。本轮14个变更Python文件与该快照逐文件SHA256一致。

早期0.8B对比曾出现baseline的FlashQLA降级到Triton，该组不用于性能归因。首次shape编译和拆批抖动保留在原始记录中，不通过删除异常波使结果变好。下述短输出负载不是饱和吞吐测试。

## 本轮修复与证据

| 问题 | 修复及验证 |
|---|---|
| 同tokens混入不同计算历史的KV | origins同时约束CPU页去重、GPU radix、P→D和D→P。此前S@284恢复混入95776个不同target KV元素、最大差0.185547；真实保存/恢复的state及KV字节断言验证修复。 |
| 任意终点与MTP状态 | 8193最后一个输入token进入decode时仍捕获prompt endpoint；保存实际accepted conv窗口/SSM row，私有packed尾槽重建后更新来源；HEAD使用独立输出缓冲命名空间。 |
| 输出发布约49.56ms | 首次pinned页分配35.53ms、state分配9.03ms是主要热点。启动预留有界arena，消除请求内大块`cudaHostAlloc`；窗口拥有独立storage边界，lease/COW/flush/序列化及OOM生命周期已验证。 |
| 折叠流水线额外等待/计算 | HEAD和aux修复批量执行；shared forward generation阻止伙伴已开始forward时仍睡20ms；max-output门控保留PASS握手和延迟释放，但不再白算达到输出上限后的整轮。 |
| capture与搬运同步 | pinned元数据异步上传；最后一页KV与state/seed共用完成等待，多页GPU临时空间仍最多一页。normal模式后台准备CPU快照，主调度统一提交；onload和PD发布仍同步。 |
| R8发布可见性竞态（P1） | R7后台commit可让两个TP rank在同tokens/length处取得不同历史：真实CPU cache＋TP2 Gloo复现rank0 origins222/state22、rank1 origins111/state11。R8 worker仅prepare，主调度交集完成epoch后commit/discard、释放hold并ACK。另修复candidate查询后被evict导致`acquire=None`的异常。 |
| R9首轮MTP布局错误 | 只移除无效draft行，却遗漏GDN/FA3的fixed4布局假设。真实模型20轮有320项比较失败；该运行不是收益证据。R10让attention builder、page table及graph元数据按实际请求/候选边界构造。 |
| 可变行数触发重复JIT | R11把实际行数改为runtime参数；H100改前/改后各384项检查通过，capture选择kernel变体32→4，MTP start-location变体89→6。R12启动预热有限BLOCK变体：独立空Triton cache的H100专项348项通过，预热后行数0..32及normal/prefill/DP局部边界无新变体，conv/SSM/KV及请求索引字节不变。 |
| 同一checkpoint重复发布 | R12仅在所有TP rank已有相同来源、满足seed要求的条目时复用，避免重建CPU页/state。真实CPU cache＋TP2 Gloo的13场景、26个rank结果全部通过，覆盖单rank缺失、clear、OOM、异常、完成时序差异和ACK；该专项不证明GPU性能。 |
| DP MTP辅助计算误用graph | R10真实服务8个seed成功，第9个请求首次HEAD超时：单批`resume_auxiliary`误replay双微批graph，复制缺失infer state时触发`vars(None)`。R12让该单批调用走现有eager路径，正常成对DP调用保留graph；132项真实dispatch方法的CPU契约先红后绿，随后真实DP MTP复验通过。 |

R8 P1回归先红后绿：原R7两项缺陷均复现；R8通过版本一致性、eviction回退、两任务ACK、clear/discard、单rank第二槽OOM、fatal清理及generation分歧检查。这些使用生产方法、真实CPU cache、线程和TP2 Gloo，CUDA上下文被替代，**不能单独证明GPU数值或服务性能**；实际服务证据见下文。

R10在H100执行真实GDN、FA3 FP、FA3 MLA state builder和Triton元数据/page-table kernel，覆盖一行HEAD、完整候选、混合/补齐/空批、CUDA graph replay及DP workspace独立性。48个gate组合均通过；原summary误写96，纠正工件与原始记录同时保留。现有相关测试91项通过。该builder专项不加载模型权重，不把graph元数据验证等同于完整模型输出验证。

## R12真实MTP功能

R12 probe完成28请求、124项checks、15项恢复/replay比较，全部通过且无HTTP错误，覆盖Agent立即续接、256-token长输出、分支、MTP token stop、自然EOS和并发长输出。

R12长输出HTTP结束后的立即下一轮，输入532 tokens命中输出checkpoint S@512；token-stop后的下一轮命中S@259，随后重放分别完整命中532/270。较早R10相同Agent立即续接只命中旧prompt257，该次较短前缀回退记录保留。异步发布仍**不保证HTTP结束时输出checkpoint已可见**；尚未发布时允许安全回退计算。

probe的exit 0只代表上述恢复比较。R12独立strict-cold三组中，`stop_immediate_next_turn`输出IDs相同，但最大logprob差0.08048176765，超过0.03，比较失败；另两组Agent最大差0.0095384、branch为0。R10也出现同一stop比较失败。没有放宽阈值，也不声称恢复路径与完整冷算逐位等价。

R12 FlashInfer非greedy专项32请求、144项checks全部通过，16次warm全命中，无HTTP错误。本轮未再出现R10首组约55.7秒采样JIT；R10完整计时仍保留。R12非greedy与Agent功能请求并行执行，其延迟不用于性能结论。

R12 DP2、每副本TP1、双微批、MTP3真实服务完成26请求、84项checks、14项比较，全部通过且无HTTP错误；14对返回IDs和标量logprob逐位相同。实际后端为full attention Triton prefill/FA3 decode、linear attention FlashQLA prefill/Triton decode，target和draft均启用overlap graph。两个DP副本各自的隔离HEAD均只验证1 token、1 step；还覆盖旧128-token流运行中插入8个完整HEAD，以及输出S@264续接、276-token完整重放。该结果验证了R10失败路径的修复，但不声称逐forward追踪了所有1/4行组合或证明全部vocabulary logits相等。

## R8：27B、DP/CPU与R3 PD

| 路径 | 实际结果及限制 |
|---|---|
| 27B R8完整对比＋追加20轮 | 55＋176＝231请求，HTTP错误0，193次warm全部完整命中，harness 77＋328项比较通过。seed→warm IDs全相同；返回logprob154/193对逐位相同，最大差0.0130941，不能写成全部bitwise通过。 |
| R8 DP2/双微批/空rank | MTP0，60请求无HTTP错误或功能断言失败；12/12组seed→full-hit的IDs及返回标量logprob逐位一致。原suite仍exit 1：8个cold-reference比较失败，其中1个greedy首token不同。 |
| R8 GPU淘汰后CPU恢复 | 26请求，checks/comparisons全通过。两轮各10×8192正常缓存请求施压，日志两次证明`GPU=0 CPU=6000`。锚点恢复及追加17-token后的6017-token完整重放，2/2组IDs/logprob逐位一致。 |
| R3 PD输出回流 | TP1/MTP0，31/12000输入两组共14个生成请求通过。D输出checkpoint在P命中46/12015；P完整HEAD命中60/12029。12015长度回流`pages_sent=1 pages_reused=1`，只传新尾页。 |
| R3 PD首token/权限 | P HEAD、D完整HEAD及D缓存S@L后输入L+1的重复输出IDs/logprob一致、首token不重复。零数据任务明确为KV控制消息、`first_token_owner=decode`；非空P→D origins完整且为正。未认证registry访问均403。 |

27B非逐位一致的记录集中于拆批边界，主要影响首token及过渡处的第2/最后token，与batch3/5/7的计算形状相关；尚未用固定hidden seed纯head实验完成因果确认。返回token的标量logprob一致，也不等于全部vocabulary logits一致。

PD证据明确来自R3。其20个PD/`pd_io_struct.py`文件与R8哈希相同，但共用ExactPrefixCache/CPU cache已有变化，不能据此冒充R8或最终R12整条PD路径重验。该TP1/MTP0组也不证明异构TP、PD＋MTP或跨机带宽性能。

## 性能：历史回退与尚存问题

原始0.8B、MTP step3、8并发257输入/8输出的历史回退是baseline TPOT **2.32ms→13.01ms**，总延迟99.96ms→192.09ms。它是本轮调查的起点，不是最终候选性能。

**最终R12**使用同一原负载连续两次20轮，保留全部40波，每侧320个warm；共704个HTTP请求、1312项harness比较全部通过，HTTP错误0，候选320次warm全命中。seed→warm的320组IDs均相同；294组返回标量logprob逐位相同，其余最大绝对差0.0034699291，不能称为全部bitwise通过。其他服务无请求时执行的完整合并计时如下：

| 指标 | baseline | R12 | 均值变化 |
|---|---:|---:|---:|
| TTFT mean | 73.21255ms | 52.53901ms | −28.24% |
| TPOT mean | 2.245395ms | 2.536606ms | **+12.97%** |
| e2e mean | 89.19329ms | 70.56587ms | −20.88% |
| TPOT p95 | 2.45881ms | 3.44297ms | — |
| TPOT max | 2.64958ms | 3.78341ms | — |

两次candidate的TPOT全量均值分别2.4692/2.6040ms。历史13.01ms回退中的大停顿在本轮未再出现，但**约13%的TPOT损失仍在，不能宣布性能回退已全部修复**。短输出的TTFT收益与TPOT代价需同时评估；本结果不外推到饱和吞吐或27B。

下列均为原负载每侧20轮、160个warm的**完整中间结果**，各运行656项harness比较通过，候选warm全命中。TTFT/TPOT/e2e均为全量均值，单位ms：

| 快照/运行 | baseline TTFT / TPOT / e2e | candidate TTFT / TPOT / e2e | 仍存问题 |
|---|---|---|---|
| R8 | 73.969 / 2.255 / 90.001 | 51.974 / 2.756 / 71.544 | TPOT仍约+20%，按全量均值为+22.2%。 |
| R11首次 | 73.942 / 2.26816 / 90.053 | 52.964 / 3.65916 / 78.843 | 有限BLOCK首次JIT仍发生在请求内。 |
| R11完整重跑 | 73.321 / 2.29844 / 89.680 | 50.877 / 2.65470 / 69.743 | TPOT仍+15.5%；不是最终R12结果。 |

R10完整运行另保留在工件中：candidate TPOT mean3.3826ms、median2.5008ms、p95为10.2915ms，baseline mean2.2164ms；round9/18约70ms停顿。R11首次运行median2.440ms、p95为10.122ms；缓存全量扫描确认请求窗口内新编译四个变体：start-location的BLOCK8/32及capture选择的BLOCK8/32，源文件到cubin间隔24/24/32/40ms，与round1/19的三个长gap重叠。这证明停顿内发生首次编译，不能把每一毫秒都归因于编译。`r11-tail-jit-audit`中的剔除波次分析仅用于诊断，**不替换任何完整性能结果**。

27B R8与同GPU、同后端基线的对比如下。steady定义固定为warm repeat index 1/2，所有index 0和异常记录仍保留：

| 负载 | baseline TTFT / TPOT / e2e | R8 TTFT / TPOT / e2e |
|---|---|---|
| 单请求257 | 137.909 / 41.610 / 429.214ms | 43.357 / 42.123 / 338.301ms |
| 单请求8193 | 126.096 / 40.876 / 412.268ms | 37.921 / 42.514 / 335.555ms |
| 单请求12000 | 157.912 / 40.780 / 443.423ms | 47.025 / 41.469 / 337.343ms |
| 8并发257 | 166.557 / 40.948 / 453.515ms | 86.369 / 45.254 / 403.437ms |

27B并发TTFT改善48.1%、e2e改善11.0%，但**TPOT仍回退10.5%**；单请求TPOT回退1.2%–4.0%。追加160个warm保留全部20轮，其尾延迟为：

| 指标 | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| TTFT | 89.390ms | 85.284ms | 114.817ms | 330.894ms | 331.753ms |
| TPOT | 47.026ms | 42.519ms | 78.536ms | 89.258ms | 89.327ms |
| e2e | 419.034ms | 385.917ms | 654.165ms | 674.077ms | 675.098ms |

分位数采用`(n−1)×q`线性插值，无异常删除。round1、round6的平均TPOT分别69.14/79.74ms；已记录的TileLang编译在这些波之前结束，不能把它们统称为该次JIT。该20轮只提供候选尾延迟证据，未配套同期20轮baseline。

R6 trace解释了首gap的一部分：HEAD即时发布首token，之后首decode event约42ms已ready，但CPU等下一轮host forward返回，到约84ms才post第2个token；真正event同步仅0.015ms。基线首token也经过普通流水线，该等待包含在TTFT。单独把HEAD改为普通stage主要会把约40ms从TPOT移入TTFT，不据此宣称提速。R8确实删除了达到max-output后的整轮计算，单请求末gap从约40ms降到1.4–3.6ms；并发拆批仍留下损失与长尾。

## 使用限制

- 默认关闭。normal输出快照后台prepare、主调度统一commit；**CPU onload仍同步，PD发布仍同步**。新请求恢复可能阻塞已有请求的post，未宣称全异步或保证尾延迟。
- checkpoint表示同一计算来源的KV/state。完整cold prefill、decode形成的状态、不同chunk和batch形状不保证浮点等价；实际cold比较出现过不同greedy token。要求与完整冷算逐位相同的使用方不能据本验证获得保证。
- 1GiB不足以同时保留本负载8路MTP的16个prompt/output checkpoint：每个约130.64MiB，合计约2.04GiB；相关验证使用4GiB。容量拒绝和较短前缀回退是允许行为，不保证命中。
- arena窗口、GPU槽、staging和CPU lease必须保留到DMA完成；clear/OOM/abort不得省略完成等待。部署须统一更新使用新共享请求/PD结构的进程。

## 复现与工件索引

原MTP性能负载（服务端均使用同一模型、MTP step3、4GiB CPU预算与已归档启动参数）：

```bash
exp -m "MTP8 exact checkpoint original workload twenty waves" \
  python test/benchmark/agent_checkpoint_cache.py \
  --baseline-url http://127.0.0.1:BASELINE_PORT \
  --candidate-url http://127.0.0.1:CANDIDATE_PORT \
  --baseline-revision 1eb4810c --candidate-revision DEPLOYED_SNAPSHOT \
  --model-dir /models/Qwen3.5-0.8B \
  --suites concurrency --concurrency 8 --concurrency-input-len 257 \
  --max-new-tokens 8 --seed 1558 --settle-ms 200 --repeats 20 \
  --run-id mtp-concurrent8-4gb-20260910 --require-exact-hits \
  --output /path/to/new-immutable-run
```

27B完整对比使用`--suites boundaries,concurrency --lengths 257,8193,12000 --max-new-tokens 8 --repeats 3 --concurrency 8 --run-id tpot-27b-20260910`，追加组仅改为concurrency/repeats20并使用独立输出目录。脚本同时保留cold→seed/warm与seed→warm，不放宽阈值。

本机工件根：`~/experiments/artifacts/checkpoint-tpot-fix-20260910/`。目录内保留源码指纹、参数、日志、原始请求及审计；远端根为`/dev/shm/lightllm-cache-optim-20260910/artifacts/`。

| 内容 | 工件目录 | exp ID前缀 |
|---|---|---|
| R8 P1先红后绿 | `r8-review-regressions/` | `260910-093103`红；`093115/093243`绿 |
| R8 MTP原负载20轮 | `tpot-candidate-twenty-r8/` | `260910-093658`，exit0 |
| R9真实模型失败 | `tpot-candidate-twenty-r9/` | `260910-095145`，exit1 |
| R10 GDN/FA3 builder与graph | `variable-verify-layout-gpu-r10/` | `260910-101300/101301`；91项测试`101308` |
| R10原负载中间结果，长尾未解决 | `tpot-candidate-twenty-r10/` | `260910-101714/101715` |
| R10真实MTP功能/strict-cold | `tpot-async-publication-http-r10/` | `260910-101324/101325`；见`diagnostic.json` |
| R10非greedy功能 | `tpot-flashinfer-nongreedy-r10/` | `260910-101442/101443` |
| R10 DP MTP真实请求失败 | `dp-mtp-r10-failures/` | `260910-102313/102314`，exit1 |
| R11 runtime行数kernel | `variable-row-jit-r11/` | `260910-102227/102228`；相关测试`102310` |
| R11原20轮/完整重跑 | `tpot-candidate-twenty-r11/`、`tpot-candidate-twenty-r11-repeat/` | `260910-102757`、`103017` |
| R11首次BLOCK编译关联诊断 | `r11-tail-jit-audit/` | `260910-102911/102923/103132` |
| R12发布复用/DP辅助graph契约 | `publication-dedup-tp2-r12b/`、`dp-mtp-aux-graph-contract/` | `260910-103124`；`102722`红、`102747`绿 |
| R12空cache启动预热GPU专项 | `warmup-kernels-r12/` | `260910-103513`；早期临时目录满失败`103442`保留 |
| R12最终原负载40轮 | `tpot-candidate-twenty-r12/`、`tpot-candidate-twenty-r12-repeat/`；汇总`r12-final-performance.json` | `260910-103813/103814`、`103935/103936` |
| R12 Agent/stop/分支/严格冷算 | `tpot-async-publication-http-r12/` | 本地`260910-103702-….cmaYtv`；远端`103703`（pid932434） |
| R12 FlashInfer非greedy | `tpot-flashinfer-nongreedy-r12/` | 本地`260910-103702`；远端`103703`（pid932419） |
| R12 DP MTP真实服务 | `dp-mtp-r12-fa3-p72/` | `260910-103654/103655`；严格审计`103833`；审计输出权限失败`103744`保留 |
| 27B R8完整对比/20轮尾延迟 | `27b-r8/` | `260910-094142/094143`、`094231/094232` |
| R6首gap实际trace | `27b-profile-r6/` | `260910-092052/092053` |
| R8 DP/CPU及严格重放审计 | `dp-r8/` | `260910-094012`原DP exit1；`094051`CPU；`094130`审计 |
| R3 PD短/长输入 | `pd-r3/` | `260910-083441/083442`、`083520/083521` |

最终R12的性能、Agent恢复、非greedy和DP MTP各自保留独立原始工件；较早R3/R8的PD、CPU压力和27B证据保留其原快照范围，不冒充最终快照重验。
