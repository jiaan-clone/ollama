# Ollama Scheduler：文本模型的加载调度全解析

> 学习笔记 · 主题：用户使用文本语言模型时，Ollama 如何调度加载模型
> 核心文件：[`server/sched.go`](../server/sched.go)（1782 行）
> 关联文件：[`server/routes.go`](../server/routes.go)、[`llm/server.go`](../llm/server.go)

---

## 一、它在整条链路中的位置

接 `/api/generate` 推理流程——HTTP handler 在真正推理**之前**，必须先拿到一个"活着的 llama-server 进程"（即 runner）。这一步就是 Scheduler 负责的：

```
GenerateHandler (routes.go:561)
   └─ scheduleRunner (routes.go:207)        ← 合并模型默认值/请求参数
        └─ sched.getRunner (routes.go:250 → sched.go:165)   ← 调度入口
             ├─ 命中已加载 → 直接复用 (fast path)
             └─ 未命中     → 排队 → processPending 冷加载 (slow path)
                  └─ load() → 启动 llama-server 子进程
        ← 通过 successCh 拿回 runner，再调 runner.llama.Completion() 推理
```

Scheduler 的本质是一个**单 goroutine 的事件循环 + 引用计数的进程池**。它要回答三个问题：**要不要复用？能不能装下？该踢谁出去？**

---

## 二、核心数据结构

### Scheduler（[`sched.go:61`](../server/sched.go#L61)）

```go
type Scheduler struct {
    pendingReqCh  chan *LlmRequest   // 待调度请求队列
    finishedReqCh chan *LlmRequest   // 请求完成事件
    expiredCh     chan *runnerRef    // 该卸载的 runner
    unloadedCh    chan any           // 卸载完成信号

    loadedMu      sync.Mutex
    activeLoading llm.LlamaServer    // 关键：全局同时只能加载一个模型
    loaded        map[string]*runnerRef   // 已加载模型池（核心查找表）
}
```

四条 channel + 一张 map，整个调度就是围绕它们转。**`activeLoading` 是设计要点**——它保证任意时刻只有一个模型在"加载中"，避免两个大模型同时挤进显存导致双双 OOM；但已经加载好的模型之间，推理请求可以完全并发。

### runnerRef（[`sched.go:1382`](../server/sched.go#L1382)）

一个 runner 就是一个**已启动的 llama-server 子进程**的句柄，外加调度状态：

- `refCount` + `refMu`：引用计数，>0 表示有请求正在用，**不可卸载**
- `sessionDuration` / `expireTimer` / `expiresAt`：keep-alive 空闲过期机制
- `Options`（嵌入的 api.Options）、`model`、`numParallel`、`contextShift` 等：加载时固定下来的配置，**复用判定时用来比对**
- `gpus` / `discreteGPUs` / `vramSize`：显存与设备信息，卸载后等待显存回收时用

### LlmRequest（[`sched.go:30`](../server/sched.go#L30)）

一次请求的载体，关键是两条**带 1 缓冲的 channel**：`successCh`（拿到 runner）和 `errCh`（出错），实现 HTTP handler 与调度器之间的异步握手。还有几个 `*Auto` 标志位（`numCtxAuto`/`numBatchAuto`/`useMMapAuto`），记录某个参数是用户显式指定的还是 Ollama 自动推导的——这个区别在后面**复用判定**和 **OOM 降级**时非常关键。

---

## 三、调度入口 getRunner —— 先尝试"零成本复用"

[`sched.go:165`](../server/sched.go#L165)，每个请求都先走这里：

1. **规范化 NumCtx**：下限 4；多模态模型下限提到 2048（[L167-177](../server/sched.go#L167)）。
2. **解析 context shift**（[L137-143](../server/sched.go#L137)）：显式配置优先；否则 GGUF 模型且 `NumCtx < 8192` 时自动开启滑动窗口。
3. **算模型 key**（`schedulerModelKey`，[L111](../server/sched.go#L111)）：GGUF 用 `ModelPath`，否则用 `digest:`/`name:` 前缀，避免不同模型撞键。
4. **查 `loaded` map**（短暂持锁，读完立即释放，[L214-218](../server/sched.go#L214)）：

```go
if runner != nil && !runner.needsReload(c, req) {
    req.useLoadedRunner(runner, s.finishedReqCh)   // 快路径：直接复用
} else {
    select {
    case s.pendingReqCh <- req:                     // 慢路径：入队
    default:
        req.errCh <- ErrMaxQueue                     // 队列满，立刻报忙
    }
}
```

**这是调度器最重要的一个分叉**：模型已加载且无需重载 → 走快路径，几乎零成本；否则进 pending 队列等冷加载。队列满时**不阻塞**调用方，直接返回 `ErrMaxQueue`（"server busy"）。

### 快路径 useLoadedRunner（[`sched.go:501`](../server/sched.go#L501)）

- `refCount++`（占用，防止被卸载）
- 停掉 `expireTimer`（取消待执行的空闲卸载）
- 把 runner 发给 `successCh`，HTTP handler 立刻解除阻塞
- 起一个后台 goroutine：`<-ctx.Done()` 后把请求发到 `finishedReqCh`，触发引用计数归还

---

## 四、复用判定 needsReload —— "现有进程还能用吗？"

[`sched.go:1426`](../server/sched.go#L1426)。决定快/慢路径的关键。以下**任一**变化都会触发重载（重启子进程）：

| 触发条件 | 说明 |
|---|---|
| runner 类型不匹配 | imagegen ↔ 文本 runner |
| **Options 任一字段变了** | `NumCtx`/`NumBatch`/`NumGPU`/`MainGPU`/`UseMMap`/`NumThread` 等深度比对 |
| context shift 开关变了 | 按新 NumCtx 重新解析后比对 |
| adapter / projector 路径变了 | LoRA、投影器换了 |
| **`Ping` 失败** | 子进程已崩溃，必须重建 |

**精妙之处在"自动参数"的处理**（[L1450-1458](../server/sched.go#L1450)）：如果新旧请求的 NumCtx/NumBatch **都是自动推导**的，就**跳过比对**、沿用旧值。这样系统因 OOM 把上下文从 8192 自动降到 4096 后，下一个请求不会因为"参数不一致"又触发一次无谓的重载——在适应性和稳定性之间取得平衡。同时 NumCtx 会先按训练上下文 `trainContext` 截断再比，避免把 `NumCtx=1000000` 和实际生效值当成不同配置。

---

## 五、主循环 processPending —— 调度的心脏

[`sched.go:250`](../server/sched.go#L250)，单 goroutine 串行处理，**这是整个调度器的核心**。每收到一个 pending 请求，进入一个内层重试循环，按顺序决策（[L268-384](../server/sched.go#L268)）：

```
快照 loaded map（持锁→立即释放）
│
├─ A. 已有该模型 runner？
│     ├─ needsReload=false → useLoadedRunner，break（复用）
│     └─ needsReload=true  → 标记 runnerToExpire（要先卸载旧的）
│
├─ B. 没有 + 已达 MaxRunners 上限？
│     └─ findRunnerToUnload() 挑一个踢掉
│
├─ C. 没有 + 未达上限 + 当前一个模型都没加载（loadedCount==0）？
│     └─ load(requireFull=false)   ← 第一个模型：尽量装，装不下就部分卸载到 CPU
│
└─ D. 没有 + 未达上限 + 已有其他模型？
      └─ load(requireFull=true)    ← 必须完整放进显存才算"装得下"
            ├─ 返回 false（needEvict=false）→ 装下了，break
            └─ 返回 true → 需要腾地方：
                  ├─ 已重试过(oomRetryAttempted) → evictAllAndWait 全部清空再试
                  └─ 否则 findRunnerToUnload 踢一个
```

**MaxRunners 自动推导**（[L305-315](../server/sched.go#L305)）：用户没设 `OLLAMA_MAX_LOADED_MODELS` 时，按 `3 × GPU数量`（`defaultModelsPerGPU=3`）自动决定每台机器最多同时加载几个模型。

**腾地方的执行**（[L362-383](../server/sched.go#L362)）：把待卸载 runner 的 `sessionDuration` 清零、发到 `expiredCh`，然后**阻塞等 `unloadedCh`**，确认卸载完成后 `continue` 重试本次加载。这保证"先腾出、确认显存回收、再加载"的串行安全。

> 关键对比：`requireFull=false`（第一个模型）允许部分层 offload 到 CPU 也要跑起来；`requireFull=true`（已有其他模型）则要求新模型能完整放进剩余显存，否则触发驱逐。

---

## 六、冷加载 load —— 真正启动 llama-server

[`sched.go:535`](../server/sched.go#L535)，最复杂的一段。流程：

1. **定并发度 numParallel**（[L536-549](../server/sched.go#L536)）：来自 `OLLAMA_NUM_PARALLEL`；embedding 模型强制为 1；部分架构（mllama、qwen3vl、nemotron_h 等）不支持并行也强制为 1。
2. **加载 GGUF 元数据**（`llm.LoadModel`，[L566](../server/sched.go#L566)）：拿到层数、训练上下文长度。
3. **预测显存 + 选放置位置**（[L574-581](../server/sched.go#L574)）：
   - `effectiveLlamaServerContext = NumCtx × numParallel`（受训练上下文截断）
   - `llm.PredictServerVRAM` 估算权重 + KV cache 显存
   - `selectLlamaServerPlacement` 选 GPU：单卡按 80% 容量装箱，多卡按可用显存挑组
   - `applyAutomaticGenerationBatch` 自动选 batch（256/512/1024/2048），再加显存附加量（batch≥2048 加 2GB，≥1024 加 768MB）
4. **预检（pre-flight）**（[L586-609](../server/sched.go#L586)）：若 `requireFull` 且已有模型，预测显存 > 可用的 80% 就**直接返回 true 请求驱逐**，不浪费时间启动子进程。
5. **mmap 默认值**（[L611](../server/sched.go#L611)）：CPU-only、Windows+CUDA、Metal 部分 offload、Linux 独显主机内存吃紧等情况会关掉 mmap。
6. **启动子进程**（`newServerFn` → `llm.NewLlamaServer`，[L616](../server/sched.go#L616)），设 `activeLoading = llama`。
7. **`llama.Load()`**（[L670](../server/sched.go#L670)）：等子进程上报实际层分布。
8. **错误处理（含 OOM 一次性重试，[L671-716](../server/sched.go#L671)）**——见下节。
9. **构造 runnerRef 插入 loaded map**（[L739-773](../server/sched.go#L739)），清空 `activeLoading`。
10. **后台 goroutine 等就绪**（`WaitUntilRunning`，[L775-796](../server/sched.go#L775)）：进程 HTTP 端口就绪后 `refCount++`、`loading=false`、发 `successCh`；失败则发 `errCh` 并推 `expiredCh` 清理。

---

## 七、OOM 的两级降级重试（健壮性核心）

加载崩溃时（[L692-712](../server/sched.go#L692)），`oomRetryAttempted` 标志保证**只重试一次**，避免死循环：

1. **第一级——降上下文**：若 NumCtx 是自动的，`reduceAutoNumCtxForLoadOOM` 按档位降（如 32768→4096），返回 true 让主循环重试。
2. **第二级——清空其他模型**：若还有别的模型占着显存，`evictAllAndWait`（[L1641](../server/sched.go#L1641)）把它们全卸载、等显存回收，再重试一次。
3. **再崩 → 直接报错**给 `errCh`，请求失败。

---

## 八、卸载侧：引用计数 + keep-alive + 驱逐选择

### processCompleted（[`sched.go:392`](../server/sched.go#L392)）

请求结束 → `refCount--`（[L409](../server/sched.go#L409)）。归零后：

- `sessionDuration ≤ 0` → 立刻发 `expiredCh` 卸载
- `sessionDuration > 0` → 设/重置 `expireTimer`，到点（keep-alive，默认 5 分钟）才卸载

`expiredCh` 消费端（[L439-492](../server/sched.go#L439)）：`refCount>0` 就 10ms 后重排（不能卸载在用的）；否则 `unload()` + 从 map 删除 + **等显存回收**后发 `unloadedCh`。

### 显存回收等待 waitForVRAMRecovery（[`sched.go:1492`](../server/sched.go#L1492)）

**只针对独显**（iGPU/Metal 跳过）。CUDA 驱动卸载后释放显存有延迟，若不等就加载下一个会误判可用显存、把层挤到 CPU。所以每 250ms 轮询，直到空闲显存回升或 5s 超时。

### 驱逐选择 findRunnerToUnload（[`sched.go:1715`](../server/sched.go#L1715)）

按 `ByDurationAndName` 排序（[L1612](../server/sched.go#L1612)）：**主键 sessionDuration 升序**（keep-alive 越短越先踢），次键模型名字典序。然后**优先选 `refCount==0` 的空闲 runner**；都不空闲就选排第一的等它空。

---

## 九、记住这几条就够了

1. **快/慢双路径**：模型已加载且配置没变 → 几乎零成本复用；否则入队冷加载。分叉点是 `needsReload`。
2. **一次只加载一个模型**（`activeLoading`），但已加载模型间推理高度并发——这是防双重 OOM 的关键约束。
3. **引用计数 + keep-alive 定时器**共同决定卸载：只有 `refCount==0` 且 keep-alive 到期才卸载，绝不杀正在服务的进程。
4. **自动参数复用判定**：NumCtx/NumBatch 都为自动时跳过比对，让系统能悄悄降档而不触发无谓重载。
5. **80% 显存安全线**：放置和预检都留 20% 余量给 batch 附加显存。
6. **OOM 两级一次性降级**：先降上下文，再清空他模型，再崩才报错。
7. **独显卸载后必须等显存回收**，否则下次加载会被滞后的驱动读数坑到。
8. **驱逐策略**：keep-alive 最短 + 空闲优先。

---

## 推荐的代码阅读顺序

如果要照着代码走一遍，建议这个顺序（与执行流一致）：

1. [`getRunner`](../server/sched.go#L165) → [`needsReload`](../server/sched.go#L1426) → [`useLoadedRunner`](../server/sched.go#L501)（快路径）
2. [`processPending`](../server/sched.go#L250) 的决策树（A/B/C/D 分支）
3. [`load`](../server/sched.go#L535) 含 OOM 重试
4. [`processCompleted`](../server/sched.go#L392) → [`waitForVRAMRecovery`](../server/sched.go#L1492) → [`findRunnerToUnload`](../server/sched.go#L1715)（卸载侧）

---

## 关键函数索引

| 函数 | 位置 | 职责 |
|---|---|---|
| `getRunner` | [sched.go:165](../server/sched.go#L165) | 调度入口，规范化参数 + 快/慢路径分叉 |
| `schedulerModelKey` | [sched.go:111](../server/sched.go#L111) | 计算 loaded map 的键 |
| `needsReload` | [sched.go:1426](../server/sched.go#L1426) | 判定已加载 runner 能否复用 |
| `useLoadedRunner` | [sched.go:501](../server/sched.go#L501) | 快路径复用：refCount++、停过期定时器 |
| `processPending` | [sched.go:250](../server/sched.go#L250) | 主循环，A/B/C/D 决策树 |
| `load` | [sched.go:535](../server/sched.go#L535) | 冷加载：预测显存、选 GPU、启动子进程 |
| `processCompleted` | [sched.go:392](../server/sched.go#L392) | 请求完成 → refCount--、过期定时 |
| `evictAllAndWait` | [sched.go:1641](../server/sched.go#L1641) | OOM 重试时清空其他模型 |
| `waitForVRAMRecovery` | [sched.go:1492](../server/sched.go#L1492) | 独显卸载后等显存回收 |
| `findRunnerToUnload` | [sched.go:1715](../server/sched.go#L1715) | 驱逐选择：duration 最短 + 空闲优先 |
