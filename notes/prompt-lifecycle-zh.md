# Prompt 的一生：`ollama run gemma3 "hi"` 全链路追踪

> 学习笔记 · 主题：以 **prompt 为主线**，追踪 `ollama run gemma3 "hi"` 从 server 收到请求到返回客户端的全过程
> 关联文件：[`server/routes.go`](../server/routes.go)、[`server/prompt.go`](../server/prompt.go)、[`template/template.go`](../template/template.go)、[`llm/llama_server.go`](../llm/llama_server.go)、[`cmd/cmd.go`](../cmd/cmd.go)
> 配套笔记：[scheduler-analysis-zh.md](./scheduler-analysis-zh.md)（调度器如何加载/复用 runner）

**核心前提**：gemma3 是 **GGUF 文本模型**，由 ollama server 拉起的**独立子进程 `llama-server`**（上游 llama.cpp 的二进制）真正跑推理。所以全程跨**两个进程**，靠本地 HTTP 通信。

---

## 全景图（prompt 作为主角）

```
"hi"  ── 客户端 ──▶  POST /api/generate
  │
  ▼  【ollama server 进程】
  ① 包装：  "hi" → user message → 套 gemma3 聊天模板
            ┌─────────────────────────────┐
            │ <start_of_turn>user          │
            │ hi<end_of_turn>              │
            │ <start_of_turn>model         │   ← 模型真正看到的 prompt
            └─────────────────────────────┘
  ② 调用：  r.Completion(CompletionRequest{Prompt, Options...})
  ③ 传递：  打包成 llamaServerCompletionRequest（JSON，prompt 是字符串）
            POST http://127.0.0.1:<port>/completion
  │
  ▼  【llama-server 子进程（llama.cpp）】
       JSON → 内部 tokenize → KV cache prefill → 逐 token 解码
       逐 token SSE 回流: data: {"content":"你","stop":false} ...
  │
  ▼  【ollama server 进程】
  ④ 返回：  SSE → CompletionResponse → api.GenerateResponse
            → NDJSON 逐块写回 ── 客户端 ──▶ 终端逐字打印
```

---

## ① 如何包装用户 prompt？

入口 [`GenerateHandler`](../server/routes.go#L293)，包装发生在 [routes.go:598-699](../server/routes.go#L598)。

**第一步：把裸 prompt 包成一条 user message**（[routes.go:638](../server/routes.go#L638)）

非 `raw` 模式下，`"hi"` 不会直接喂给模型，而是先组织成 chat-like 的消息数组：

- 先放 system 消息：请求带了 `req.System` 就用它，否则继承模型自带的 `m.System`（[L625-630](../server/routes.go#L625)）
- 再把用户输入包成 `api.Message{Role:"user", Content:"hi"}`（[L638](../server/routes.go#L638)），图片挂在这条消息上
- 结果存入 `template.Values.Messages`

> 设计要点：generate 故意"伪装成 chat"，好复用同一套聊天模板渲染逻辑（注释见 [L667-672](../server/routes.go#L667)）。

**第二步：套模板**（[routes.go:675](../server/routes.go#L675) → [`chatPrompt`](../server/prompt.go#L23)）

`chatPrompt`（[prompt.go:23](../server/prompt.go#L23)）做两件事：

1. **上下文截断**（[L35-74](../server/prompt.go#L35)）：从头开始逐条丢弃旧消息，直到 tokenize 后能塞进 `NumCtx`，但**始终保留最后一条消息和所有 system 消息**。
2. **渲染**（[`renderPrompt`](../server/prompt.go#L124)）：
   - 若模型配了自定义 `Renderer`（如 gemma4）→ 走 Go 代码渲染
   - gemma3 **没有 Renderer**，走 Go 模板：`m.Template.Execute(...)`（[prompt.go:141](../server/prompt.go#L141)）

**gemma3 的实际模板**（[`template/gemma3-instruct.gotmpl`](../template/gemma3-instruct.gotmpl)）：

```gotmpl
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<start_of_turn>user
{{- if and (eq $i 1) $.System }}
{{ $.System }}
{{ end }}
{{ .Content }}<end_of_turn>
{{ else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{ end }}
{{- if $last }}<start_of_turn>model
{{ end }}
{{- end }}
```

模板执行时，[`Template.Execute`](../template/template.go#L257) 会先 `collate()` 把 system 抽出来塞进 `$.System`、合并连续同角色消息，再渲染角色标记。对 `"hi"`（无 system）最终渲染成：

```
<start_of_turn>user
hi<end_of_turn>
<start_of_turn>model
```

**这就是模型真正"看到"的 prompt**——裸字符串 `"hi"` 被加上了 gemma3 的轮次标记，并以 `<start_of_turn>model` 结尾来提示模型开始作答。

> 几个边角：`req.Raw=true` 时**完全跳过**包装，直接用用户给的 prompt（[L603](../server/routes.go#L603)）；`req.Suffix` 非空是 fill-in-middle 模式；旧版 `req.Context` 会先 detokenize 成文本拼到前面（[L657-665](../server/routes.go#L657)）；`DebugRenderOnly=true` 时只返回渲染结果不推理（[L703](../server/routes.go#L703)）——这是调试模板的利器。

---

## ② 如何调用模型？

调用发生在 [routes.go:735-825](../server/routes.go#L735) 的 goroutine 里。**分界线是 `r.Completion(...)`**（[L743](../server/routes.go#L743)）——`r` 就是上一轮调度器（`scheduleRunner`）分配好的 runner（即那个 llama-server 子进程的句柄）。

server 把渲染好的 prompt + 推理参数打包成 [`llm.CompletionRequest`](../llm/server.go#L206)（[routes.go:743-753](../server/routes.go#L743)）：

```go
r.Completion(ctx, llm.CompletionRequest{
    Prompt:     prompt,        // 渲染后的 <start_of_turn>...prompt
    Media:      media,         // 图片（"hi" 场景为空）
    Format:     req.Format,    // json / json-schema 约束
    Options:    opts,          // temperature、num_ctx、top_k...
    Shift:      ...,           // 是否启用 KV cache / 上下文滑动
    Truncate:   ...,           // prompt 超长是否截断
    LeadingBOS: leadingBOS,    // BOS 协调信息
}, func(cr llm.CompletionResponse){ ... })   // 流式回调
```

进入 [`llamaServerRunner.Completion`](../llm/llama_server.go#L1364) 后：

1. **并发闸门**：`s.sem.Acquire`（[L1372](../llm/llama_server.go#L1372)）——按 `NumParallel` 限制单个 runner 的并发请求数
2. **健康检查**：`getServerStatusRetry` 确认子进程 Ready（[L1382](../llm/llama_server.go#L1382)）
3. `NumPredict` 按 `NumCtx` 做上界约束（[L1380](../llm/llama_server.go#L1380)）

> 关键：这是个**进程边界**。ollama server 不在自己进程里跑 GGML 推理，而是把 prompt 通过 HTTP 交给 llama-server 子进程。

---

## ③ 如何向模型传递 prompt？

**prompt 默认是以"字符串"形式放进 JSON 传过去的，由 llama-server 自己 tokenize**——这是最关键的一点。

**第一步：BOS 协调**（[`completionPromptForRequest`](../llm/llama_server.go#L265) → [`completionPrompt`](../llm/llama_server.go#L229)）

`tokenizerAddsBOS()`（[L243](../llm/llama_server.go#L243)）读 GGUF 元数据 `tokenizer.ggml.add_bos_token`（gemma 系一般为 true）。若为 true，且 prompt 已带 `<bos>` 或 `leadingBOS` 前缀，就**剥掉**，避免 llama-server 内部 tokenize 时再加一个导致**双 BOS**。gemma3 走 Go 模板路径、`leadingBOS=""` 且模板不输出 `<bos>`，所以这里什么都不剥——BOS 完全交给 llama-server 加。

**第二步：截断时才在 ollama 侧 tokenize**（[L267-297](../llm/llama_server.go#L267)）

只有当 `Truncate=true` 且 `len(prompt) >= NumCtx` 时，ollama 才调子进程的 `/tokenize` 把 prompt 变成 `[]int`、按 `NumKeep` 头部保留 + 丢中间、再把**token 数组**传过去。`"hi"` 这种短 prompt 不会触发，直接走字符串。

**第三步：打包 + 发送**（[Completion L1395-1467](../llm/llama_server.go#L1395)）

构造 [`llamaServerCompletionRequest`](../llm/llama_server.go#L1230)，把采样参数从 `api.Options` 映射过去：

```go
llamaServerCompletionRequest{
    Prompt:       prompt,           // any 类型：字符串（或截断后的 []int）
    Stream:       true,             // 硬编码流式
    CachePrompt:  req.Shift,        // KV 缓存复用
    NPredict:     ...,              // 最大生成 token 数
    Temperature, TopK, TopP, MinP, Seed,
    RepeatPenalty, FreqPenalty, PresPenalty, Stop, ...
}
```

- **format**：`"json"` → 内置 JSON grammar；`{...}` schema → 直接传 `JsonSchema`（[L1419-1435](../llm/llama_server.go#L1419)）
- **多模态**（有图时）：把 prompt 里的 `[img-N]` 标记替换成本进程随机生成的 media marker，图片 base64 编码塞进 `MultimodalData`（[L1439-1451](../llm/llama_server.go#L1439)）
- **JSON 编码关键**：`enc.SetEscapeHTML(false)`（[L1455](../llm/llama_server.go#L1455)）——否则 marker 里的 `<` `>` 会被转义，子进程匹配不上

最后 **POST 到 `http://127.0.0.1:<port>/completion`**（[L1460-1467](../llm/llama_server.go#L1460)）。

**llama-server 子进程收到后**（C++ 侧）：解析 prompt 字符串 → 内部 tokenize（gemma3 词表，自动加 BOS）→ KV cache prefill → 自回归逐 token 解码 → 以 SSE 流式吐出。

---

## ④ 推理完毕后如何向客户端返回结果？

这是一条**两段流式 + 一次转换**的回流管道：`llama-server SSE → ollama 解析 → NDJSON → 客户端`。

**第一段：解析子进程的 SSE 流**（[Completion L1489-1604](../llm/llama_server.go#L1489)）

`bufio.Scanner` 逐行读，剥 `data: ` 前缀，反序列化成 `llamaServerCompletionResponse`：

- **每个 token**（`Content!="" && !Stop`）→ 调回调 `fn(CompletionResponse{Content})`（[L1537-1543](../llm/llama_server.go#L1537)）
- **token 复读保护**：同一 token 连续重复 >30 次直接中止（[L1525-1535](../llm/llama_server.go#L1525)）
- **结束帧**（`Stop=true`）→ 组装 `finalResp`，带上 `DoneReason`（stop/length）和**统计指标**：`PromptEvalCount`、`EvalCount`、各阶段耗时（[L1545-1561](../llm/llama_server.go#L1545)）
- **关键时序**：最终回调 `fn(finalResp)` **延迟到 `res.Body.Close()` 之后**才发（[L1569-1585](../llm/llama_server.go#L1569)），因为上层要在这个回调里 tokenize 整段输出来重建旧版 Context

**第二段：转换成 API 响应**（回调 fn，[routes.go:754-824](../server/routes.go#L754)）

每个 `CompletionResponse` 转成 [`api.GenerateResponse`](../server/routes.go#L756)：

- 填 `Response`、`Done`、`Metrics`、`Logprobs`
- **thinking / tool 拆分**：有 `builtinParser` 时用它把原始输出拆成正文/思考/工具调用（[L771-781](../server/routes.go#L771)）；否则用通用 `thinkingState` 解析 `<think>` 标签（[L782-787](../server/routes.go#L782)）。gemma3 普通文本走后者或直接透传
- 累计原始输出到 `sb`（[L790](../server/routes.go#L790)）
- **Done 时**：补 `TotalDuration`/`LoadDuration`，非 raw 模式 `r.Tokenize(prompt+输出)` 重建 `res.Context`（[L795-808](../server/routes.go#L795)）
- 发到 channel `ch`（[L817/824](../server/routes.go#L817)）

**第三段：写回客户端**

- **流式（默认）**：[`streamResponse`](../server/routes.go#L2239)（[routes.go:892](../server/routes.go#L892)）——设 `Content-Type: application/x-ndjson`，`c.Stream` 从 channel 取一个 `GenerateResponse` 就 JSON 编码 + 换行写出一块（**NDJSON**，逐 token 一行）
- **非流式**（`stream=false`）：先把所有 chunk 聚合成完整正文，最后一次性 `c.JSON` 返回单个对象（[routes.go:840-888](../server/routes.go#L840)）

**客户端侧**：`client.Generate` 用 `bufio.Scanner` 逐行读 NDJSON、反序列化成 `GenerateResponse`、调回调（[cmd/cmd.go:1880](../cmd/cmd.go#L1880) 的 `generate`），`displayResponse` 带换行处理逐字打印到终端。最后一块 `Done:true` 携带耗时统计、`Context`，客户端据此打印 Summary 并结束。

---

## 一句话串起来

> 裸字符串 `"hi"` →（routes.go 包成 user message）→（gemma3 模板加 `<start_of_turn>` 标记）→（打包 `llm.CompletionRequest`，`r.Completion`）→（转 JSON，prompt 仍是字符串，POST 到 `/completion` 子进程）→（llama-server 内部 tokenize + 推理）→（SSE 逐 token 回流）→（转 `api.GenerateResponse`，NDJSON 逐行写回）→（客户端逐字打印）。

---

## 四个最值得记的点

1. **包装 = 套聊天模板**：generate 伪装成 chat，复用 gemma3 模板加轮次标记，结尾 `<start_of_turn>model` 引导作答。
2. **跨进程 HTTP 调用**：ollama server 不自己推理，POST 给 llama-server 子进程。
3. **prompt 默认传字符串、子进程 tokenize**；只有超长截断时 ollama 才自己 tokenize 后传 token 数组。
4. **两段流式**：子进程 SSE → ollama 转 NDJSON 逐 token 回客户端；最终帧延迟到 body 关闭后发，用于重建 Context 和统计。

---

## 关键函数索引

| 环节 | 函数 | 位置 |
|---|---|---|
| 入口 | `GenerateHandler` | [routes.go:293](../server/routes.go#L293) |
| 包装/渲染 | prompt 构建区 | [routes.go:598-699](../server/routes.go#L598) |
| 模板套用 | `chatPrompt` / `renderPrompt` | [prompt.go:23](../server/prompt.go#L23) / [:124](../server/prompt.go#L124) |
| 模板执行 | `Template.Execute` | [template.go:257](../template/template.go#L257) |
| 调用模型 | `r.Completion` 调用点 | [routes.go:743](../server/routes.go#L743) |
| 传递 prompt | `llamaServerRunner.Completion` | [llama_server.go:1364](../llm/llama_server.go#L1364) |
| BOS/截断处理 | `completionPromptForRequest` | [llama_server.go:265](../llm/llama_server.go#L265) |
| 解析 SSE | Completion 流式循环 | [llama_server.go:1489](../llm/llama_server.go#L1489) |
| 转 API 响应 | GenerateHandler 回调 fn | [routes.go:754](../server/routes.go#L754) |
| 写回客户端 | `streamResponse` | [routes.go:2239](../server/routes.go#L2239) |
| 客户端打印 | `generate` / `displayResponse` | [cmd.go:1880](../cmd/cmd.go#L1880) / [:1680](../cmd/cmd.go#L1680) |
