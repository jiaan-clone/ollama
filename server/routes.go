package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/registry"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
	imagegenmanifest "github.com/ollama/ollama/x/imagegen/manifest"
	xserver "github.com/ollama/ollama/x/server"
)

const signinURLStr = "https://ollama.com/connect?name=%s&key=%s"

const (
	cloudErrRemoteInferenceUnavailable    = "remote model is unavailable"
	cloudErrRemoteModelDetailsUnavailable = "remote model details are unavailable"
	cloudErrWebSearchUnavailable          = "web search is unavailable"
	cloudErrWebFetchUnavailable           = "web fetch is unavailable"
	copilotChatUserAgentPrefix            = "GitHubCopilotChat/"
)

func writeModelRefParseError(c *gin.Context, err error, fallbackStatus int, fallbackMessage string) {
	switch {
	case errors.Is(err, errConflictingModelSource):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, model.ErrUnqualifiedName):
		c.JSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
	default:
		c.JSON(fallbackStatus, gin.H{"error": fallbackMessage})
	}
}

func shouldUseHarmony(model *Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, model.Config.ModelFamily) {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

func experimentEnabled(name string) bool {
	return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")

var mode string = gin.DebugMode

type Server struct {
	addr          net.Addr
	sched         *Scheduler
	defaultNumCtx int
	requestLogger *inferenceRequestLogger
	modelCaches   *modelCaches
}

func init() {
	switch mode {
	case gin.DebugMode:
	case gin.ReleaseMode:
	case gin.TestMode:
	default:
		mode = gin.DebugMode
	}

	gin.SetMode(mode)

	// Tell renderers to use [img] tags
	renderers.RenderImgTags = true
}

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

func (s *Server) modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	return s.modelOptionsWithEmbeddingBatchDefault(model, requestOpts, shouldApplyEmbeddingBatchDefault(model, requestOpts))
}

func (s *Server) modelOptionsWithEmbeddingBatchDefault(model *Model, requestOpts map[string]any, applyEmbeddingBatchDefault bool) (api.Options, error) {
	opts := api.DefaultOptions()
	if opts.NumCtx == 0 {
		opts.NumCtx = s.defaultNumCtx
	}

	// api.Options stores defaulted values, so lower layers cannot distinguish
	// an unset draft_num_predict from the default. Track that while we still
	// have the raw model/request option maps.
	draftNumPredictSet := hasOption(requestOpts, "draft_num_predict")
	if model != nil {
		draftNumPredictSet = draftNumPredictSet || hasOption(model.Options, "draft_num_predict")
		if err := opts.FromMap(model.Options); err != nil {
			return api.Options{}, err
		}
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	if applyEmbeddingBatchDefault {
		opts = llm.WithDefaultEmbeddingNumBatch(opts)
	}

	if model != nil && model.DraftPath == "" && !draftNumPredictSet {
		opts.DraftNumPredict = 0
	}

	return opts, nil
}

func shouldApplyEmbeddingBatchDefault(m *Model, requestOpts map[string]any) bool {
	if m == nil || hasOption(m.Options, "num_batch") || hasOption(requestOpts, "num_batch") {
		return false
	}
	if slices.Contains(m.Config.Capabilities, string(model.CapabilityEmbedding)) {
		return true
	}
	return m.ModelPath != "" && m.CheckCapabilities(model.CapabilityEmbedding) == nil
}

func hasOption(opts map[string]any, name string) bool {
	_, ok := opts[name]
	return ok
}

func usesAutomaticNumCtx(model *Model, requestOpts map[string]any) bool {
	if _, ok := requestOpts["num_ctx"]; ok {
		return false
	}
	if model != nil {
		if _, ok := model.Options["num_ctx"]; ok {
			return false
		}
	}
	return envconfig.ContextLength() == 0
}

func usesAutomaticNumBatch(model *Model, requestOpts map[string]any) bool {
	if _, ok := requestOpts["num_batch"]; ok {
		return false
	}
	if model != nil {
		if _, ok := model.Options["num_batch"]; ok {
			return false
		}
	}
	return true
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, and consolidated options if successful and error otherwise.
// scheduleRunner 是 server API handler 和底层模型 runner 之间的调度适配层，
// 负责把“我要用某个模型推理”变成“这里有一个已经准备好的 runner 可以调用”。
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration, shift *bool) (llm.LlamaServer, *Model, *api.Options, error) {
	// name 是请求要运行的模型名；为空说明调用方没有提供必需的 model 参数。
	if name == "" {
		// 返回 errRequired，让上层把它转换成合适的 API 错误。
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	// 根据模型名读取本地模型元信息，包括模型路径、模板、能力、默认 options 等。
	model, err := GetModel(name)
	// 如果模型不存在或模型信息读取失败，直接把错误交给上层处理。
	if err != nil {
		return nil, nil, nil, err
	}

	// 旧版 llama3.2-vision 这类 mllama 模型已经不兼容当前 Ollama，需要提示用户重新 pull。
	if slices.Contains(model.Config.ModelFamilies, "mllama") && len(model.ProjectorPaths) > 0 {
		return nil, nil, nil, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	// 校验模型是否具备调用方要求的能力，例如 completion、insert、thinking 等。
	if err := model.CheckCapabilities(caps...); err != nil {
		// 给能力错误补上模型名，方便用户知道哪个模型不支持当前请求。
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	// Deprecated runner override option; ignore if present.
	// 兼容旧请求参数：use_imagegen_runner 已废弃，调度 runner 时直接忽略。
	delete(requestOpts, "use_imagegen_runner")

	// 判断 num_ctx 是否应该由 scheduler/runner 根据模型和硬件自动决定。
	numCtxAuto := usesAutomaticNumCtx(model, requestOpts)
	// embedding 模型在没有显式 num_batch 时会使用专门的 batch 默认值。
	embeddingBatchDefault := shouldApplyEmbeddingBatchDefault(model, requestOpts)
	// 判断 num_batch 是否自动决定；如果已经应用 embedding 默认值，就不再算作普通自动 batch。
	numBatchAuto := usesAutomaticNumBatch(model, requestOpts) && !embeddingBatchDefault
	// 合并模型默认 options、请求 options 和 embedding batch 默认值，得到最终运行参数。
	opts, err := s.modelOptionsWithEmbeddingBatchDefault(model, requestOpts, embeddingBatchDefault)
	// options 解析或校验失败时，停止调度并返回错误。
	if err != nil {
		return nil, nil, nil, err
	}

	// 向 scheduler 请求 runner。scheduler 会复用已加载 runner，或排队加载新的 runner。
	runnerCh, errCh := s.sched.getRunner(ctx, model, opts, keepAlive, numCtxAuto, numBatchAuto, shift)
	// runnerRef 包装了实际 llm runner、模型信息、引用计数、过期计时器等运行时状态。
	var runner *runnerRef
	// 等待 scheduler 返回可用 runner，或返回加载/调度错误。
	select {
	// 成功时，runnerCh 会返回已经加载并可用的 runner。
	case runner = <-runnerCh:
	// 失败时，errCh 会返回队列、能力、加载、显存等错误。
	case err = <-errCh:
		return nil, nil, nil, err
	}

	// 返回实际 LlamaServer 接口、模型元信息，以及最终 options。
	// 调用方随后会用 runner.llama.Completion/Embedding 等方法执行推理。
	return runner.llama, model, &opts, nil
}

func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf(signinURLStr, url.PathEscape(h), encKey), nil
}

/*
*
GenerateHandler 是 /api/generate 的 server 侧总调度器：
它负责解析请求、定位模型、处理 cloud/remote/image/load/unload 分支、
准备 prompt、调度 runner、调用 Completion，
最后把 runner 的流式输出转换成 /api/generate 的响应。
GenerateHandler 调用模型推理的设计是：

	先通过 scheduleRunner 拿到模型 runner，
	再把请求渲染成模型可用的 prompt/media，
	然后调用 r.Completion 触发真实推理，
	通过回调接收 runner 的流式输出，
	再转换成 /api/generate 的响应格式，
	最后按 stream 参数选择流式或非流式返回。
*/
func (s *Server) GenerateHandler(c *gin.Context) {
	// 记录请求进入 GenerateHandler 的时间，后面用于计算 TotalDuration 和 LoadDuration。
	checkpointStart := time.Now()

	// req 用来承接客户端 POST /api/generate 传入的 JSON 请求体。
	var req api.GenerateRequest

	// 解析请求 JSON。空 body 和 JSON 格式错误都直接返回 400。
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// top_logprobs 是对外 API 参数，这里限制在 llama-server 支持的合理范围内。
	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	// 解析并校验模型引用，区分本地模型、cloud 模型等来源信息。
	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	// 显式 cloud 模型不在本地 runner 执行，直接代理到 cloud 推理服务。
	if modelRef.Source == modelSourceCloud {
		// TODO(drifkin): evaluate an `/api/*` passthrough for cloud where the
		// original body (modulo model name normalization) is sent to cloud.
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	// 取出解析后的模型名，后面用它在本地模型存储中查找实际模型。
	name := modelRef.Name

	// We cannot currently consolidate this into GetModel because all we'll
	// induce infinite recursion given the current code structure.
	// getExistingName 会处理大小写、tag 等解析细节，找到本地实际存在的模型名。
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// 读取模型元信息，包括配置、模板、能力、权重路径、remote 配置等。
	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	// 再次校验 top_logprobs，保持进入后续执行路径前的参数合法性。
	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	// 用户显式要求 local 时，不允许把 remote stub 当成本地模型运行。
	if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// Remote 模型分支：本地只保存一个模型 stub，真正的 generate 请求转发到远端。
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		// 如果 cloud/remote 功能被禁用，就拒绝远程推理。
		if disabled, _ := internalcloud.Status(); disabled {
			c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable)})
			return
		}

		// 保存用户请求中的原始模型名，远端响应回来后再写回响应里。
		origModel := req.Model

		// 解析模型配置中的远端服务地址。
		remoteURL, err := url.Parse(m.Config.RemoteHost)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// 安全限制：只允许转发到配置允许的 remote host。
		if !slices.Contains(envconfig.Remotes(), remoteURL.Hostname()) {
			slog.Info("remote model", "remotes", envconfig.Remotes(), "remoteURL", m.Config.RemoteHost, "hostname", remoteURL.Hostname())
			c.JSON(http.StatusBadRequest, gin.H{"error": "this server cannot run this remote model"})
			return
		}

		// 远端真正识别的是 RemoteModel，因此请求发出前要替换模型名。
		req.Model = m.Config.RemoteModel

		// 请求没有指定 template 时，继承本地 stub 中记录的 template。
		if req.Template == "" && m.Template.String() != "" {
			req.Template = m.Template.String()
		}

		// Options 为空时先初始化，方便下面合并模型默认参数。
		if req.Options == nil {
			req.Options = map[string]any{}
		}

		// 合并模型默认 options，但不覆盖用户请求里显式设置的 option。
		for k, v := range m.Options {
			if _, ok := req.Options[k]; !ok {
				req.Options[k] = v
			}
		}

		// update the system prompt from the model if one isn't already specified
		// 请求没有 system prompt 时，继承模型自带 system prompt。
		if req.System == "" && m.System != "" {
			req.System = m.System
		}

		// /api/generate 没有完整 chat messages 语义；模型内嵌 messages 更适合 /api/chat。
		if len(m.Messages) > 0 {
			slog.Warn("embedded messages in the model not supported with '/api/generate'; try '/api/chat' instead")
		}

		// 根据 stream 参数决定响应 Content-Type。
		contentType := "application/x-ndjson"
		if req.Stream != nil && !*req.Stream {
			contentType = "application/json; charset=utf-8"
		}
		c.Header("Content-Type", contentType)

		// 远端 generate 的每段响应都会进入这个回调，再被转写给当前客户端。
		fn := func(resp api.GenerateResponse) error {
			// 对用户保持原始模型名，同时附加 remote 信息用于说明实际执行位置。
			resp.Model = origModel
			resp.RemoteModel = m.Config.RemoteModel
			resp.RemoteHost = m.Config.RemoteHost

			// 把远端响应编码成 JSON。
			data, err := json.Marshal(resp)
			if err != nil {
				return err
			}

			// 按 NDJSON 形式写回当前 HTTP 响应。
			if _, err = c.Writer.Write(append(data, '\n')); err != nil {
				return err
			}
			// 立即 flush，保持流式响应的实时性。
			c.Writer.Flush()
			return nil
		}

		// 创建指向 remote host 的 API client，并转发 /api/generate 请求。
		client := api.NewClient(remoteURL, http.DefaultClient)
		err = client.Generate(c, &req, fn)
		if err != nil {
			// 远端需要授权时，返回 signin_url 给客户端。
			var authError api.AuthorizationError
			if errors.As(err, &authError) {
				sURL, sErr := signinURL()
				if sErr != nil {
					slog.Error(sErr.Error())
					c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
					return
				}

				c.JSON(authError.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
				return
			}
			// 远端返回 API 状态错误时，保留远端状态码和错误体。
			var apiError api.StatusError
			if errors.As(err, &apiError) {
				c.JSON(apiError.StatusCode, apiError)
				return
			}
			// 其他错误按内部错误返回。
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// remote 分支已经完成响应写入，不再进入本地 runner 流程。
		return
	}

	// expire the runner if unload is requested (empty prompt, keep alive is 0)
	// 空 prompt 且 keep_alive=0 表示用户只想卸载模型 runner。
	if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	// Handle image generation models
	// 图片生成模型有独立处理流程，不走文本 completion。
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		s.handleImageGenerate(c, req, name.String(), checkpointStart)
		return
	}

	// raw 模式表示调用方已经自己准备好 prompt，因此不能再混用 template/system/context。
	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	// builtinParser 用于模型内置输出协议解析，例如 thinking/tool 等结构化片段。
	var builtinParser parsers.Parser
	// Harmony 模型需要使用 harmony parser，并把 max thinking 映射成 harmony 支持的 high。
	if shouldUseHarmony(m) {
		// harmony's Reasoning field only understands low/medium/high; map "max" to "high"
		if req.Think != nil {
			if s, ok := req.Think.Value.(string); ok && s == "max" {
				req.Think.Value = "high"
			}
		}
		if m.Config.Parser == "" {
			m.Config.Parser = "harmony"
		}
	}

	// 非 raw 模式下，模型配置了 parser 就初始化内置 parser。
	if !req.Raw && m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			// no tools or last message for generate endpoint
			builtinParser.Init(nil, nil, req.Think)
		}
	}

	// /api/generate 的基础要求是模型支持 completion。
	caps := []model.Capability{model.CapabilityCompletion}
	// suffix 表示 fill-in-middle/insert 类请求，因此额外要求 insert 能力。
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}

	// 根据模型能力处理 thinking：支持则加入能力要求，不支持却显式请求则报错。
	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			// 支持 thinking 的模型默认开启 thinking。
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	// 调度 runner：加载或复用模型进程，并返回实际 runner、模型信息和最终 options。
	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive, req.Shift)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
		return
	} else if err != nil {
		// 统一处理加载失败、队列满、OOM 等调度错误。
		handleScheduleError(c, req.Model, err)
		return
	}

	// 记录 runner 可用的时间点，用于最终响应里的 LoadDuration。
	checkpointLoaded := time.Now()

	// load the model
	// 空 prompt 且不是卸载请求时，表示只加载模型，不生成内容。
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	// mllama 当前只支持一张图片，多图直接拒绝。
	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image while more than one image requested"})
		return
	}

	// 把 API 层图片数据转换成 llm 层 CompletionRequest 使用的 MediaData。
	media := make([]llm.MediaData, len(req.Images))
	for i := range req.Images {
		media[i] = llm.NewMediaData(i, req.Images[i])
	}

	// prompt 初始值是用户请求里的原始 prompt；后面非 raw 模式会改成模板渲染后的 prompt。
	prompt := req.Prompt
	// leadingBOS 记录 Go 模板渲染时可能已经输出的 BOS，避免 runner 侧重复处理。
	var leadingBOS string
	// 非 raw 模式需要根据模型模板、system、messages、thinking 等信息渲染最终 prompt。
	if !req.Raw {
		// 默认使用模型自带模板。
		tmpl := m.Template
		// 如果请求显式传入 template，则优先使用请求中的 template。
		if req.Template != "" {
			tmpl, err = template.Parse(req.Template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		}

		// values 是模板渲染输入。
		var values template.Values
		// suffix 不为空时是 insert/fill-in-middle 模式。
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			// 普通 generate 会被组织成 chat-like messages，方便复用聊天模板/renderer。
			var msgs []api.Message
			// 请求指定 system 时优先使用请求的 system。
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			} else if m.System != "" {
				// 否则使用模型自带 system。
				msgs = append(msgs, api.Message{Role: "system", Content: m.System})
			}

			// 只有没有旧版 context 时才合并模型内嵌 messages，避免重复上下文。
			if req.Context == nil {
				msgs = append(msgs, m.Messages...)
			}

			// 把 generate 的 prompt 包成一条 user message。
			userMsg := api.Message{Role: "user", Content: req.Prompt}
			// 把图片数据附加到 user message 上。
			for _, m := range media {
				userMsg.Images = append(userMsg.Images, m.Data)
			}
			values.Messages = append(msgs, userMsg)
		}

		// 把 thinking 设置写入模板变量。
		values.Think = req.Think != nil && req.Think.Bool()
		values.ThinkLevel = ""
		if req.Think != nil {
			values.ThinkLevel = req.Think.String()
		}
		values.IsThinkSet = req.Think != nil

		// b 用来累计最终渲染出的 prompt。
		var b bytes.Buffer
		// req.Context 是旧版 generate 上下文；先 detokenize 成文本再拼到 prompt 前。
		if req.Context != nil {
			slog.Warn("the context field is deprecated and will be removed in a future version of Ollama")
			s, err := r.Detokenize(c.Request.Context(), req.Context)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			b.WriteString(s)
		}

		// check that we're in the `api/chat`-like flow, and if so, generate the
		// prompt the same way
		// TEMP(drifkin): we should really just detect the chat-like flow and call
		// the real chat handler, but doing this as a stopgap to get renderer
		// support for generate
		// 普通 generate 在这里复用 chatPrompt，让模型内置 renderer/chat template 生效。
		if values.Messages != nil && values.Suffix == "" && req.Template == "" {
			genTruncate := (req.Truncate == nil || *req.Truncate) && !m.IsMLX()
			prompt, media, err = chatPrompt(c.Request.Context(), m, r.Tokenize, optionsForPrompt(opts, r), values.Messages, []api.Tool{}, req.Think, genTruncate)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			// TEMP(drifkin): req.Context will be removed very soon, but we're temporarily supporting it in this flow here
			// 如果请求带旧版 context，就把 detokenize 出来的文本和 chatPrompt 结果拼起来。
			if req.Context != nil {
				b.WriteString(prompt)
				prompt = b.String()
			}
			// 记录当前模型的 BOS 处理情况。
			leadingBOS = leadingBOSForModel(m)
		} else {
			// Direct template execution flow.
			// insert/raw-template 等场景直接执行模板生成 prompt。
			if err := tmpl.Execute(&b, values); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			// 使用直接模板执行得到的结果作为最终 prompt。
			prompt = b.String()
		}
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	// DebugRenderOnly 用于调试模板渲染，只返回 prompt，不真正调用 runner。
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(media),
			},
		})
		return
	}

	// thinkingState 是通用 thinking 标签解析器，用于没有 builtinParser 的模型。
	var thinkingState *thinking.Parser
	// 有 builtinParser 时由 builtinParser 负责解析 thinking；否则尝试从模板推断 thinking 标签。
	if builtinParser == nil {
		openingTag, closingTag := thinking.InferTags(m.Template.Template)
		if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
			thinkingState = &thinking.Parser{
				OpeningTag: openingTag,
				ClosingTag: closingTag,
			}
			// 如果 prompt 本身已经以 opening tag 结尾，先把这个状态喂给 parser。
			if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
				thinkingState.AddContent(openingTag)
			}
		}
	}

	// ch 用来把 runner goroutine 中产生的响应或错误传给 HTTP 响应逻辑。
	ch := make(chan any)
	// 推理在 goroutine 中执行，外层可以根据 stream 参数选择流式或聚合式返回。
	go func() {
		// TODO (jmorganca): avoid building the response twice both here and below
		// sb 累计 runner 原始输出，用于生成最终 Context。
		var sb strings.Builder
		// goroutine 结束时关闭 channel，通知响应逻辑没有更多数据。
		defer close(ch)
		// 调用 runner 的 Completion，真正触发模型推理。r 已经是某个模型对应的 runner。在s.scheduleRunner中分配
		// r.Completion 是 server 层从“准备请求”进入“真实模型推理”的分界线。
		if err := r.Completion(c.Request.Context(), llm.CompletionRequest{
			Prompt:          prompt,                                      // 最终渲染后的 prompt
			Media:           media,                                       // 图片/多模态输入
			Format:          req.Format,                                  // json/schema 格式约束
			Options:         opts,                                        // temperature、num_ctx 等推理参数
			Shift:           req.Shift == nil || *req.Shift,              // 是否使用上下文滑动/缓存
			Truncate:        req.Truncate == nil || *req.Truncate,        // prompt 太长时是否截断
			Logprobs:        req.Logprobs,                                // 是否返回 token 概率
			TopLogprobs:     req.TopLogprobs,                             // 返回多少个候选 token 概率
			PreservedTokens: preservedTokensForCompletion(builtinParser), // parser 相关保留 token
			LeadingBOS:      leadingBOS,                                  // BOS 处理信息，避免重复
		}, func(cr llm.CompletionResponse) {
			// 把 llm 层 CompletionResponse 转成 /api/generate 的 GenerateResponse。
			res := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Response:  cr.Content,
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
				Logprobs: toAPILogprobs(cr.Logprobs),
			}

			// builtinParser 负责把模型原始输出拆成正文、thinking 和 tool calls。
			if builtinParser != nil {
				content, thinking, toolCalls, err := builtinParser.Add(cr.Content, cr.Done)
				if err != nil {
					ch <- gin.H{"error": err.Error()}
					return
				}
				res.Response = content
				res.Thinking = thinking
				if cr.Done && len(toolCalls) > 0 {
					res.ToolCalls = toolCalls
				}
			} else if thinkingState != nil {
				// 没有 builtinParser 时，用通用 thinking parser 拆分 thinking 和正文。
				thinking, content := thinkingState.AddContent(cr.Content)
				res.Thinking = thinking
				res.Response = content
			}

			// 累计原始 completion 内容，用于最终 tokenize 成 GenerateResponse.Context。
			if _, err := sb.WriteString(cr.Content); err != nil {
				ch <- gin.H{"error": err.Error()}
			}

			// runner 标记 Done 时，补充结束原因、耗时和上下文。
			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

				// raw 模式下不自动生成旧版 context；非 raw 才 tokenize prompt+输出。
				if !req.Raw {
					tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
					res.Context = tokens
				}
			}

			if builtinParser != nil {
				// Emit chunks that carry logprobs even if the parser is still buffering
				// visible content, otherwise generate logprobs disappear for models with
				// builtin thinking/tool parsers.
				// parser 可能暂时缓冲可见文本，所以只有包含有效内容/状态/logprobs 时才发给客户端。
				if res.Response != "" || res.Thinking != "" || res.Done || len(res.ToolCalls) > 0 || len(res.Logprobs) > 0 {
					ch <- res
				}

				return
			}

			// 没有 builtinParser 的普通路径，每段 CompletionResponse 都转发出去。
			ch <- res
		}); err != nil {
			// 推理运行期 OOM 时，通知 scheduler 过期相关 runner，避免继续复用坏状态。
			s.sched.expireRunnersForRuntimeOOM(m, err)
			var serr api.StatusError
			if errors.As(err, &serr) {
				// runner 返回带状态码的 API 错误时，保留状态码。
				ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
			} else {
				// 其他错误按通用 error 传给响应逻辑。
				ch <- gin.H{"error": err.Error()}
			}
		}
	}()

	// stream=false 时，server 会先把所有 chunk 聚合，再一次性返回 JSON。
	if req.Stream != nil && !*req.Stream {
		// r 保存最后一个 GenerateResponse，其中包含 Done、metrics、context 等最终字段。
		var r api.GenerateResponse
		// 非流式响应需要累计所有 chunk 的 logprobs。
		var allLogprobs []api.Logprob
		// 分别累计 thinking 和正文内容。
		var sbThinking strings.Builder
		var sbContent strings.Builder
		// 消费 runner goroutine 发到 channel 的所有响应。
		for rr := range ch {
			switch t := rr.(type) {
			case api.GenerateResponse:
				// 普通响应 chunk：累计 thinking、正文，并保留最新响应作为最终响应基底。
				sbThinking.WriteString(t.Thinking)
				sbContent.WriteString(t.Response)
				r = t
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				// 错误响应：从 gin.H 中提取 error 和 status。
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				// channel 中出现未知类型，说明内部响应协议异常。
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		// 把累计出的完整 thinking、正文和 logprobs 写回最终响应。
		r.Thinking = sbThinking.String()
		r.Response = sbContent.String()
		r.Logprobs = allLogprobs

		// 非流式模式返回单个 JSON 对象。
		c.JSON(http.StatusOK, r)
		return
	}

	// 默认流式模式：把 channel 中的每段响应按 NDJSON 写回客户端。
	streamResponse(c, ch)
}

func (s *Server) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()

	adjustTokenLimit := func(tokens []int, limit int) int {
		if bos := kvData.Uint("tokenizer.ggml.bos_token_id"); len(tokens) > 0 && tokens[0] != int(bos) && kvData.Bool("add_bos_token", true) {
			limit--
		}
		if eos := kvData.Uint("tokenizer.ggml.eos_token_id"); len(tokens) > 0 && tokens[len(tokens)-1] != int(eos) && kvData.Bool("add_eos_token", true) {
			limit--
		}
		return limit
	}

	inputTokensAndContext := func(text string) ([]int, int, error) {
		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return nil, 0, err
		}

		// TODO @nicolepardal: avoid reaching into kvData here; pass required tokenizer metadata via model/options instead
		ctxLen := int(kvData.ContextLength())
		if opts.NumCtx > 0 {
			ctxLen = min(opts.NumCtx, ctxLen)
		}

		return tokens, adjustTokenLimit(tokens, ctxLen), nil
	}

	truncateInputToLimit := func(text string, limit int) (string, bool, error) {
		tokens, ctxLen, err := inputTokensAndContext(text)
		if err != nil {
			return "", false, err
		}
		if limit > 0 {
			ctxLen = min(ctxLen, adjustTokenLimit(tokens, limit))
		}

		if ctxLen <= 0 {
			return "", false, fmt.Errorf("input after truncation exceeds maximum context length")
		}
		if len(tokens) <= ctxLen {
			return text, false, nil
		}

		truncated, err := r.Detokenize(ctx, tokens[:ctxLen])
		if err != nil {
			return "", false, err
		}
		return truncated, true, nil
	}

	truncateInput := func(text string) (string, bool, error) {
		return truncateInputToLimit(text, 0)
	}

	embedWithRetry := func(text string) ([]float32, int, error) {
		if req.Truncate != nil && !*req.Truncate {
			tokens, ctxLen, err := inputTokensAndContext(text)
			if err != nil {
				return nil, 0, err
			}
			if ctxLen <= 0 {
				return nil, 0, fmt.Errorf("input after truncation exceeds maximum context length")
			}
			if len(tokens) > ctxLen {
				return nil, 0, api.StatusError{
					StatusCode:   http.StatusBadRequest,
					ErrorMessage: "the input length exceeds the context length",
				}
			}
		} else {
			var err error
			text, _, err = truncateInput(text)
			if err != nil {
				return nil, 0, err
			}
		}

		emb, tokCount, err := r.Embedding(ctx, text)
		if err == nil {
			return emb, tokCount, nil
		}

		var serr api.StatusError
		if !errors.As(err, &serr) || serr.StatusCode != http.StatusBadRequest {
			return nil, 0, err
		}
		if req.Truncate != nil && !*req.Truncate {
			return nil, 0, err
		}

		truncated, ok, err := truncateInputToLimit(text, opts.NumBatch)
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			return nil, 0, fmt.Errorf("input exceeds maximum context length and cannot be truncated further")
		}

		return r.Embedding(ctx, truncated)
	}

	var g errgroup.Group
	embeddings := make([][]float32, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			embedding, tokenCount, err := embedWithRetry(text)
			if err != nil {
				return err
			}
			// TODO: this first normalization should be done by the model
			embedding, err = normalize(embedding)
			if err != nil {
				return err
			}
			if req.Dimensions > 0 && req.Dimensions < len(embedding) {
				embedding, err = normalize(embedding[:req.Dimensions])
				if err != nil {
					return err
				}
			}
			embeddings[i] = embedding
			atomic.AddUint64(&totalTokens, uint64(tokenCount))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{
				"error": strings.TrimSpace(serr.ErrorMessage),
			})
			return
		}

		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": strings.TrimSpace(err.Error()),
		})
		return
	}

	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	}
	c.JSON(http.StatusOK, resp)
}

func normalize(vec []float32) ([]float32, error) {
	var sum float32
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("embedding contains NaN or Inf values")
		}
		sum += v * v
	}

	norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
	for i := range vec {
		vec[i] *= norm
	}
	return vec, nil
}

func (s *Server) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	name := modelRef.Name

	r, m, _, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	embedding, _, err := r.Embedding(c.Request.Context(), req.Prompt)
	if err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := api.EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) PullHandler(c *gin.Context) {
	var req api.PullRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TEMP(drifkin): we're temporarily allowing to continue pulling cloud model
	// stub-files until we integrate cloud models into `/api/tags` (in which case
	// this roundabout way of "adding" cloud models won't be needed anymore). So
	// right here normalize any `:cloud` models into the legacy-style suffixes
	// `:<tag>-cloud` and `:cloud`
	modelRef, err := parseNormalizePullModelRef(cmp.Or(req.Model, req.Name))
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, errtypes.InvalidModelNameErrMsg)
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		if err := PullModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
			return
		}

		s.refreshModelListCache(name)
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) PushHandler(c *gin.Context) {
	var req api.PushRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var mname string
	if req.Model != "" {
		mname = req.Model
	} else if req.Name != "" {
		mname = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		name, err := getExistingName(model.ParseName(mname))
		if err != nil {
			ch <- gin.H{"error": err.Error()}
			return
		}

		if err := PushModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

// getExistingName searches the models directory for the longest prefix match of
// the input name and returns the input name with all existing parts replaced
// with each part found. If no parts are found, the input name is returned as
// is.
func getExistingName(n model.Name) (model.Name, error) {
	var zero model.Name
	existing, err := manifest.Manifests(true)
	if err != nil {
		return zero, err
	}
	var set model.Name // tracks parts already canonicalized
	for e := range existing {
		if set.Host == "" && strings.EqualFold(e.Host, n.Host) {
			n.Host = e.Host
		}
		if set.Namespace == "" && strings.EqualFold(e.Namespace, n.Namespace) {
			n.Namespace = e.Namespace
		}
		if set.Model == "" && strings.EqualFold(e.Model, n.Model) {
			n.Model = e.Model
		}
		if set.Tag == "" && strings.EqualFold(e.Tag, n.Tag) {
			n.Tag = e.Tag
		}
	}

	return n, nil
}

func (s *Server) DeleteHandler(c *gin.Context) {
	var r api.DeleteRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseNormalizePullModelRef(cmp.Or(r.Model, r.Name))
	if err != nil {
		switch {
		case errors.Is(err, errConflictingModelSource):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		case errors.Is(err, model.ErrUnqualifiedName):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("name %q is invalid", cmp.Or(r.Model, r.Name))})
		default:
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}

	n, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		return
	}

	m, err := manifest.ParseNamedManifest(n)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if err := m.Remove(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	s.deleteModelListCache(n)

	if err := m.RemoveLayers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func (s *Server) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	requestedModel := req.Model

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, err.Error())
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		if modelShowCacheable(req) && s.modelCaches != nil && s.modelCaches.show != nil {
			if disabled, _ := internalcloud.Status(); disabled {
				c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(cloudErrRemoteModelDetailsUnavailable)})
				return
			}

			ctx := context.Background()
			if c.Request != nil {
				ctx = c.Request.Context()
			}
			if resp, ok := s.modelCaches.show.GetCloudSWR(ctx, req); ok {
				c.JSON(http.StatusOK, resp)
				return
			}
		}
		proxyCloudJSONRequest(c, req, cloudErrRemoteModelDetailsUnavailable)
		return
	}

	name := modelRef.Name
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req.Model = name.DisplayShortest()

	var resp *api.ShowResponse
	if modelShowCacheable(req) && s.modelCaches != nil && s.modelCaches.show != nil {
		resp, err = s.modelCaches.show.GetLocal(req)
	} else {
		resp, err = GetModelInfo(req)
	}
	if err != nil {
		var statusErr api.StatusError
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case errors.As(err, &statusErr):
			c.JSON(statusErr.StatusCode, gin.H{"error": statusErr.ErrorMessage})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if modelRef.Source == modelSourceLocal && resp.RemoteHost != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", modelRef.Original)})
		return
	}

	userAgent := c.Request.UserAgent()
	if strings.HasPrefix(userAgent, copilotChatUserAgentPrefix) {
		if resp.ModelInfo == nil {
			resp.ModelInfo = map[string]any{}
		}
		// Copilot Chat prefers `general.basename`, but this is usually not what
		// users are familiar with, so echo back the requested model name.
		resp.ModelInfo["general.basename"] = requestedModel
	}

	c.JSON(http.StatusOK, resp)
}

func GetModelInfo(req api.ShowRequest) (*api.ShowResponse, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, model.Unqualified(name)
	}
	name, err := getExistingName(name)
	if err != nil {
		return nil, err
	}

	m, err := GetModel(name.String())
	if err != nil {
		return nil, err
	}

	if m.Config.RemoteHost != "" {
		if disabled, _ := internalcloud.Status(); disabled {
			return nil, api.StatusError{
				StatusCode:   http.StatusForbidden,
				ErrorMessage: internalcloud.DisabledError(cloudErrRemoteModelDetailsUnavailable),
			}
		}
	}

	modelDetails := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.Config.ModelFamily,
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}

	// For image generation models, populate details from imagegen package
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		if info, err := imagegenmanifest.GetModelInfo(name.String()); err == nil {
			modelDetails.Family = info.Architecture
			modelDetails.ParameterSize = format.HumanNumber(uint64(info.ParameterCount))
			modelDetails.QuantizationLevel = info.Quantization
		}
	}

	// For safetensors LLM models (experimental), populate details from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			if arch, ok := info["general.architecture"].(string); ok && arch != "" {
				modelDetails.Family = arch
			}
			if paramCount, ok := info["general.parameter_count"].(int64); ok && paramCount > 0 {
				modelDetails.ParameterSize = format.HumanNumber(uint64(paramCount))
			}
		}
		// Older manifests may not have file_type populated for safetensors models.
		if modelDetails.QuantizationLevel == "" {
			if dtype, err := xserver.GetSafetensorsDtype(name); err == nil && dtype != "" {
				modelDetails.QuantizationLevel = dtype
			}
		}
	}

	if req.System != "" {
		m.System = req.System
	}

	msgs := make([]api.Message, len(m.Messages))
	for i, msg := range m.Messages {
		msgs[i] = api.Message{Role: msg.Role, Content: msg.Content}
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, err
	}

	resp := &api.ShowResponse{
		License:      strings.Join(m.License, "\n"),
		System:       m.System,
		Template:     m.Template.String(),
		Details:      modelDetails,
		Messages:     msgs,
		Capabilities: m.Capabilities(),
		ModifiedAt:   mf.FileInfo().ModTime(),
		Requires:     m.Config.Requires,
		// Several integrations crash on a nil/omitempty+empty ModelInfo, so by
		// default we return an empty map.
		ModelInfo: make(map[string]any),
	}

	if m.Config.RemoteHost != "" {
		resp.RemoteHost = m.Config.RemoteHost
		resp.RemoteModel = m.Config.RemoteModel

		if m.Config.ModelFamily != "" {
			resp.ModelInfo = make(map[string]any)
			resp.ModelInfo["general.architecture"] = m.Config.ModelFamily

			if m.Config.BaseName != "" {
				resp.ModelInfo["general.basename"] = m.Config.BaseName
			}

			if m.Config.ContextLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.context_length", m.Config.ModelFamily)] = m.Config.ContextLen
			}

			if m.Config.EmbedLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.embedding_length", m.Config.ModelFamily)] = m.Config.EmbedLen
			}
		}
	}

	var params []string
	cs := 30
	for k, v := range m.Options {
		switch val := v.(type) {
		case []any:
			for _, nv := range val {
				params = append(params, fmt.Sprintf("%-*s %#v", cs, k, nv))
			}
		default:
			params = append(params, fmt.Sprintf("%-*s %#v", cs, k, v))
		}
	}
	resp.Parameters = strings.Join(params, "\n")

	if len(req.Options) > 0 {
		if m.Options == nil {
			m.Options = make(map[string]any)
		}
		for k, v := range req.Options {
			m.Options[k] = v
		}
	}

	var sb strings.Builder
	fmt.Fprintln(&sb, "# Modelfile generated by \"ollama show\"")
	modelfile := m.String()
	if m.IsMLX() {
		fmt.Fprintf(&sb, "FROM %s\n", m.ShortName)
		if _, rest, ok := strings.Cut(modelfile, "\n"); ok {
			fmt.Fprint(&sb, rest)
		}
	} else {
		fmt.Fprintln(&sb, "# To build a new Modelfile based on this, replace FROM with:")
		fmt.Fprintf(&sb, "# FROM %s\n\n", m.ShortName)
		fmt.Fprint(&sb, modelfile)
	}
	resp.Modelfile = sb.String()

	// skip loading tensor information if this is a remote model
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		return resp, nil
	}

	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	// For safetensors LLM models (experimental), populate ModelInfo from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name); err == nil {
			resp.ModelInfo = info
		}
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	kvData, tensors, err := getModelData(m.ModelPath, req.Verbose)
	if err != nil {
		return nil, err
	}

	resp.Template = selectedModelTemplate(m, kvData)
	if isUnknownQuantization(resp.Details.QuantizationLevel) {
		if fileType := kvData.FileType().String(); !isUnknownQuantization(fileType) {
			resp.Details.QuantizationLevel = fileType
		}
	}

	delete(kvData, "general.name")
	delete(kvData, "tokenizer.chat_template")
	resp.ModelInfo = kvData

	tensorData := make([]api.Tensor, len(tensors.Items()))
	for cnt, t := range tensors.Items() {
		tensorData[cnt] = api.Tensor{Name: t.Name, Type: t.Type(), Shape: t.Shape}
	}
	resp.Tensors = tensorData

	if len(m.ProjectorPaths) > 0 {
		projectorData, _, err := getModelData(m.ProjectorPaths[0], req.Verbose)
		if err != nil {
			return nil, err
		}
		resp.ProjectorInfo = projectorData
	}

	return resp, nil
}

func getModelData(digest string, verbose bool) (ggml.KV, ggml.Tensors, error) {
	maxArraySize := 0
	if verbose {
		maxArraySize = -1
	}
	data, err := llm.LoadModel(digest, maxArraySize)
	if err != nil {
		return nil, ggml.Tensors{}, err
	}

	kv := data.KV()

	if !verbose {
		for k := range kv {
			if t, ok := kv[k].([]any); len(t) > 5 && ok {
				kv[k] = []any{}
			}
		}
	}

	return kv, data.Tensors(), nil
}

func selectedModelTemplate(m *Model, kv ggml.KV) string {
	if m.HasChatTemplate && chatModeForModel(m) == chatExecutionModeNative {
		if chatTemplate := kv.String("tokenizer.chat_template"); chatTemplate != "" {
			return chatTemplate
		}
	}
	return m.Template.String()
}

func (s *Server) ListHandler(c *gin.Context) {
	if s.modelCaches == nil || s.modelCaches.modelList == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "model list cache unavailable"})
		return
	}

	models, err := s.modelCaches.modelList.List(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func (s *Server) CopyHandler(c *gin.Context) {
	var r api.CopyRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src := model.ParseName(r.Source)
	if !src.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("source %q is invalid", r.Source)})
		return
	}
	src, err := getExistingName(src)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	dst := model.ParseName(r.Destination)
	if !dst.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("destination %q is invalid", r.Destination)})
		return
	}
	dst, err = getExistingName(dst)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := CopyModel(src, dst); errors.Is(err, os.ErrNotExist) {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", r.Source)})
	} else if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	} else {
		s.refreshModelListCache(dst)
	}
}

func (s *Server) HeadBlobHandler(c *gin.Context) {
	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if _, err := os.Stat(path); err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("blob %q not found", c.Param("digest"))})
		return
	}

	c.Status(http.StatusOK)
}

func (s *Server) CreateBlobHandler(c *gin.Context) {
	if ib, ok := intermediateBlobs[c.Param("digest")]; ok {
		p, err := manifest.BlobsPath(ib)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, c.Param("digest"))
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		} else {
			c.Status(http.StatusOK)
			return
		}
	}

	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	default:
		c.Status(http.StatusOK)
		return
	}

	layer, err := manifest.NewLayer(c.Request.Body, "")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if layer.Digest != c.Param("digest") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("digest mismatch, expected %q, got %q", c.Param("digest"), layer.Digest)})
		return
	}

	c.Status(http.StatusCreated)
}

func isLocalIP(ip netip.Addr) bool {
	if interfaces, err := net.Interfaces(); err == nil {
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}

			for _, a := range addrs {
				if parsed, _, err := net.ParseCIDR(a.String()); err == nil {
					if parsed.String() == ip.String() {
						return true
					}
				}
			}
		}
	}

	return false
}

func allowedHost(host string) bool {
	host = strings.ToLower(host)

	if host == "" || host == "localhost" {
		return true
	}

	if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
		return true
	}

	tlds := []string{
		"localhost",
		"local",
		"internal",
	}

	// check if the host is a local TLD
	for _, tld := range tlds {
		if strings.HasSuffix(host, "."+tld) {
			return true
		}
	}

	return false
}

func allowedHostsMiddleware(addr net.Addr) gin.HandlerFunc {
	return func(c *gin.Context) {
		if addr == nil {
			c.Next()
			return
		}

		if addr, err := netip.ParseAddrPort(addr.String()); err == nil && !addr.Addr().IsLoopback() {
			c.Next()
			return
		}

		host, _, err := net.SplitHostPort(c.Request.Host)
		if err != nil {
			host = c.Request.Host
		}

		if addr, err := netip.ParseAddr(host); err == nil {
			if addr.IsLoopback() || addr.IsPrivate() || addr.IsUnspecified() || isLocalIP(addr) {
				c.Next()
				return
			}
		}

		if allowedHost(host) {
			if c.Request.Method == http.MethodOptions {
				c.AbortWithStatus(http.StatusNoContent)
				return
			}

			c.Next()
			return
		}

		c.AbortWithStatus(http.StatusForbidden)
	}
}

func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",

		// OpenAI compatibility headers
		"OpenAI-Beta",
		"x-stainless-arch",
		"x-stainless-async",
		"x-stainless-custom-poll-interval",
		"x-stainless-helper-method",
		"x-stainless-lang",
		"x-stainless-os",
		"x-stainless-package-version",
		"x-stainless-poll-helper",
		"x-stainless-retry-count",
		"x-stainless-runtime",
		"x-stainless-runtime-version",
		"x-stainless-timeout",
	}
	corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true
	r.Use(
		cors.New(corsConfig),
		allowedHostsMiddleware(s.addr),
	)

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/status", s.StatusHandler)

	// Local model cache management (new implementation is at end of function)
	r.POST("/api/pull", s.PullHandler)
	r.POST("/api/push", s.PushHandler)
	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.DELETE("/api/delete", s.DeleteHandler)

	r.POST("/api/me", s.WhoamiHandler)

	r.POST("/api/signout", s.SignoutHandler)
	// deprecated
	r.DELETE("/api/user/keys/:encodedKey", s.SignoutHandler)

	// Create
	r.POST("/api/create", s.CreateHandler)
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)
	r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)
	r.POST("/api/copy", s.CopyHandler)
	r.POST("/api/experimental/web_search", s.WebSearchExperimentalHandler)
	r.POST("/api/experimental/web_fetch", s.WebFetchExperimentalHandler)
	r.GET("/api/experimental/model-recommendations", s.ModelRecommendationsExperimentalHandler)

	// Inference
	r.GET("/api/ps", s.PsHandler)
	// 注册 /api/generate 的 POST 路由，把请求交给 s.GenerateHandler 处理，并可选加上推理请求日志中间件。
	r.POST("/api/generate", s.withInferenceRequestLogging("/api/generate", s.GenerateHandler)...)
	r.POST("/api/chat", s.withInferenceRequestLogging("/api/chat", s.ChatHandler)...)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)

	// Inference (OpenAI compatibility)
	// TODO(cloud-stage-a): apply Modelfile overlay deltas for local models with cloud
	// parents on v1 request families while preserving this explicit :cloud passthrough.
	r.POST("/v1/chat/completions", s.withInferenceRequestLogging("/v1/chat/completions", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.ChatMiddleware(), s.ChatHandler)...)
	r.POST("/v1/completions", s.withInferenceRequestLogging("/v1/completions", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.CompletionsMiddleware(), s.GenerateHandler)...)
	r.POST("/v1/embeddings", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.EmbeddingsMiddleware(), s.EmbedHandler)
	r.GET("/v1/models", middleware.ListMiddleware(), s.ListHandler)
	r.GET("/v1/models/:model", cloudModelPathPassthroughMiddleware(cloudErrRemoteModelDetailsUnavailable), middleware.RetrieveMiddleware(), s.ShowHandler)
	r.POST("/v1/responses", s.withInferenceRequestLogging("/v1/responses", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.ResponsesMiddleware(), s.ChatHandler)...)
	// OpenAI-compatible image generation endpoints
	r.POST("/v1/images/generations", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.ImageGenerationsMiddleware(), s.GenerateHandler)
	r.POST("/v1/images/edits", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.ImageEditsMiddleware(), s.GenerateHandler)
	// OpenAI-compatible audio endpoint
	r.POST("/v1/audio/transcriptions", middleware.TranscriptionMiddleware(), s.ChatHandler)

	// Inference (Anthropic compatibility)
	r.POST("/v1/messages", s.withInferenceRequestLogging("/v1/messages", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.AnthropicMessagesMiddleware(), s.ChatHandler)...)

	if rc != nil {
		// wrap old with new
		rs := &registry.Local{
			Client:   rc,
			Logger:   slog.Default(), // TODO(bmizerany): Take a logger, do not use slog.Default()
			Fallback: r,

			Prune: PruneLayers,
		}
		return rs, nil
	}

	return r, nil
}

func (s *Server) ModelRecommendationsExperimentalHandler(c *gin.Context) {
	recs := defaultModelRecommendations
	source := "default"
	if s.modelCaches != nil && s.modelCaches.recommendations != nil {
		ctx := context.Background()
		if c.Request != nil {
			ctx = c.Request.Context()
		}
		recs = s.modelCaches.recommendations.GetSWR(ctx)
		source = "cache"
	}

	slog.Debug("serving model recommendations", "recommendation_source", source, "count", len(recs))
	c.JSON(http.StatusOK, api.ModelRecommendationsResponse{
		Recommendations: recs,
	})
}

// 定义 Serve(ln net.Listener) error。ln 是外面 RunServer 已经创建好的 TCP listener。
// Serve 做的是：初始化日志和存储，清理模型文件，创建 Server 和路由，启动 scheduler/cache，
// 探测 GPU，最后启动 HTTP 服务并处理优雅退出。
func Serve(ln net.Listener) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("server config", "env", envconfig.Values())
	cloudDisabled, _ := internalcloud.Status() // 读取 Ollama Cloud 是否禁用，并打印状态。
	slog.Info(fmt.Sprintf("Ollama cloud disabled: %t", cloudDisabled))

	blobsDir, err := manifest.BlobsPath("") // 获取模型 blob 存储目录。如果路径获取失败，直接返回错误。
	if err != nil {
		return err
	}
	if err := fixBlobs(blobsDir); err != nil { // 调用 fixBlobs(blobsDir) 修复/整理 blob 存储中的兼容性问题。失败则停止启动。
		return err
	}

	if !envconfig.NoPrune() {
		if _, err := manifest.Manifests(false); err != nil {
			slog.Warn("corrupt manifests detected, skipping prune operation.  Re-pull or delete to clear", "error", err)
		} else {
			// clean up unused layers and manifests
			if err := PruneLayers(); err != nil {
				return err
			}

			manifestsPath, err := manifest.Path()
			if err != nil {
				return err
			}

			if err := manifest.PruneDirectory(manifestsPath); err != nil {
				return err
			}
		}
	}

	s := &Server{
		addr:        ln.Addr(),
		modelCaches: newModelCaches(),
	}
	if err := s.initRequestLogging(); err != nil {
		return err
	}

	var rc *ollama.Registry
	if useClient2 {
		var err error
		rc, err = ollama.DefaultRegistry()
		if err != nil {
			return err
		}
	}

	// 调用 s.GenerateRoutes(rc) 生成 HTTP 路由处理器。
	// 这里会注册 /api/generate、/api/chat、/api/tags、OpenAI 兼容接口等
	h, err := s.GenerateRoutes(rc)
	if err != nil {
		return err
	}
	// 把刚生成的路由挂到 Go 默认 http.DefaultServeMux 的 / 路径上。
	http.Handle("/", h)
	// 创建两个可取消 context：
	// ctx 是 server 总生命周期；schedCtx 是模型调度器生命周期。
	ctx, done := context.WithCancel(context.Background())
	schedCtx, schedDone := context.WithCancel(ctx)
	// 初始化模型调度器，并挂到 s.sched。后面推理请求会通过它加载/复用 runner。
	sched := InitScheduler(schedCtx)
	s.sched = sched
	// 启动模型缓存后台任务。
	s.modelCaches.Start(ctx)
	// 打印监听地址和 Ollama 版本，例如 Listening on 127.0.0.1:11434。
	slog.Info(fmt.Sprintf("Listening on %s (version %s)", ln.Addr(), version.Version))
	// 创建 http.Server。Handler: nil 表示使用默认 http.DefaultServeMux，
	// 所以前面的 http.Handle("/", h) 会生效。注释里说这样也能顺带暴露 net/http/pprof。
	srvr := &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		//
		// TODO(bmizerany): Decide if we want to make this
		// configurable so it is not exposed by default, or allow
		// users to bind it to a different port. This was a quick
		// and easy way to get pprof, but it may not be the best
		// way.
		Handler: nil,
	}

	// 创建系统信号 channel，并监听 SIGINT、SIGTERM，也就是 Ctrl+C 或进程终止信号。
	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	// 启动一个 goroutine 等待退出信号。
	// 收到信号后：关闭 HTTP server、停止 scheduler context、卸载所有模型 runner、取消总 context。
	go func() {
		<-signals
		srvr.Close()
		schedDone()
		sched.unloadAllRunners()
		done()
	}()

	// 启动一个 goroutine 等待退出信号。
	// 收到信号后：关闭 HTTP server、停止 scheduler context、卸载所有模型 runner、取消总 context。
	s.sched.Run(schedCtx)

	// 注册 WebP 图片解码器，让多模态输入可以使用 WebP 图片。
	// register the experimental webp decoder
	// so webp images can be used in multimodal inputs
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)

	// 启动时探测 GPU 信息，并打印 GPU 详情或警告。
	// At startup we retrieve GPU information so we can get log messages before loading a model
	// This will log warnings to the log in case we have problems with detected GPUs
	gpus := discover.GPUDevices(ctx, nil)
	discover.LogDetails(gpus)

	// 统计总 VRAM，扣掉 OLLAMA_GPU_OVERHEAD 这类预留开销。
	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory - envconfig.GpuOverhead()
	}

	// 根据总 VRAM 设置默认上下文长度：
	// 大显存约 47GiB 以上用 262144，约 23GiB 以上用 32768，否则默认 4096。

	// Set default context based on VRAM tier
	// Use slightly lower thresholds (47/23 GiB vs. 48/24 GiB) to account for small differences in the exact value
	switch {
	case totalVRAM >= 47*format.GibiByte:
		s.defaultNumCtx = 262144
	case totalVRAM >= 23*format.GibiByte:
		s.defaultNumCtx = 32768
	default:
		s.defaultNumCtx = 4096
	}
	// 打印基于 VRAM 计算出来的默认上下文长度。
	slog.Info("vram-based default context", "total_vram", format.HumanBytes2(totalVRAM), "default_num_ctx", s.defaultNumCtx)

	// 真正启动 HTTP server，开始阻塞监听请求。这里会接管前面传入的 ln。
	err = srvr.Serve(ln)

	// 如果 Serve 返回的不是 http.ErrServerClosed，说明不是正常关闭，直接返回错误。
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	// 如果是正常关闭，则等待 ctx.Done()，确保信号处理里的清理逻辑完成。
	<-ctx.Done()
	// 返回 nil，表示 server 正常退出。
	return nil
}

func waitForStream(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/json")
	var latest api.ProgressResponse
	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			latest = r
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			errorMsg, ok := r["error"].(string)
			if !ok {
				errorMsg = "unknown error"
			}
			c.JSON(status, gin.H{"error": errorMsg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
			return
		}
	}

	c.JSON(http.StatusOK, latest)
}

func streamResponse(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		// errors are provided as a gin.H with an "error" field and
		// an optional "status" field.  For errors that are streamed
		// before any content, we need to set the status code and
		// content type for the error.
		if h, ok := val.(gin.H); ok {
			if e, ok := h["error"].(string); ok {
				status, ok := h["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				if !c.Writer.Written() {
					c.Header("Content-Type", "application/json")
					c.JSON(status, gin.H{"error": e})
				} else {
					if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
						slog.Error("streamResponse failed to encode json error", "error", err)
					}
				}

				return false
			}
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}

		return true
	})
}

func (s *Server) StatusHandler(c *gin.Context) {
	disabled, source := internalcloud.Status()
	c.JSON(http.StatusOK, api.StatusResponse{
		Cloud: api.CloudStatus{
			Disabled: disabled,
			Source:   source,
		},
	})
}

func (s *Server) WebSearchExperimentalHandler(c *gin.Context) {
	s.webExperimentalProxyHandler(c, "/api/web_search", cloudErrWebSearchUnavailable)
}

func (s *Server) WebFetchExperimentalHandler(c *gin.Context) {
	s.webExperimentalProxyHandler(c, "/api/web_fetch", cloudErrWebFetchUnavailable)
}

func (s *Server) webExperimentalProxyHandler(c *gin.Context, proxyPath, disabledOperation string) {
	body, err := readRequestBody(c.Request)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}

	proxyCloudRequestWithPath(c, body, proxyPath, disabledOperation)
}

func (s *Server) WhoamiHandler(c *gin.Context) {
	// todo allow other hosts
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	user, err := client.Whoami(c)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized {
			// Preserve an actionable sign-in response for launch; other failures
			// below mean account or plan verification is temporarily unavailable.
			sURL := authErr.SigninURL
			if sURL == "" {
				var sErr error
				sURL, sErr = signinURL()
				if sErr != nil {
					slog.Error(sErr.Error())
					c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
					return
				}
			}
			c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": sURL})
			return
		}

		slog.Error(err.Error())
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "account unavailable"})
		return
	}

	if user == nil || user.Name == "" {
		sURL, sErr := signinURL()
		if sErr != nil {
			slog.Error(sErr.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
			return
		}

		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": sURL})
		return
	}

	if strings.TrimSpace(user.Plan) == "" {
		slog.Warn("account plan was not set; defaulting to free")
		user.Plan = "free"
	}
	c.JSON(http.StatusOK, user)
}

func (s *Server) SignoutHandler(c *gin.Context) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		slog.Error("couldn't get public key", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))

	// todo allow other hosts
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	err = client.Disconnect(c, encKey)
	if err != nil {
		var authError api.AuthorizationError
		if errors.As(err, &authError) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "you are not currently signed in"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	c.JSON(http.StatusOK, nil)
}

func (s *Server) PsHandler(c *gin.Context) {
	models := []api.ProcessModelResponse{}

	for _, v := range s.sched.loaded {
		m := v.model
		displayName := model.ParseName(m.ShortName).DisplayShortest()
		modelDetails := api.ModelDetails{
			Format:            m.Config.ModelFormat,
			Family:            m.Config.ModelFamily,
			Families:          m.Config.ModelFamilies,
			ParameterSize:     m.Config.ModelType,
			QuantizationLevel: m.Config.FileType,
		}

		mr := api.ProcessModelResponse{
			Model:     displayName,
			Name:      displayName,
			Size:      int64(v.totalSize),
			SizeVRAM:  int64(v.vramSize),
			Digest:    m.Digest,
			Details:   modelDetails,
			ExpiresAt: v.expiresAt,
		}
		if v.llama != nil {
			mr.ContextLength = v.llama.ContextLength()
			total, vram := v.llama.MemorySize()
			mr.Size = int64(total)
			mr.SizeVRAM = int64(vram)
		}
		// The scheduler waits to set expiresAt, so if a model is loading it's
		// possible that it will be set to the unix epoch. For those cases, just
		// calculate the time w/ the sessionDuration instead.
		var epoch time.Time
		if v.expiresAt == epoch {
			mr.ExpiresAt = time.Now().Add(v.sessionDuration)
		}

		models = append(models, mr)
	}

	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		// longest duration remaining listed first
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})

	c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}

func toolCallId() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

func preservedTokensForCompletion(builtinParser parsers.Parser) []string {
	if builtinParser != nil {
		return builtinParser.PreservedTokens()
	}
	return nil
}

func toolCallTagForCompletion(toolParser *tools.Parser) string {
	if toolParser == nil {
		return ""
	}
	return toolParser.Tag()
}

func leadingBOSForModel(m *Model) string {
	if m == nil || m.Config.Renderer == "" {
		return ""
	}

	return renderers.LeadingBOSForRenderer(resolveRendererName(m))
}

func optionsForPrompt(opts *api.Options, runner llm.LlamaServer) *api.Options {
	if opts == nil || runner == nil {
		return opts
	}

	if ctxLen := runner.ContextLength(); ctxLen > 0 && opts.NumCtx > ctxLen {
		copied := *opts
		copied.NumCtx = ctxLen
		return &copied
	}

	return opts
}

type chatExecutionMode int

const (
	chatExecutionModeNative chatExecutionMode = iota
	chatExecutionModeRendered
)

func chatModeForModel(m *Model) chatExecutionMode {
	if m.IsMLX() || usesOllamaRenderedChat(m) {
		return chatExecutionModeRendered
	}

	return chatExecutionModeNative
}

func llamaServerConfigForModel(m *Model) llm.LlamaServerConfig {
	return llm.LlamaServerConfig{
		DisableJinja:   usesOllamaRenderedChat(m),
		DraftModelPath: m.DraftPath,
	}
}

func usesOllamaRenderedChat(m *Model) bool {
	return m != nil && (m.Config.Renderer != "" || m.Config.Parser != "" || shouldUseHarmony(m) || shouldUseGoTemplate(m))
}

func shouldUseGoTemplate(m *Model) bool {
	if !m.HasGoTemplate {
		return false
	}
	if goTemplateEnvSet() {
		return envconfig.GoTemplate(true)
	}

	return !m.PreferChatTemplate && envconfig.GoTemplate(true)
}

func writeChatResponse(c *gin.Context, req api.ChatRequest, ch chan any) {
	if req.Stream != nil && !*req.Stream {
		var resp api.ChatResponse
		var toolCalls []api.ToolCall
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.ChatResponse:
				sbThinking.WriteString(t.Message.Thinking)
				sbContent.WriteString(t.Message.Content)
				resp = t
				if len(req.Tools) > 0 {
					toolCalls = append(toolCalls, t.Message.ToolCalls...)
				}
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		resp.Message.Content = sbContent.String()
		resp.Message.Thinking = sbThinking.String()
		resp.Logprobs = allLogprobs

		if len(toolCalls) > 0 {
			resp.Message.ToolCalls = toolCalls
		}

		c.JSON(http.StatusOK, resp)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		if c.GetBool(legacyCloudAnthropicKey) {
			proxyCloudJSONRequestWithPath(c, req, "/api/chat", cloudErrRemoteInferenceUnavailable)
			return
		}
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		if disabled, _ := internalcloud.Status(); disabled {
			c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable)})
			return
		}

		origModel := req.Model

		remoteURL, err := url.Parse(m.Config.RemoteHost)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if !slices.Contains(envconfig.Remotes(), remoteURL.Hostname()) {
			slog.Info("remote model", "remotes", envconfig.Remotes(), "remoteURL", m.Config.RemoteHost, "hostname", remoteURL.Hostname())
			c.JSON(http.StatusBadRequest, gin.H{"error": "this server cannot run this remote model"})
			return
		}

		req.Model = m.Config.RemoteModel
		if req.Options == nil {
			req.Options = map[string]any{}
		}

		var msgs []api.Message
		if len(req.Messages) > 0 {
			msgs = append(m.Messages, req.Messages...)
			if req.Messages[0].Role != "system" && m.System != "" {
				msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
			}
		}

		msgs = filterThinkTags(msgs, m)
		req.Messages = msgs

		for k, v := range m.Options {
			if _, ok := req.Options[k]; !ok {
				req.Options[k] = v
			}
		}

		contentType := "application/x-ndjson"
		if req.Stream != nil && !*req.Stream {
			contentType = "application/json; charset=utf-8"
		}
		c.Header("Content-Type", contentType)

		fn := func(resp api.ChatResponse) error {
			resp.Model = origModel
			resp.RemoteModel = m.Config.RemoteModel
			resp.RemoteHost = m.Config.RemoteHost

			data, err := json.Marshal(resp)
			if err != nil {
				return err
			}

			if _, err = c.Writer.Write(append(data, '\n')); err != nil {
				return err
			}
			c.Writer.Flush()
			return nil
		}

		client := api.NewClient(remoteURL, http.DefaultClient)
		err = client.Chat(c, &req, fn)
		if err != nil {
			var authError api.AuthorizationError
			if errors.As(err, &authError) {
				sURL, sErr := signinURL()
				if sErr != nil {
					slog.Error(sErr.Error())
					c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
					return
				}

				c.JSON(authError.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
				return
			}
			var apiError api.StatusError
			if errors.As(err, &apiError) {
				c.JSON(apiError.StatusCode, apiError)
				return
			}
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if len(req.Tools) > 0 {
		caps = append(caps, model.CapabilityTools)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			// Set think to nil when being used with Anthropic API to connect to tools like claude code
			if _, ok := c.Get("relax_thinking"); ok {
				slog.Warn("model does not support thinking, relaxing thinking to nil", "model", req.Model)
				req.Think = nil
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
				return
			}
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive, req.Shift)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support chat", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	msgs := append(m.Messages, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)

	if shouldUseHarmony(m) {
		// harmony's Reasoning field only understands low/medium/high; map "max" to "high"
		if req.Think != nil {
			if s, ok := req.Think.Value.(string); ok && s == "max" {
				req.Think.Value = "high"
			}
		}
		if m.Config.Parser == "" {
			m.Config.Parser = "harmony"
		}
	}

	if chatModeForModel(m) == chatExecutionModeNative {
		s.handleNativeChat(c, req, m, r, opts, msgs, checkpointStart, checkpointLoaded)
		return
	}

	var builtinParser parsers.Parser
	processedTools := req.Tools

	if m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			// Determine last message for chat prefill
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			// Initialize parser and get processed tools
			processedTools = builtinParser.Init(req.Tools, lastMessage, req.Think)
		}
	}

	truncate := req.Truncate == nil || *req.Truncate
	if m.IsMLX() {
		truncate = false
	}
	promptOpts := optionsForPrompt(opts, r)
	prompt, media, err := chatPrompt(c.Request.Context(), m, r.Tokenize, promptOpts, msgs, processedTools, req.Think, truncate)
	if err != nil {
		slog.Error("chat prompt error", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(media),
			},
		})
		return
	}

	var thinkingState *thinking.Parser
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
		thinkingState = &thinking.Parser{
			OpeningTag: openingTag,
			ClosingTag: closingTag,
		}

		if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
			thinkingState.AddContent(openingTag)
		}
	}

	var toolParser *tools.Parser
	if len(req.Tools) > 0 && (builtinParser == nil || !builtinParser.HasToolSupport()) {
		toolParser = tools.NewParser(m.Template.Template, req.Tools)
	}

	type structuredOutputsState int
	const (
		structuredOutputsState_None structuredOutputsState = iota
		structuredOutputsState_ReadyToApply
		structuredOutputsState_Applying
	)

	ch := make(chan any)
	go func() {
		defer close(ch)

		structuredOutputsState := structuredOutputsState_None

		for {
			var tb strings.Builder

			currentFormat := req.Format
			// structured outputs via double request is enabled when:
			// 1. the model supports the thinking capability and
			// 2. it uses a built-in parser or our generic thinking parser

			// Note that the current approach does not work for (potential future)
			// non-thinking models that emit anything before actual content. This
			// current approach uses the transition from parsed thinking content to
			// parsed non-thinking content as the signal to turn constraining on

			// TODO(parthsareen): temporary fix for https://github.com/ollama/ollama/issues/15260.
			// To revisit for other models and have a consistent pattern across models through parsers.
			forceImmediate := m.Config.Parser == "gemma4" && req.Think != nil && !req.Think.Bool()
			if req.Format != nil && structuredOutputsState == structuredOutputsState_None && !forceImmediate && ((builtinParser != nil || thinkingState != nil) && slices.Contains(m.Capabilities(), model.CapabilityThinking)) {
				currentFormat = nil
			}

			// sets up new context given parent context per request
			ctx, cancel := context.WithCancel(c.Request.Context())

			err := r.Completion(ctx, llm.CompletionRequest{
				Prompt:          prompt,
				Media:           media,
				Format:          currentFormat,
				Options:         opts,
				Shift:           req.Shift == nil || *req.Shift,
				Truncate:        truncate,
				Logprobs:        req.Logprobs,
				TopLogprobs:     req.TopLogprobs,
				PreservedTokens: preservedTokensForCompletion(builtinParser),
				ToolCallTag:     toolCallTagForCompletion(toolParser),
				LeadingBOS:      leadingBOSForModel(m),
			}, func(r llm.CompletionResponse) {
				res := api.ChatResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC(),
					Message:   api.Message{Role: "assistant", Content: r.Content},
					Done:      r.Done,
					Metrics: api.Metrics{
						PromptEvalCount:    r.PromptEvalCount,
						PromptEvalDuration: r.PromptEvalDuration,
						EvalCount:          r.EvalCount,
						EvalDuration:       r.EvalDuration,
					},
					Logprobs: toAPILogprobs(r.Logprobs),
				}

				if r.Done {
					res.DoneReason = r.DoneReason.String()
					res.TotalDuration = time.Since(checkpointStart)
					res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
				}

				if builtinParser != nil {
					slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser input", "parser", m.Config.Parser, "content", r.Content)

					content, thinking, toolCalls, err := builtinParser.Add(r.Content, r.Done)
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}

					res.Message.Content = content
					res.Message.Thinking = thinking
					for i := range toolCalls {
						toolCalls[i].ID = toolCallId()
					}
					res.Message.ToolCalls = toolCalls

					tb.WriteString(thinking)
					// we are now receiving content from the model - we should start applying structured outputs
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && res.Message.Content != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						cancel()
						return
					}

					if res.Message.Content != "" || res.Message.Thinking != "" || len(res.Message.ToolCalls) > 0 || r.Done || len(res.Logprobs) > 0 {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser output", "parser", m.Config.Parser, "content", content, "thinking", thinking, "toolCalls", toolCalls, "done", r.Done)
						ch <- res
					} else {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser empty output", "parser", m.Config.Parser)
					}
					return
				}

				if thinkingState != nil {
					thinkingContent, remainingContent := thinkingState.AddContent(res.Message.Content)
					if thinkingContent == "" && remainingContent == "" && !r.Done {
						// need to accumulate more to decide what to send
						return
					}
					res.Message.Thinking = thinkingContent
					tb.WriteString(thinkingContent)
					// emit the collected thinking text before restarting with structured outputs and clear unstructured content
					// to avoid leaking mixed tokens like "</think>Hello"
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && remainingContent != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						res.Message.Content = ""
						ch <- res
						cancel()
						return
					}
					res.Message.Content = remainingContent
				}

				if len(req.Tools) > 0 {
					toolCalls, content := toolParser.Add(res.Message.Content)
					if len(content) > 0 {
						res.Message.Content = content
					} else if len(toolCalls) > 0 {
						for i := range toolCalls {
							toolCalls[i].ID = toolCallId()
						}
						res.Message.ToolCalls = toolCalls
						res.Message.Content = ""
					} else if res.Message.Thinking != "" {
						// don't return, fall through to send
					} else {
						//  Send logprobs while content is being buffered by the parser for tool calls
						if len(res.Logprobs) > 0 && !r.Done {
							logprobRes := res
							logprobRes.Message.Content = ""
							logprobRes.Message.ToolCalls = nil
							ch <- logprobRes
						}

						if r.Done {
							res.Message.Content = toolParser.Content()
							ch <- res
						}
						return
					}
				}

				ch <- res
			})
			if err != nil {
				if structuredOutputsState == structuredOutputsState_ReadyToApply && strings.Contains(err.Error(), "context canceled") && c.Request.Context().Err() == nil {
					// only ignores error if it's a context cancellation due to setting structured outputs
				} else {
					s.sched.expireRunnersForRuntimeOOM(m, err)
					var serr api.StatusError
					if errors.As(err, &serr) {
						ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
					} else {
						ch <- gin.H{"error": err.Error()}
					}
					return
				}
			}

			// ignored structured outputs cancellation falls through to here, start a new request with the structured outputs and updated prompt. use the
			if structuredOutputsState == structuredOutputsState_ReadyToApply {
				structuredOutputsState = structuredOutputsState_Applying
				msg := api.Message{
					Role:     "assistant",
					Thinking: tb.String(),
				}

				msgs = append(msgs, msg)
				prompt, _, err = chatPrompt(c.Request.Context(), m, r.Tokenize, promptOpts, msgs, processedTools, req.Think, truncate)
				if err != nil {
					slog.Error("chat prompt error applying structured outputs", "error", err)
					ch <- gin.H{"error": err.Error()}
					return
				}
				// force constraining by terminating thinking header, the parser is already at this state
				// when the last message is thinking, the rendered for gpt-oss cannot disambiguate between having the
				// model continue thinking or ending thinking and outputting the final message.
				// TODO(parthsareen): consider adding prefill disambiguation logic to the renderer for structured outputs.
				if shouldUseHarmony(m) || (builtinParser != nil && m.Config.Parser == "harmony") {
					prompt += "<|end|><|start|>assistant<|channel|>final<|message|>"
				}
				continue
			}

			break
		}
	}()

	writeChatResponse(c, req, ch)
}

func (s *Server) handleNativeChat(c *gin.Context, req api.ChatRequest, m *Model, r llm.LlamaServer, opts *api.Options, msgs []api.Message, checkpointStart, checkpointLoaded time.Time) {
	nativeReq := llm.ChatRequest{
		Messages:    msgs,
		Tools:       req.Tools,
		Format:      req.Format,
		Options:     opts,
		Think:       req.Think,
		Shift:       req.Shift == nil || *req.Shift,
		Logprobs:    req.Logprobs,
		TopLogprobs: req.TopLogprobs,
	}
	truncate := req.Truncate == nil || *req.Truncate
	var err error
	nativeReq.Messages, err = truncateNativeChatMessages(c.Request.Context(), m, r, optionsForPrompt(opts, r), nativeReq, truncate)
	if err != nil {
		slog.Error("chat template prompt error", "error", err)
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.JSON(serr.StatusCode, gin.H{"error": serr.ErrorMessage})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.DebugRenderOnly {
		prompt, err := r.ApplyChatTemplate(c.Request.Context(), nativeReq)
		if err != nil {
			var serr api.StatusError
			if errors.As(err, &serr) {
				c.JSON(serr.StatusCode, gin.H{"error": serr.ErrorMessage})
			} else {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			}
			return
		}

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       countChatImages(msgs),
			},
		})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)

		err := r.Chat(c.Request.Context(), nativeReq, func(r llm.ChatResponse) {
			res := api.ChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Message:   r.Message,
				Done:      r.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    r.PromptEvalCount,
					PromptEvalDuration: r.PromptEvalDuration,
					EvalCount:          r.EvalCount,
					EvalDuration:       r.EvalDuration,
				},
				Logprobs: toAPILogprobs(r.Logprobs),
			}

			if res.Message.Role == "" {
				res.Message.Role = "assistant"
			}

			if r.Done {
				res.DoneReason = r.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
			}

			ch <- res
		})
		if err != nil {
			s.sched.expireRunnersForRuntimeOOM(m, err)
			var serr api.StatusError
			if errors.As(err, &serr) {
				ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
			} else {
				ch <- gin.H{"error": err.Error()}
			}
		}
	}()

	writeChatResponse(c, req, ch)
}

func truncateNativeChatMessages(ctx context.Context, m *Model, r llm.LlamaServer, opts *api.Options, req llm.ChatRequest, truncate bool) ([]api.Message, error) {
	if !truncate || opts == nil || opts.NumCtx <= 0 || len(req.Messages) <= 1 {
		return req.Messages, nil
	}

	lastMsgIdx := len(req.Messages) - 1
	currMsgIdx := 0
	var system []api.Message

	for i := 0; i <= lastMsgIdx; i++ {
		system = system[:0]
		for j := range i {
			if req.Messages[j].Role == "system" {
				system = append(system, req.Messages[j])
			}
		}

		renderReq := req
		renderReq.Messages = append(slices.Clone(system), req.Messages[i:]...)
		prompt, err := r.ApplyChatTemplate(ctx, renderReq)
		if err != nil {
			return nil, err
		}

		tokens, err := r.Tokenize(ctx, prompt)
		if err != nil {
			return nil, err
		}

		ctxLen := len(tokens)
		if m != nil && m.ProjectorPaths != nil {
			for _, msg := range renderReq.Messages {
				ctxLen += 768 * len(msg.Images)
			}
		}

		if ctxLen <= opts.NumCtx {
			currMsgIdx = i
			break
		}
		if i == lastMsgIdx {
			currMsgIdx = lastMsgIdx
			break
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating native chat messages which exceed context length", "truncated", currMsgIdx)
	}

	system = system[:0]
	for j := range currMsgIdx {
		if req.Messages[j].Role == "system" {
			system = append(system, req.Messages[j])
		}
	}
	return append(slices.Clone(system), req.Messages[currMsgIdx:]...), nil
}

func countChatImages(msgs []api.Message) int {
	var count int
	for _, msg := range msgs {
		count += len(msg.Images)
	}
	return count
}

func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.Canceled):
		c.JSON(499, gin.H{"error": "request canceled"})
	case errors.Is(err, ErrMaxQueue):
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}

		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				// TODO(drifkin): this is from before we added proper thinking support.
				// However, even if thinking is not enabled (and therefore we shouldn't
				// change the user output), we should probably perform this filtering
				// for all thinking models (not just qwen3 & deepseek-r1) since it tends
				// to save tokens and improve quality.
				thinkingState := &thinking.Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}

// handleImageGenerate handles image generation requests within GenerateHandler.
// This is called when the model has the Image capability.
func (s *Server) handleImageGenerate(c *gin.Context, req api.GenerateRequest, modelName string, checkpointStart time.Time) {
	// Validate image dimensions
	const maxDimension int32 = 4096
	if req.Width > maxDimension || req.Height > maxDimension {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("width and height must be <= %d", maxDimension)})
		return
	}

	// Schedule the runner for image generation
	runner, m, _, err := s.scheduleRunner(c.Request.Context(), modelName, []model.Capability{model.CapabilityImage}, nil, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	// Handle load-only request (empty prompt)
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	// Check streaming preference
	isStreaming := req.Stream == nil || *req.Stream

	contentType := "application/x-ndjson"
	if !isStreaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	// Get seed from options if provided
	var seed int64
	if s, ok := req.Options["seed"]; ok {
		switch v := s.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}

	var media []llm.MediaData
	for i, imgData := range req.Images {
		media = append(media, llm.NewMediaData(i, imgData))
	}

	var streamStarted bool
	var finalResponse api.GenerateResponse

	if err := runner.Completion(c.Request.Context(), llm.CompletionRequest{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  req.Steps,
		Seed:   seed,
		Media:  media,
	}, func(cr llm.CompletionResponse) {
		streamStarted = true
		res := api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			Done:      cr.Done,
		}

		if cr.TotalSteps > 0 {
			res.Completed = int64(cr.Step)
			res.Total = int64(cr.TotalSteps)
		}

		if cr.Image != "" {
			res.Image = cr.Image
		}

		if cr.Done {
			res.DoneReason = cr.DoneReason.String()
			res.Metrics.TotalDuration = time.Since(checkpointStart)
			res.Metrics.LoadDuration = checkpointLoaded.Sub(checkpointStart)
		}

		if !isStreaming {
			finalResponse = res
			return
		}

		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
	}); err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		// Only send JSON error if streaming hasn't started yet
		// (once streaming starts, headers are committed and we can't change status code)
		if !isStreaming || !streamStarted {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		} else {
			data, _ := json.Marshal(gin.H{"error": err.Error()})
			c.Writer.Write(append(data, '\n'))
			c.Writer.Flush()
		}
		return
	}

	if !isStreaming {
		c.JSON(http.StatusOK, finalResponse)
	}
}
