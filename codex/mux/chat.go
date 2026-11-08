package mux

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// The llama.cpp-compatible face for web chat (PROTOCOL.md): POST /v1/chat/completions on the engine and
// shim services, GET /props, and (on the compat listener only) /v1/models as llama-server lists it.
// A chat completion is a /v1/responses run whose events are written as chat completion chunks: the same
// prompt, the same parsing, the same rules.

// resolveModel finds the service a request names: its service name, or the display name web chat shows
// for it (with or without .gguf). No name is the compat service.
func (s *Server) resolveModel(name string) *Service {
	if name == "" {
		name = s.cfg.CompatService
	}
	if e := s.svcs[name]; e != nil {
		return e
	}
	for _, e := range s.order {
		if e.cfg.Display != "" && (name == e.cfg.Display || name == e.cfg.LlamaID()) {
			return e
		}
	}
	return nil
}

func (s *Server) modelNotFound(name string) *apiError {
	names := make([]string, 0, len(s.svcs))
	for n, e := range s.svcs {
		if e.kind != KindJob {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found", Param: "model",
		Message: fmt.Sprintf("model %q is not served here (models: %s)", name, strings.Join(names, ", "))}
}

// handleChatCompletions is POST /v1/chat/completions.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		writeAPIError(w, badRequest("invalid_request", "", "cannot read the request body: %v", err))
		return
	}
	var probe struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &probe) != nil {
		writeAPIError(w, badRequest("invalid_json", "", "the body is not a JSON object"))
		return
	}
	svc := s.resolveModel(probe.Model)
	if svc == nil && probe.Model == "" {
		writeAPIError(w, badRequest("missing_model", "model", "model is required (this host has no compat_service to default to)"))
		return
	}
	if svc == nil || svc.kind == KindJob {
		writeAPIError(w, s.modelNotFound(probe.Model))
		return
	}
	if svc.kind == KindEmbed {
		writeAPIError(w, badRequest("not_a_chat_model", "model", "%s is an embedding model: use POST /v1/embeddings", svc.name))
		return
	}
	pr, aerr := prepareChat(body, svc.cfg)
	if aerr != nil {
		writeAPIError(w, aerr)
		return
	}
	s.dispatch(w, r, svc, pr)
}

// compatService is the service /props and /v1/models describe: the compat service, else the first
// engine that is up. nil if none is up.
func (s *Server) compatService() *Service {
	up := func(e *Service) bool {
		st := e.Info().State
		return st == StateUp || st == StateBusy
	}
	if name := s.cfg.CompatService; name != "" {
		if e := s.svcs[name]; e != nil && up(e) {
			return e
		}
		return nil
	}
	for _, e := range s.order {
		if e.kind == KindEngine && up(e) {
			return e
		}
	}
	return nil
}

// handleProps is GET /props, as llama-server answers it: what web chat needs to know about the model.
func (s *Server) handleProps(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	svc := s.compatService()
	if svc == nil {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "not_found_error", Code: "service_unavailable", Message: "no engine is up"})
		return
	}
	path := svc.cfg.GGUF
	if path == "" {
		path = svc.name
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"default_generation_settings": map[string]any{"n_ctx": svc.cfg.Ctx},
		"modalities":                  map[string]any{"vision": svc.cfg.Vision},
		"chat_template":               "",
		"model_path":                  path,
	})
}

// handleCompatModels is GET /v1/models on the compat listener: the one model, named as llama-server names
// its GGUF. (The main listener's /v1/models is ai-mux's own, with service names as ids.)
func (s *Server) handleCompatModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	data := []map[string]any{}
	if e := s.svcs[s.cfg.CompatService]; e != nil {
		data = append(data, map[string]any{"id": e.cfg.LlamaID(), "object": "model"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// compatHandler is what the compat listener serves.
func (s *Server) compatHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", s.handleCompatModels)
	mux.HandleFunc("/props", s.handleProps)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	return mux
}

// ---- the request ---------------------------------------------------------------------------------------------

type chatToolCallIn struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type chatMsgIn struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"`
	ReasoningContent *string          `json:"reasoning_content"`
	Reasoning        *string          `json:"reasoning"`
	ToolCalls        []chatToolCallIn `json:"tool_calls"`
	ToolCallID       string           `json:"tool_call_id"`
}

// prepareChat validates a /v1/chat/completions body against the service and translates it. A client
// chooses the model and the reasoning effort; max_tokens and max_completion_tokens are ignored (the
// server owns the output budget), as are cache_prompt, stream_options and the other transport fields.
func prepareChat(body []byte, sc *ServiceConfig) (*prepared, *apiError) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, badRequest("invalid_json", "", "the body is not a JSON object: %v", err)
	}
	if aerr := refuseSet(top, steeringFields, chatRefused); aerr != nil {
		return nil, aerr
	}
	p := &prepared{chat: true, model: sc.Name, toolInfo: map[string]*toolInfo{}}
	if v, ok := top["stream"]; ok && !isNull(v) {
		if err := json.Unmarshal(v, &p.stream); err != nil {
			return nil, badRequest("invalid_stream", "stream", "stream must be a boolean")
		}
	}
	// the effort: reasoning_effort or effort; without one, the middle of what the service offers
	for _, key := range []string{"reasoning_effort", "effort"} {
		v, ok := top[key]
		if !ok || isNull(v) {
			continue
		}
		if err := json.Unmarshal(v, &p.effort); err != nil {
			return nil, badRequest("invalid_reasoning_effort", key, "%s must be a string", key)
		}
		if p.effort != "" {
			if aerr := checkEffort(sc, p.effort, key); aerr != nil {
				return nil, aerr
			}
			break
		}
	}
	if p.effort == "" && len(sc.Efforts) > 0 {
		p.effort = sc.Efforts[len(sc.Efforts)/2]
	}

	cb := &chatBuilder{vision: sc.Vision, tools: p.toolInfo}
	var toolsRaw []json.RawMessage
	if v, ok := top["tools"]; ok && !isNull(v) {
		if err := json.Unmarshal(v, &toolsRaw); err != nil {
			return nil, badRequest("invalid_tools", "tools", "tools must be an array")
		}
	}
	dropTools := false
	if v, ok := top["tool_choice"]; ok {
		var s string
		dropTools = json.Unmarshal(v, &s) == nil && s == "none"
	}
	if aerr := cb.addTools(toolsRaw); aerr != nil {
		return nil, aerr
	}

	var msgs []json.RawMessage
	if v, ok := top["messages"]; !ok || isNull(v) {
		return nil, badRequest("missing_messages", "messages", "messages is required")
	} else if err := json.Unmarshal(v, &msgs); err != nil {
		return nil, badRequest("invalid_messages", "messages", "messages must be an array")
	}
	for i, raw := range msgs {
		if aerr := cb.addChatMessage(raw); aerr != nil {
			aerr.Param = fmt.Sprintf("messages[%d]", i)
			return nil, aerr
		}
	}
	if len(cb.msgs) == 0 {
		return nil, badRequest("empty_input", "messages", "messages has no user, assistant or tool message")
	}
	if len(cb.system) > 0 {
		p.messages = append(p.messages, chatMsg{Role: "system", Content: strings.Join(cb.system, "\n\n")})
	}
	p.messages = append(p.messages, cb.msgs...)
	if !dropTools {
		p.tools = cb.chatTools
	}
	p.images = cb.images
	return p, nil
}

// addChatMessage adds one message of a chat completion request. Every system and developer message is
// folded into one leading system message (the chat template wants it first).
func (b *chatBuilder) addChatMessage(raw json.RawMessage) *apiError {
	var m chatMsgIn
	if err := json.Unmarshal(raw, &m); err != nil {
		return badRequest("invalid_message", "", "a message is not an object: %v", err)
	}
	switch m.Role {
	case "system", "developer":
		c, err := b.content(m.Content, false)
		if err != nil {
			return err
		}
		if s := c.(string); strings.TrimSpace(s) != "" {
			b.system = append(b.system, s)
		}
	case "user":
		c, err := b.content(m.Content, true)
		if err != nil {
			return err
		}
		b.msgs = append(b.msgs, chatMsg{Role: "user", Content: c})
	case "assistant":
		c, err := b.content(m.Content, false)
		if err != nil {
			return err
		}
		out := chatMsg{Role: "assistant", Content: c}
		if m.ReasoningContent != nil {
			out.ReasoningContent = strings.TrimSpace(*m.ReasoningContent)
		} else if m.Reasoning != nil {
			out.ReasoningContent = strings.TrimSpace(*m.Reasoning)
		}
		for _, tc := range m.ToolCalls {
			if tc.Function.Name == "" {
				return badRequest("invalid_tool_call", "", "a tool call needs function.name")
			}
			call := chatToolCall{ID: tc.ID, Type: "function"}
			call.Function.Name = tc.Function.Name
			call.Function.Arguments = toArgsObject(tc.Function.Arguments)
			out.ToolCalls = append(out.ToolCalls, call)
		}
		b.msgs = append(b.msgs, out)
	case "tool":
		c, err := b.toolOutput(m.Content)
		if err != nil {
			return err
		}
		b.msgs = append(b.msgs, chatMsg{Role: "tool", ToolCallID: m.ToolCallID, Content: c})
	default:
		return badRequest("unsupported_role", "", "message role %q is not supported", m.Role)
	}
	return nil
}

// ---- the answer ----------------------------------------------------------------------------------------------

type chatUsageOut struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	Details          struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func newChatUsage(in, out, reasoning int) *chatUsageOut {
	u := &chatUsageOut{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out}
	u.Details.ReasoningTokens = reasoning
	return u
}

// responsesUsage reads the usage object a Responses-shaped completion carries.
type responsesUsage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func (u responsesUsage) chat() *chatUsageOut {
	return newChatUsage(u.InputTokens, u.OutputTokens, u.OutputTokensDetails.ReasoningTokens)
}

type chatChoice struct {
	Index        int            `json:"index"`
	Delta        map[string]any `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chatChoice  `json:"choices"`
	Usage   *chatUsageOut `json:"usage,omitempty"`
}

// chatSink writes the events of a response being built as chat completion chunks (plain
// `data: {…}` lines and `data: [DONE]`): reasoning_content, content and tool_calls deltas, then a chunk
// with finish_reason and usage. It ignores the events chat completions has no place for, ai_mux.stats among them.
type chatSink struct {
	*sseSink
	id, model string
	created   int64
	calls     int // tool calls sent so far
	ended     bool
}

func newChatSink(w http.ResponseWriter, model string) *chatSink {
	return &chatSink{sseSink: newSSESink(w), id: newID("chatcmpl-"), model: model, created: time.Now().Unix()}
}

func (c *chatSink) chunk(delta map[string]any, finish *string, usage *chatUsageOut) {
	b, err := json.Marshal(chatChunk{ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
		Choices: []chatChoice{{Index: 0, Delta: delta, FinishReason: finish}}, Usage: usage})
	if err == nil {
		c.put("data: %s\n\n", b)
	}
}

// comment is a keepalive. SSE comments are valid but a hand-written reader may not skip them, so a chat
// stream keeps its connection alive with an empty delta, which every reader of this format handles.
func (c *chatSink) comment(string) {
	if !c.ended {
		c.chunk(map[string]any{}, nil, nil)
	}
}

func (c *chatSink) end() {
	c.ended = true
	c.put("data: [DONE]\n\n")
}

func (c *chatSink) write(typ string, data []byte) {
	if c.ended {
		return
	}
	var ev struct {
		Delta string `json:"delta"`
		Item  struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"item"`
		Response struct {
			Usage             responsesUsage `json:"usage"`
			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Error *struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		} `json:"response"`
	}
	switch typ {
	case "response.created":
		c.chunk(map[string]any{"role": "assistant", "content": ""}, nil, nil)
	case "response.reasoning_text.delta":
		json.Unmarshal(data, &ev)
		c.chunk(map[string]any{"reasoning_content": ev.Delta}, nil, nil)
	case "response.output_text.delta":
		json.Unmarshal(data, &ev)
		c.chunk(map[string]any{"content": ev.Delta}, nil, nil)
	case "response.output_item.done":
		json.Unmarshal(data, &ev)
		if ev.Item.Type == "function_call" {
			c.chunk(map[string]any{"tool_calls": []any{map[string]any{
				"index": c.calls, "id": ev.Item.CallID, "type": "function",
				"function": map[string]any{"name": ev.Item.Name, "arguments": ev.Item.Arguments},
			}}}, nil, nil)
			c.calls++
		}
	case "response.completed", "response.incomplete":
		json.Unmarshal(data, &ev)
		finish := chatFinish(c.calls > 0, typ == "response.incomplete")
		c.chunk(map[string]any{}, &finish, ev.Response.Usage.chat())
		c.end()
	case "response.failed":
		json.Unmarshal(data, &ev)
		code, msg := "server_error", "the response failed"
		if e := ev.Response.Error; e != nil {
			code, msg = e.Code, e.Message
		}
		b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg, "type": "server_error", "code": code}})
		c.put("data: %s\n\n", b)
		c.end()
	}
}

func chatFinish(calls, incomplete bool) string {
	switch {
	case incomplete:
		return "length"
	case calls:
		return "tool_calls"
	}
	return "stop"
}

// chatCompletionObj is the whole answer to a stream:false chat completion, built from the response.
func chatCompletionObj(rb *respBuilder, usage map[string]any, incomplete bool) map[string]any {
	var reasoning, content strings.Builder
	var calls []any
	for _, it := range rb.items {
		switch it["type"] {
		case "reasoning":
			for _, c := range it["content"].([]any) {
				reasoning.WriteString(c.(map[string]any)["text"].(string))
			}
		case "message":
			for _, c := range it["content"].([]any) {
				content.WriteString(c.(map[string]any)["text"].(string))
			}
		case "function_call":
			calls = append(calls, map[string]any{"id": it["call_id"], "type": "function",
				"function": map[string]any{"name": it["name"], "arguments": it["arguments"]}})
		}
	}
	msg := map[string]any{"role": "assistant", "content": content.String()}
	if content.Len() == 0 && len(calls) > 0 {
		msg["content"] = nil
	}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	var u responsesUsage
	b, _ := json.Marshal(usage)
	json.Unmarshal(b, &u)
	return map[string]any{
		"id": newID("chatcmpl-"), "object": "chat.completion", "created": rb.created, "model": rb.model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": chatFinish(len(calls) > 0, incomplete)}},
		"usage":   u.chat(),
	}
}
