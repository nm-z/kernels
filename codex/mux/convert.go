package mux

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// refusedFields steer generation; the server sets them, so a request that sets one is refused.
var refusedFields = []string{
	"temperature", "top_p", "top_k", "seed", "max_output_tokens", "presence_penalty", "frequency_penalty", "stop",
	"max_tokens", "max_completion_tokens", "repetition_penalty", "repeat_penalty", "min_p", "typical_p",
	"logit_bias", "logprobs", "top_logprobs", "n", "best_of", "stop_sequences", "max_tool_calls",
}

// imageData is one input_image: its bytes (from a data: URL) and its hash, in prompt order.
type imageData struct {
	hash  string
	bytes []byte
}

// toolInfo remembers how a model-visible tool maps back to the client's tool.
type toolInfo struct {
	visible   string
	name      string
	namespace string
	custom    bool
	params    map[string][]string // parameter -> its JSON schema types (nil: unknown)
}

// prepared is a request translated to what the template and the run need.
type prepared struct {
	model    string
	stream   bool
	effort   string
	messages []any
	tools    []any
	toolInfo map[string]*toolInfo
	images   []imageData
}

// chatFn / chatTool are the OpenAI chat tool shape, with a stable field order (it is part of the prompt).
type chatFn struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}
type chatTool struct {
	Type     string `json:"type"`
	Function chatFn `json:"function"`
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// chatMsg is an OpenAI chat message in the shape the chat template reads.
type chatMsg struct {
	Role             string         `json:"role"`
	Content          any            `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

func isNull(b json.RawMessage) bool {
	t := bytes.TrimSpace(b)
	return len(t) == 0 || bytes.Equal(t, []byte("null"))
}

// prepare validates a /v1/responses body against the endpoint and translates it.
func prepare(body []byte, ep *Endpoint) (*prepared, *apiError) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, badRequest("invalid_json", "", "the body is not a JSON object: %v", err)
	}
	for _, f := range refusedFields {
		if v, ok := top[f]; ok && !isNull(v) {
			return nil, badRequest("unsupported_parameter", f, "%s is set by the server and cannot be set by a client (a client chooses only the model and the reasoning effort)", f)
		}
	}
	if v, ok := top["previous_response_id"]; ok && !isNull(v) {
		return nil, badRequest("unsupported_parameter", "previous_response_id", "previous_response_id is not supported: ai-mux keeps no responses, send the whole input")
	}
	p := &prepared{toolInfo: map[string]*toolInfo{}}
	if v, ok := top["model"]; ok {
		if err := json.Unmarshal(v, &p.model); err != nil {
			return nil, badRequest("invalid_model", "model", "model must be a string")
		}
	}
	if p.model == "" {
		return nil, badRequest("missing_model", "model", "model is required")
	}
	if v, ok := top["stream"]; ok && !isNull(v) {
		if err := json.Unmarshal(v, &p.stream); err != nil {
			return nil, badRequest("invalid_stream", "stream", "stream must be a boolean")
		}
	}
	// reasoning.effort
	if v, ok := top["reasoning"]; ok && !isNull(v) {
		var r struct {
			Effort *string `json:"effort"`
		}
		if err := json.Unmarshal(v, &r); err != nil {
			return nil, badRequest("invalid_reasoning", "reasoning", "reasoning must be an object")
		}
		if r.Effort != nil && *r.Effort != "" {
			p.effort = *r.Effort
			ok := false
			for _, e := range ep.cfg.Efforts {
				if e == p.effort {
					ok = true
				}
			}
			if !ok {
				return nil, badRequest("unsupported_reasoning_effort", "reasoning.effort",
					"reasoning effort %q is not offered by %s (efforts: %s)", p.effort, ep.name, strings.Join(ep.cfg.Efforts, ", "))
			}
		}
	}

	cb := &chatBuilder{vision: ep.cfg.Vision, tools: p.toolInfo}
	var instr string
	if v, ok := top["instructions"]; ok && !isNull(v) {
		if err := json.Unmarshal(v, &instr); err != nil {
			return nil, badRequest("invalid_instructions", "instructions", "instructions must be a string")
		}
	}
	if strings.TrimSpace(instr) != "" {
		cb.system = append(cb.system, instr)
	}

	var toolsRaw []json.RawMessage
	if v, ok := top["tools"]; ok && !isNull(v) {
		if err := json.Unmarshal(v, &toolsRaw); err != nil {
			return nil, badRequest("invalid_tools", "tools", "tools must be an array")
		}
	}
	dropTools := false
	if v, ok := top["tool_choice"]; ok {
		var s string
		if json.Unmarshal(v, &s) == nil && s == "none" {
			dropTools = true
		}
	}
	if err := cb.addTools(toolsRaw); err != nil {
		return nil, err
	}

	// input: a string, or an array of items
	inRaw, ok := top["input"]
	if !ok || isNull(inRaw) {
		return nil, badRequest("missing_input", "input", "input is required")
	}
	var s string
	if json.Unmarshal(inRaw, &s) == nil {
		cb.flush()
		cb.msgs = append(cb.msgs, chatMsg{Role: "user", Content: s})
	} else {
		var items []json.RawMessage
		if err := json.Unmarshal(inRaw, &items); err != nil {
			return nil, badRequest("invalid_input", "input", "input must be a string or an array of items")
		}
		for i, it := range items {
			if err := cb.addItem(it); err != nil {
				err.Param = fmt.Sprintf("input[%d]", i)
				return nil, err
			}
		}
	}
	cb.flush()
	if len(cb.msgs) == 0 {
		return nil, badRequest("empty_input", "input", "input has no messages")
	}
	msgs := make([]any, 0, len(cb.msgs)+1)
	if len(cb.system) > 0 {
		msgs = append(msgs, chatMsg{Role: "system", Content: strings.Join(cb.system, "\n\n")})
	}
	for _, m := range cb.msgs {
		msgs = append(msgs, m)
	}
	p.messages = msgs
	if !dropTools {
		p.tools = cb.chatTools
	}
	p.images = cb.images
	return p, nil
}

// templateVars are the variables the chat template is rendered with.
func (p *prepared) templateVars() map[string]any {
	v := map[string]any{"messages": p.messages, "add_generation_prompt": true}
	if len(p.tools) > 0 {
		v["tools"] = p.tools
	}
	switch p.effort {
	case "":
	case "none", "minimal":
		v["enable_thinking"] = false
	case "low":
		v["reasoning_effort"] = "low"
	case "medium":
		v["reasoning_effort"] = "medium"
	case "high", "xhigh":
		v["reasoning_effort"] = "xhigh"
	default:
		v["reasoning_effort"] = p.effort
	}
	return v
}

// ---- items -------------------------------------------------------------------------------------

type chatBuilder struct {
	vision    bool
	system    []string
	msgs      []chatMsg
	pending   *chatMsg // an assistant message being assembled from reasoning, message and call items
	images    []imageData
	tools     map[string]*toolInfo
	chatTools []any
}

func (b *chatBuilder) flush() {
	if b.pending == nil {
		return
	}
	m := b.pending
	b.pending = nil
	if m.Content == "" && len(m.ToolCalls) == 0 {
		return // reasoning with nothing after it is not a turn
	}
	b.msgs = append(b.msgs, *m)
}

func (b *chatBuilder) assistant() *chatMsg {
	if b.pending == nil {
		b.pending = &chatMsg{Role: "assistant", Content: ""}
	}
	return b.pending
}

func (b *chatBuilder) addItem(raw json.RawMessage) *apiError {
	var it struct {
		Type      string            `json:"type"`
		Role      string            `json:"role"`
		Content   json.RawMessage   `json:"content"`
		Name      string            `json:"name"`
		Namespace string            `json:"namespace"`
		Arguments json.RawMessage   `json:"arguments"`
		CallID    string            `json:"call_id"`
		Input     json.RawMessage   `json:"input"`
		Output    json.RawMessage   `json:"output"`
		Tools     []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(raw, &it); err != nil {
		return badRequest("invalid_input_item", "", "an input item is not an object: %v", err)
	}
	typ := it.Type
	if typ == "" && it.Role != "" {
		typ = "message"
	}
	switch typ {
	case "message":
		return b.addMessage(it.Role, it.Content)
	case "reasoning":
		text := reasoningText(it.Content)
		if text == "" {
			return nil
		}
		if b.pending != nil && (b.pending.Content != "" || len(b.pending.ToolCalls) > 0 || b.pending.ReasoningContent != "") {
			b.flush()
		}
		b.assistant().ReasoningContent = text
	case "function_call", "custom_tool_call":
		var args json.RawMessage
		if typ == "function_call" {
			args = toArgsObject(it.Arguments)
		} else {
			var in string
			if err := json.Unmarshal(it.Input, &in); err != nil {
				in = string(it.Input)
			}
			args, _ = json.Marshal(map[string]string{"input": in})
		}
		name := visibleName(it.Namespace, it.Name)
		tc := chatToolCall{ID: it.CallID, Type: "function"}
		tc.Function.Name = name
		tc.Function.Arguments = args
		m := b.assistant()
		m.ToolCalls = append(m.ToolCalls, tc)
	case "function_call_output", "custom_tool_call_output":
		b.flush()
		content, err := b.toolOutput(it.Output)
		if err != nil {
			return err
		}
		b.msgs = append(b.msgs, chatMsg{Role: "tool", ToolCallID: it.CallID, Content: content})
	case "additional_tools":
		return b.addTools(it.Tools)
	case "item_reference", "compaction", "compaction_summary", "context_compaction", "compaction_trigger",
		"configuration_update", "web_search_call", "image_generation_call", "tool_search_call", "tool_search_output",
		"local_shell_call", "agent_message":
		// items with no chat-template form: they carry nothing the model can read
	default:
		return badRequest("unsupported_input_item", "", "input item type %q is not supported", it.Type)
	}
	return nil
}

func visibleName(namespace, name string) string {
	if namespace == "" || namespace == "functions" {
		return name
	}
	return namespace + name
}

// toArgsObject makes function_call.arguments (a JSON string holding an object) the object, keeping
// the key order (the template renders parameters in it). Anything else becomes {"arguments": <text>}.
func toArgsObject(raw json.RawMessage) json.RawMessage {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		s = string(raw)
	}
	t := strings.TrimSpace(s)
	if t == "" {
		return json.RawMessage("{}")
	}
	var probe map[string]json.RawMessage
	if json.Unmarshal([]byte(t), &probe) == nil && strings.HasPrefix(t, "{") {
		return json.RawMessage(t)
	}
	b, _ := json.Marshal(map[string]string{"arguments": s})
	return b
}

func reasoningText(raw json.RawMessage) string {
	if isNull(raw) {
		return ""
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range parts {
		sb.WriteString(p.Text)
	}
	return strings.TrimSpace(sb.String())
}

type contentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	ImageURL json.RawMessage `json:"image_url"`
	Refusal  string          `json:"refusal"`
}

// content turns message or tool-output content into what the template reads: a string for text, a
// list of parts when there are images. Images are collected in prompt order.
func (b *chatBuilder) content(raw json.RawMessage, allowImages bool) (any, *apiError) {
	if isNull(raw) {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var parts []contentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, badRequest("invalid_content", "", "content must be a string or an array of parts")
	}
	hasImage := false
	for _, p := range parts {
		if p.Type == "input_image" || p.Type == "image_url" || p.Type == "image" {
			hasImage = true
		}
	}
	if !hasImage {
		var sb strings.Builder
		for _, p := range parts {
			switch p.Type {
			case "input_text", "output_text", "text", "":
				sb.WriteString(p.Text)
			case "refusal":
				sb.WriteString(p.Refusal)
			default:
				return nil, badRequest("unsupported_content_part", "", "content part type %q is not supported", p.Type)
			}
		}
		return sb.String(), nil
	}
	if !allowImages {
		return nil, badRequest("images_not_allowed", "", "images are not allowed here")
	}
	if !b.vision {
		return nil, badRequest("images_not_supported", "", "this endpoint has no vision encoder")
	}
	var out []any
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text", "":
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		case "input_image", "image_url", "image":
			img, err := decodeImage(p.ImageURL)
			if err != nil {
				return nil, err
			}
			b.images = append(b.images, img)
			out = append(out, map[string]any{"type": "image"})
		default:
			return nil, badRequest("unsupported_content_part", "", "content part type %q is not supported", p.Type)
		}
	}
	return out, nil
}

func decodeImage(raw json.RawMessage) (imageData, *apiError) {
	var url string
	if err := json.Unmarshal(raw, &url); err != nil {
		var o struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal(raw, &o); err != nil || o.URL == "" {
			return imageData{}, badRequest("invalid_image", "", "an image needs an image_url")
		}
		url = o.URL
	}
	head, data, ok := strings.Cut(url, ",")
	if !ok || !strings.HasPrefix(head, "data:") || !strings.Contains(head, ";base64") {
		return imageData{}, badRequest("invalid_image", "", "an image must be a data: URL with base64 data")
	}
	bs, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		bs, err = base64.RawStdEncoding.DecodeString(data)
	}
	if err != nil || len(bs) == 0 {
		return imageData{}, badRequest("invalid_image", "", "the image is not valid base64")
	}
	return imageData{hash: hashBytes(bs), bytes: bs}, nil
}

func (b *chatBuilder) addMessage(role string, raw json.RawMessage) *apiError {
	switch role {
	case "system", "developer":
		b.flush()
		c, err := b.content(raw, false)
		if err != nil {
			return err
		}
		if s := c.(string); strings.TrimSpace(s) != "" {
			b.system = append(b.system, s)
		}
	case "user":
		b.flush()
		c, err := b.content(raw, true)
		if err != nil {
			return err
		}
		b.msgs = append(b.msgs, chatMsg{Role: "user", Content: c})
	case "assistant":
		c, err := b.content(raw, false)
		if err != nil {
			return err
		}
		if b.pending != nil && (b.pending.Content != "" || len(b.pending.ToolCalls) > 0) {
			b.flush()
		}
		b.assistant().Content = c
	default:
		return badRequest("unsupported_role", "", "message role %q is not supported", role)
	}
	return nil
}

// toolOutput is a function_call_output's output: text, or parts that may hold images.
func (b *chatBuilder) toolOutput(raw json.RawMessage) (any, *apiError) {
	if isNull(raw) {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var parts []contentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return string(raw), nil // an object or other JSON: the model reads it as text
	}
	hasImage := false
	for _, p := range parts {
		if p.Type == "input_image" || p.Type == "image_url" || p.Type == "image" {
			hasImage = true
		}
	}
	if hasImage {
		return b.content(raw, true)
	}
	var texts []string
	for _, p := range parts {
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "\n"), nil
}

// ---- tools -------------------------------------------------------------------------------------

type rawTool struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Parameters  json.RawMessage   `json:"parameters"`
	Function    *rawTool          `json:"function"`
	Format      json.RawMessage   `json:"format"`
	Tools       []json.RawMessage `json:"tools"`
}

func (b *chatBuilder) addTools(raws []json.RawMessage) *apiError {
	for _, r := range raws {
		var t rawTool
		if err := json.Unmarshal(r, &t); err != nil {
			return badRequest("invalid_tools", "tools", "a tool is not an object")
		}
		if err := b.addTool(&t, ""); err != nil {
			return err
		}
	}
	return nil
}

func (b *chatBuilder) addTool(t *rawTool, namespace string) *apiError {
	switch t.Type {
	case "function", "":
		if t.Function != nil { // the chat shape
			t = t.Function
		}
		if t.Name == "" {
			return badRequest("invalid_tools", "tools", "a function tool needs a name")
		}
		params := t.Parameters
		if isNull(params) {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		vis := visibleName(namespace, t.Name)
		b.tools[vis] = &toolInfo{visible: vis, name: t.Name, namespace: namespace, params: schemaTypes(params)}
		b.chatTools = append(b.chatTools, chatTool{Type: "function", Function: chatFn{Name: vis, Description: t.Description, Parameters: params}})
	case "custom":
		if t.Name == "" {
			return badRequest("invalid_tools", "tools", "a custom tool needs a name")
		}
		desc := "The value of input is the tool's whole input, as text."
		var f struct {
			Type       string `json:"type"`
			Syntax     string `json:"syntax"`
			Definition string `json:"definition"`
		}
		if json.Unmarshal(t.Format, &f) == nil && f.Definition != "" {
			desc = fmt.Sprintf("The tool's whole input, written in this %s grammar:\n%s", f.Syntax, f.Definition)
		}
		// a struct, not a map: the field order is part of the prompt
		type inputProp struct {
			Type        string `json:"type"`
			Description string `json:"description"`
		}
		params, _ := json.Marshal(struct {
			Type       string               `json:"type"`
			Properties map[string]inputProp `json:"properties"`
			Required   []string             `json:"required"`
		}{"object", map[string]inputProp{"input": {"string", desc}}, []string{"input"}})
		vis := visibleName(namespace, t.Name)
		b.tools[vis] = &toolInfo{visible: vis, name: t.Name, namespace: namespace, custom: true, params: map[string][]string{"input": {"string"}}}
		b.chatTools = append(b.chatTools, chatTool{Type: "function", Function: chatFn{Name: vis, Description: t.Description, Parameters: params}})
	case "namespace":
		for _, r := range t.Tools {
			var sub rawTool
			if err := json.Unmarshal(r, &sub); err != nil {
				return badRequest("invalid_tools", "tools", "a tool is not an object")
			}
			if err := b.addTool(&sub, t.Name); err != nil {
				return err
			}
		}
	default:
		// web_search, local_shell, image_generation, tool_search …: the model cannot call them here
	}
	return nil
}

// schemaTypes reads a JSON schema's properties' types.
func schemaTypes(params json.RawMessage) map[string][]string {
	var s struct {
		Properties map[string]struct {
			Type json.RawMessage `json:"type"`
		} `json:"properties"`
	}
	if json.Unmarshal(params, &s) != nil || s.Properties == nil {
		return nil
	}
	out := map[string][]string{}
	for name, p := range s.Properties {
		var one string
		var many []string
		if json.Unmarshal(p.Type, &one) == nil && one != "" {
			out[name] = []string{one}
		} else if json.Unmarshal(p.Type, &many) == nil && len(many) > 0 {
			out[name] = many
		}
	}
	return out
}
