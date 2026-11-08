package mux

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"
)

// The body nmz-chat-gw sends, exactly (it names no model).
const gatewayBody = `{"cache_prompt":true,"max_tokens":1024,"messages":[{"role":"user","content":"hi"}],"stream":true}`

func (e *tenv) at(base, method, path, body string) *http.Response {
	e.t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, _ := http.NewRequest(method, base+path, rd)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	return resp
}

func (e *tenv) getAt(base, path string) (int, string) {
	e.t.Helper()
	resp := e.at(base, "GET", path, "")
	return resp.StatusCode, readBody(e.t, resp)
}

// chatStream is a chat completion stream read to its end.
type chatStream struct {
	chunks []map[string]any
	done   bool     // data: [DONE]
	other  []string // lines that are not `data: …`: event names, comments
}

func readChatStream(t *testing.T, body io.Reader) chatStream {
	t.Helper()
	var cs chatStream
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
		case line == "data: [DONE]":
			cs.done = true
		case strings.HasPrefix(line, "data: "):
			var m map[string]any
			if err := json.Unmarshal([]byte(line[6:]), &m); err != nil {
				t.Fatalf("bad chunk %q: %v", line, err)
			}
			cs.chunks = append(cs.chunks, m)
		default:
			cs.other = append(cs.other, line)
		}
	}
	return cs
}

func (cs chatStream) delta(i int) map[string]any {
	cs2 := cs.chunks[i]["choices"].([]any)
	return cs2[0].(map[string]any)["delta"].(map[string]any)
}

// text joins the string deltas of a field.
func (cs chatStream) text(field string) string {
	var sb strings.Builder
	for i, c := range cs.chunks {
		if c["choices"] == nil {
			continue
		}
		if s, ok := cs.delta(i)[field].(string); ok {
			sb.WriteString(s)
		}
	}
	return sb.String()
}

func (cs chatStream) finish() (string, map[string]any) {
	last := cs.chunks[len(cs.chunks)-1]
	ch := last["choices"].([]any)[0].(map[string]any)
	f, _ := ch["finish_reason"].(string)
	u, _ := last["usage"].(map[string]any)
	return f, u
}

func (cs chatStream) toolCalls() []map[string]any {
	var out []map[string]any
	for i, c := range cs.chunks {
		if c["choices"] == nil {
			continue
		}
		if tcs, ok := cs.delta(i)["tool_calls"].([]any); ok {
			for _, tc := range tcs {
				out = append(out, tc.(map[string]any))
			}
		}
	}
	return out
}

func TestChatCompletionsLikeTheGateway(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "normal", display: "Qwen3.8-27B", compat: true})
	e.waitState("q", StateUp, 10*time.Second)

	resp := e.at(e.compat, "POST", "/v1/chat/completions", gatewayBody)
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status %d %s: %s", resp.StatusCode, resp.Header.Get("Content-Type"), readBody(t, resp))
	}
	cs := readChatStream(t, resp.Body)
	resp.Body.Close()

	// plain chat completions: no event names, no comments, no ai-mux events; ends with [DONE]
	if len(cs.other) != 0 || !cs.done {
		t.Fatalf("other lines %q, [DONE] %v", cs.other, cs.done)
	}
	for _, c := range cs.chunks {
		if c["object"] != "chat.completion.chunk" || c["model"] != "q" || c["id"] != cs.chunks[0]["id"] || !strings.HasPrefix(c["id"].(string), "chatcmpl-") || c["created"] == nil {
			t.Fatalf("chunk %v", c)
		}
	}
	if d := cs.delta(0); d["role"] != "assistant" || d["content"] != "" {
		t.Errorf("first chunk %v", d)
	}
	if got := cs.text("reasoning_content"); got != "The user wants the weather in Paris: caf\u00e9 \u65e5\u672c\u8a9e. I should call the tool." {
		t.Errorf("reasoning_content %q", got)
	}
	if got := cs.text("content"); got != "Let me check." {
		t.Errorf("content %q", got)
	}
	// reasoning comes first, then the content, then the tool calls
	order := ""
	for i := 1; i < len(cs.chunks)-1; i++ {
		d := cs.delta(i)
		switch {
		case d["reasoning_content"] != nil:
			order += "r"
		case d["content"] != nil:
			order += "c"
		case d["tool_calls"] != nil:
			order += "t"
		}
	}
	if !regexp.MustCompile("^r+c+tt$").MatchString(order) {
		t.Errorf("order %q", order)
	}
	tcs := cs.toolCalls()
	if len(tcs) != 2 || tcs[0]["index"] != float64(0) || tcs[1]["index"] != float64(1) || tcs[0]["type"] != "function" || !strings.HasPrefix(tcs[0]["id"].(string), "call_") {
		t.Fatalf("tool calls %v", tcs)
	}
	if f := tcs[0]["function"].(map[string]any); f["name"] != "get_weather" || f["arguments"] != `{"city":"Paris","days":3}` {
		t.Errorf("tool call 0 %v", f)
	}
	finish, usage := cs.finish()
	n := float64(len(mustEncode(fakeAnswer)))
	if finish != "tool_calls" || usage["completion_tokens"] != n || usage["prompt_tokens"].(float64) < 8 ||
		usage["total_tokens"] != usage["prompt_tokens"].(float64)+n || usage["completion_tokens_details"].(map[string]any)["reasoning_tokens"].(float64) < 50 {
		t.Errorf("finish %q usage %v", finish, usage)
	}
	// only the last chunk has a finish_reason
	for i, c := range cs.chunks[:len(cs.chunks)-1] {
		if c["choices"].([]any)[0].(map[string]any)["finish_reason"] != nil || c["usage"] != nil {
			t.Errorf("chunk %d ends early: %v", i, c)
		}
	}

	// the engine got the template the default effort asks for, and the server's output budget, not max_tokens
	prompt := e.readOut("prompt")
	if !strings.Contains(prompt, "[effort:medium]") || !strings.Contains(prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n") {
		t.Errorf("prompt:\n%s", prompt)
	}
	var rj struct {
		MaxNew int `json:"max_new"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	if rj.MaxNew != 8192 {
		t.Errorf("max_new %d: the gateway's max_tokens must not set it", rj.MaxNew)
	}
}

func TestChatCompletionsNonStreaming(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "normal"})
	e.waitState("q", StateUp, 10*time.Second)
	resp := e.post("/v1/chat/completions", `{"model":"q","reasoning_effort":"high","messages":[{"role":"user","content":"weather?"}],`+
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},`+
		`{"type":"function","function":{"name":"apply_patch","parameters":{"type":"object","properties":{"input":{"type":"string"}}}}}]}`)
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("status %d: %s", resp.StatusCode, readBody(t, resp))
	}
	var r struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role             string  `json:"role"`
				Content          *string `json:"content"`
				ReasoningContent string  `json:"reasoning_content"`
				ToolCalls        []struct {
					ID       string                           `json:"id"`
					Type     string                           `json:"type"`
					Function struct{ Name, Arguments string } `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage map[string]any `json:"usage"`
	}
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	if r.Object != "chat.completion" || !strings.HasPrefix(r.ID, "chatcmpl-") || r.Model != "q" || len(r.Choices) != 1 {
		t.Fatalf("response %+v", r)
	}
	m := r.Choices[0]
	if m.FinishReason != "tool_calls" || m.Message.Role != "assistant" || m.Message.Content == nil || *m.Message.Content != "Let me check." ||
		!strings.Contains(m.Message.ReasoningContent, "caf\u00e9") || len(m.Message.ToolCalls) != 2 {
		t.Fatalf("choice %+v", m)
	}
	tc := m.Message.ToolCalls
	if tc[0].Function.Name != "get_weather" || tc[0].Function.Arguments != `{"city":"Paris","days":3}` || tc[0].Type != "function" || !strings.HasPrefix(tc[0].ID, "call_") {
		t.Errorf("tool call 0 %+v", tc[0])
	}
	// apply_patch was offered as a function with a string parameter, so it comes back as one
	if tc[1].Function.Name != "apply_patch" || tc[1].Function.Arguments != `{"input":"*** Begin Patch\n*** End Patch"}` {
		t.Errorf("tool call 1 %+v", tc[1])
	}
	if r.Usage["completion_tokens"] != float64(len(mustEncode(fakeAnswer))) || r.Usage["prompt_tokens"].(float64) < 8 {
		t.Errorf("usage %v", r.Usage)
	}
	if !strings.Contains(e.readOut("prompt"), "[effort:xhigh]") {
		t.Error("effort high should render as xhigh")
	}
}

func TestChatEfforts(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "summary"}, epSpec{name: "two", scenario: "summary", efforts: []string{"low", "high"}})
	e.waitState("q", StateUp, 10*time.Second)
	e.waitState("two", StateUp, 10*time.Second)
	prompts := func() []string { return strings.Split(strings.TrimSuffix(e.readOut("prompt"), "\n====\n"), "\n====\n") }
	run := func(body string) string {
		t.Helper()
		before := len(strings.Split(e.readOut("prompt"), "\n====\n")) - 1
		resp := e.post("/v1/chat/completions", body)
		b := readBody(t, resp)
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", body, resp.StatusCode, b)
		}
		waitFor(t, 3*time.Second, "the run to reach the engine", func() bool { return len(strings.Split(e.readOut("prompt"), "\n====\n"))-1 > before })
		ps := prompts()
		return ps[len(ps)-1]
	}
	msg := `"messages":[{"role":"user","content":"hi"}]`
	for body, want := range map[string]string{
		`{"model":"q",` + msg + `}`:                                        "[effort:medium]",
		`{"model":"q","reasoning_effort":"low",` + msg + `}`:               "[effort:low]",
		`{"model":"q","effort":"xhigh",` + msg + `}`:                       "[effort:xhigh]",
		`{"model":"q","reasoning_effort":"high",` + msg + `}`:              "[effort:xhigh]",
		`{"model":"q","reasoning_effort":"none",` + msg + `}`:              "<think>\n\n</think>\n\n",
		`{"model":"q","reasoning_effort":null,"effort":"low",` + msg + `}`: "[effort:low]",
		`{"model":"two",` + msg + `}`:                                      "[effort:xhigh]", // the middle of [low, high] is its second, high, which the template spells xhigh
	} {
		if p := run(body); !strings.Contains(p, want) {
			t.Errorf("%s: want %q in\n%s", body, want, p)
		}
	}
	for _, body := range []string{
		`{"model":"q","reasoning_effort":"max",` + msg + `}`,
		`{"model":"q","effort":"max",` + msg + `}`,
		`{"model":"two","reasoning_effort":"medium",` + msg + `}`,
	} {
		resp := e.post("/v1/chat/completions", body)
		b := readBody(t, resp)
		if resp.StatusCode != 400 || !strings.Contains(b, `"code":"unsupported_reasoning_effort"`) || !strings.Contains(b, "efforts:") {
			t.Errorf("%s: %d %s", body, resp.StatusCode, b)
		}
	}
}

func TestChatRefusalsAndIgnored(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "summary"})
	e.waitState("q", StateUp, 10*time.Second)
	msg := `"model":"q","messages":[{"role":"user","content":"hi"}]`
	for _, f := range []string{"temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "stop", "n", "logit_bias", "response_format"} {
		v := "1"
		switch f {
		case "stop":
			v = `["x"]`
		case "logit_bias":
			v = `{"1":2}`
		case "response_format":
			v = `{"type":"json_object"}`
		}
		resp := e.post("/v1/chat/completions", `{`+msg+`,"`+f+`":`+v+`}`)
		b := readBody(t, resp)
		if resp.StatusCode != 400 || !strings.Contains(b, `"code":"unsupported_parameter"`) || !strings.Contains(b, `"param":"`+f+`"`) {
			t.Errorf("%s: %d %s", f, resp.StatusCode, b)
		}
	}
	if e.readOut("prompt") != "" {
		t.Error("a refused request reached the engine")
	}
	// null is not set; the llama.cpp transport fields and the output budget are ignored, not refused
	resp := e.post("/v1/chat/completions", `{`+msg+`,"temperature":null,"stop":null,"cache_prompt":true,"stream_options":{"include_usage":true},`+
		`"max_tokens":1024,"max_completion_tokens":50,"parallel_tool_calls":true,"user":"u","store":false,"metadata":{"a":"b"}}`)
	if b := readBody(t, resp); resp.StatusCode != 200 {
		t.Fatalf("ignored fields refused: %d %s", resp.StatusCode, b)
	}
	var rj struct {
		MaxNew int `json:"max_new"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	if rj.MaxNew != 8192 {
		t.Errorf("max_new %d: max_tokens/max_completion_tokens must not set it", rj.MaxNew)
	}
	// ... while /v1/responses still refuses its budget
	resp = e.post("/v1/responses", `{"model":"q","input":"hi","max_output_tokens":50}`)
	if resp.StatusCode != 400 {
		t.Errorf("max_output_tokens on /v1/responses: %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestChatToolsBothWays(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "summary"})
	e.waitState("q", StateUp, 10*time.Second)
	resp := e.post("/v1/chat/completions", `{"model":"q","stream":false,"tool_choice":"auto",`+
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],`+
		`"messages":[{"role":"system","content":"S1"},{"role":"developer","content":[{"type":"text","text":"S2"}]},`+
		`{"role":"user","content":"weather in Paris?"},`+
		`{"role":"assistant","content":null,"reasoning_content":"need the tool","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},`+
		`{"role":"tool","tool_call_id":"call_1","content":"sunny"},`+
		`{"role":"user","content":[{"type":"text","text":"and "},{"type":"text","text":"tomorrow?"}]}]}`)
	if b := readBody(t, resp); resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	p := e.readOut("prompt")
	for _, w := range []string{
		`[tool:{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]`,
		"S1\n\nS2<|im_end|>\n<|im_start|>user\nweather in Paris?<|im_end|>\n",                                                                     // one leading system message
		"<|im_start|>assistant\n<think>\nneed the tool\n</think>\n\n<tool_call>[call_1 get_weather {\"city\":\"Paris\"}]</tool_call><|im_end|>\n", // arguments are an object
		"<|im_start|>tool\n[call_1]sunny<|im_end|>\n<|im_start|>user\nand tomorrow?<|im_end|>\n",
	} {
		if !strings.Contains(p, w) {
			t.Errorf("prompt lacks %q:\n%s", w, p)
		}
	}
	// tool_choice none offers no tools
	resp = e.post("/v1/chat/completions", `{"model":"q","tool_choice":"none","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}`)
	readBody(t, resp)
	ps := strings.Split(strings.TrimSuffix(e.readOut("prompt"), "\n====\n"), "\n====\n")
	if strings.Contains(ps[len(ps)-1], "[tool:") {
		t.Error("tool_choice none kept the tools")
	}
}

func TestChatErrors(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "summary", ctx: 1500, display: "Qwen"},
		epSpec{name: "emb", scenario: "embed", kind: KindEmbed},
		epSpec{name: "bad", scenario: "smallpos"})
	e.waitState("q", StateUp, 10*time.Second)
	e.waitState("emb", StateUp, 10*time.Second)
	e.waitState("bad", StateFailed, 10*time.Second)
	for _, c := range []struct {
		name, body string
		status     int
		code       string
	}{
		{"too long", `{"model":"q","messages":[{"role":"user","content":"` + strings.Repeat("x", 4000) + `"}]}`, 400, "context_length_exceeded"},
		{"unknown model", `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`, 404, "model_not_found"},
		{"no model, no compat service", `{"messages":[{"role":"user","content":"hi"}]}`, 400, "missing_model"},
		{"embedding model", `{"model":"emb","messages":[{"role":"user","content":"hi"}]}`, 400, "not_a_chat_model"},
		{"service failed", `{"model":"bad","messages":[{"role":"user","content":"hi"}]}`, 503, "service_unavailable"},
		{"not json", `{`, 400, "invalid_json"},
		{"no messages", `{"model":"q"}`, 400, "missing_messages"},
		{"empty messages", `{"model":"q","messages":[]}`, 400, "empty_input"},
		{"only a system message", `{"model":"q","messages":[{"role":"system","content":"s"}]}`, 400, "empty_input"},
		{"bad role", `{"model":"q","messages":[{"role":"wizard","content":"hi"}]}`, 400, "unsupported_role"},
		{"bad image", `{"model":"q","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}]}`, 400, "invalid_image"},
		{"images in a system message", `{"model":"q","messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"user","content":"hi"}]}`, 400, "images_not_allowed"},
	} {
		resp := e.post("/v1/chat/completions", c.body)
		b := readBody(t, resp)
		var er struct {
			Error struct{ Message, Type, Code string } `json:"error"`
		}
		json.Unmarshal([]byte(b), &er)
		if resp.StatusCode != c.status || er.Error.Code != c.code || er.Error.Message == "" || er.Error.Type == "" {
			t.Errorf("%s: %d %s", c.name, resp.StatusCode, b)
		}
	}
	// a stream refused before it starts is a plain JSON error, not an event stream
	resp := e.post("/v1/chat/completions", `{"model":"q","stream":true,"messages":[{"role":"user","content":"`+strings.Repeat("x", 4000)+`"}]}`)
	if resp.StatusCode != 400 || resp.Header.Get("Content-Type") != "application/json" {
		t.Errorf("stream overflow: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	resp.Body.Close()
	// the display name, with or without .gguf, names the model
	for _, name := range []string{"Qwen", "Qwen.gguf"} {
		resp := e.post("/v1/chat/completions", `{"model":"`+name+`","messages":[{"role":"user","content":"hi"}]}`)
		if b := readBody(t, resp); resp.StatusCode != 200 {
			t.Errorf("model %q: %d %s", name, resp.StatusCode, b)
		}
	}
	// GET is not a chat completion
	if code, _ := e.getAt(e.base, "/v1/chat/completions"); code != 405 {
		t.Errorf("GET: %d", code)
	}
}

func TestChatCancelOnClientDisconnect(t *testing.T) {
	e := startMux(t, epSpec{name: "h", scenario: "hang"})
	e.waitState("h", StateUp, 10*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	resp := e.postCtx(ctx, "/v1/chat/completions", `{"model":"h","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	buf := make([]byte, 4096)
	var acc string
	for !strings.Contains(acc, "reasoning_content") {
		n, err := resp.Body.Read(buf)
		acc += string(buf[:n])
		if err != nil {
			t.Fatal(err)
		}
	}
	cancel()
	resp.Body.Close()
	waitFor(t, 5*time.Second, "the engine to get cancel", func() bool { return strings.Contains(e.readOut("cancel"), "r1") })
	waitFor(t, 5*time.Second, "the engine to stop", func() bool { return strings.Contains(e.readOut("cancelled"), "yes") })
	waitFor(t, 5*time.Second, "idle", func() bool { in := e.srv.Service("h").Info(); return in.State == StateUp && in.Active == 0 })
}

func TestChatLengthAndEngineError(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "maxnew", maxNew: 40})
	e.waitState("m", StateUp, 10*time.Second)
	resp := e.post("/v1/chat/completions", `{"model":"m","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
	cs := readChatStream(t, resp.Body)
	resp.Body.Close()
	finish, usage := cs.finish()
	if finish != "length" || usage["completion_tokens"] != float64(40) || !cs.done || cs.text("content") != strings.Repeat("a", 40) {
		t.Errorf("finish %q usage %v done %v", finish, usage, cs.done)
	}
	resp = e.post("/v1/chat/completions", `{"model":"m","reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	if r["choices"].([]any)[0].(map[string]any)["finish_reason"] != "length" {
		t.Errorf("non-streaming %v", r)
	}

	e2 := startMux(t, epSpec{name: "m", scenario: "error"})
	e2.waitState("m", StateUp, 10*time.Second)
	resp = e2.post("/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	cs = readChatStream(t, resp.Body)
	resp.Body.Close()
	last := cs.chunks[len(cs.chunks)-1]
	er, _ := last["error"].(map[string]any)
	if er == nil || !strings.Contains(er["message"].(string), "kernel crashed") || er["code"] != "engine_error" || !cs.done {
		t.Errorf("stream error: %v (done %v)", last, cs.done)
	}
	resp = e2.post("/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if b := readBody(t, resp); resp.StatusCode != 502 || !strings.Contains(b, "kernel crashed") {
		t.Errorf("non-streaming error: %d %s", resp.StatusCode, b)
	}
}

func TestChatOnAShim(t *testing.T) {
	e := startMux(t, epSpec{name: "s", scenario: "shim", kind: KindShim, ctx: 128000})
	e.waitState("s", StateUp, 10*time.Second)
	tools := `"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},` +
		`{"type":"function","function":{"name":"apply_patch","parameters":{"type":"object","properties":{"input":{"type":"string"}}}}}]`
	resp := e.post("/v1/chat/completions", `{"model":"s","stream":true,"reasoning_effort":"low","messages":[{"role":"system","content":"S"},{"role":"user","content":"hi"}],`+tools+`}`)
	cs := readChatStream(t, resp.Body)
	resp.Body.Close()
	if len(cs.other) != 0 || !cs.done || cs.text("reasoning_content") != "Let me think. caf\u00e9" || cs.text("content") != "Hello world." {
		t.Fatalf("stream: other %q done %v reasoning %q content %q", cs.other, cs.done, cs.text("reasoning_content"), cs.text("content"))
	}
	tcs := cs.toolCalls()
	if len(tcs) != 2 || tcs[0]["id"] != "call_shim1" || tcs[1]["function"].(map[string]any)["arguments"] != `{"input":"*** Begin Patch\n*** End Patch"}` {
		t.Errorf("tool calls %v", tcs)
	}
	if finish, usage := cs.finish(); finish != "tool_calls" || usage["prompt_tokens"] != float64(11) || usage["completion_tokens"] != float64(22) {
		t.Errorf("finish %q usage %v", finish, usage)
	}
	var run struct {
		Messages []map[string]any `json:"messages"`
		Effort   string           `json:"effort"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("shim.runs"), "\n", 2)[0]), &run)
	if run.Effort != "low" || len(run.Messages) != 2 || run.Messages[0]["role"] != "system" || run.Messages[1]["content"] != "hi" {
		t.Errorf("shim run %+v", run)
	}
	resp = e.post("/v1/chat/completions", `{"model":"s","messages":[{"role":"user","content":"hi"}]}`)
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	if msg := r["choices"].([]any)[0].(map[string]any)["message"].(map[string]any); msg["content"] != "Hello world." || len(msg["tool_calls"].([]any)) != 2 {
		t.Errorf("non-streaming %v", r)
	}
	json.Unmarshal([]byte(strings.Split(strings.TrimSpace(e.readOut("shim.runs")), "\n")[1]), &run)
	if run.Effort != "medium" { // the middle of the five defaults
		t.Errorf("default effort %q, want medium", run.Effort)
	}
}

func TestChatImages(t *testing.T) {
	e := startMux(t, epSpec{name: "v", scenario: "summary"})
	e.waitState("v", StateUp, 10*time.Second)
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNGfake"))
	resp := e.post("/v1/chat/completions", `{"model":"v","messages":[{"role":"user","content":[{"type":"text","text":"see "},{"type":"image_url","image_url":{"url":"data:image/png;base64,`+png+`"}}]}]}`)
	if b := readBody(t, resp); resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	if p := e.readOut("prompt"); !strings.Contains(p, "see <|vision_start|>"+strings.Repeat("<|image_pad|>", 4)+"<|vision_end|>") {
		t.Errorf("prompt:\n%s", p)
	}
}

// ---- /props and the compat listener -------------------------------------------------------------------------

func TestProps(t *testing.T) {
	e := startMux(t,
		epSpec{name: "bad", scenario: "smallpos"},
		epSpec{name: "shim", scenario: "shim", kind: KindShim, ctx: 5000},
		epSpec{name: "q", scenario: "summary", ctx: 123456, cfgGGUF: "/models/q.gguf"})
	e.waitState("q", StateUp, 10*time.Second)
	e.waitState("shim", StateUp, 10*time.Second)
	e.waitState("bad", StateFailed, 10*time.Second)
	code, body := e.getAt(e.base, "/props")
	var got map[string]any
	json.Unmarshal([]byte(body), &got)
	want := map[string]any{
		"default_generation_settings": map[string]any{"n_ctx": float64(123456)},
		"modalities":                  map[string]any{"vision": true},
		"chat_template":               "",
		"model_path":                  "/models/q.gguf",
	}
	// the first engine that is up: not the failed one, not the shim that is listed before it
	if code != 200 || !reflect.DeepEqual(got, want) {
		t.Errorf("/props: %d %s", code, body)
	}
	if resp := e.at(e.base, "POST", "/props", "{}"); resp.StatusCode != 405 {
		t.Errorf("POST /props: %d", resp.StatusCode)
	}
	// none up: 404
	resp := e.post("/v1/services/q/down", "")
	resp.Body.Close()
	e.waitState("q", StateDown, 10*time.Second)
	if code, body := e.getAt(e.base, "/props"); code != 404 || !strings.Contains(body, `"error"`) {
		t.Errorf("/props with no engine up: %d %s", code, body)
	}
}

func TestCompatListener(t *testing.T) {
	e := startMux(t,
		epSpec{name: "other", scenario: "summary", ctx: 777},
		epSpec{name: "qwen3.8-27b", scenario: "summary", ctx: 262144, display: "Qwen3.8-27B", compat: true, cfgGGUF: "/m/Qwen3.8-27B.Q5_K_M.gguf"})
	e.waitState("qwen3.8-27b", StateUp, 10*time.Second)
	e.waitState("other", StateUp, 10*time.Second)
	if e.compat == "" || e.compat == e.base {
		t.Fatalf("no compat listener (%q)", e.compat)
	}

	// the compat listener's /v1/models is llama-server's: the id is the GGUF's name
	code, body := e.getAt(e.compat, "/v1/models")
	if code != 200 || body != `{"data":[{"id":"Qwen3.8-27B.gguf","object":"model"}],"object":"list"}`+"\n" {
		t.Errorf("compat /v1/models: %d %s", code, body)
	}
	// /props describes the compat service even though another engine is listed first
	code, body = e.getAt(e.compat, "/props")
	var props struct {
		Settings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		Modalities struct{ Vision bool } `json:"modalities"`
		ModelPath  string                `json:"model_path"`
	}
	json.Unmarshal([]byte(body), &props)
	if code != 200 || props.Settings.NCtx != 262144 || !props.Modalities.Vision || props.ModelPath != "/m/Qwen3.8-27B.Q5_K_M.gguf" {
		t.Errorf("compat /props: %d %s", code, body)
	}
	// ... and the main listener's /v1/models is unchanged: service names are the ids Codex uses
	var models struct{ Data []map[string]any }
	e.getJSON("/v1/models", &models)
	if len(models.Data) != 2 || models.Data[1]["id"] != "qwen3.8-27b" || models.Data[1]["kind"] != "engine" || models.Data[1]["ctx"] != float64(262144) ||
		models.Data[1]["state"] != "up" || models.Data[1]["vision"] != true || models.Data[1]["object"] != "model" || len(models.Data[1]["efforts"].([]any)) != 5 {
		t.Errorf("main /v1/models: %v", models.Data)
	}
	// the gateway's chat runs on the compat service without naming it
	resp := e.at(e.compat, "POST", "/v1/chat/completions", gatewayBody)
	cs := readChatStream(t, resp.Body)
	resp.Body.Close()
	if f, _ := cs.finish(); !cs.done || f != "stop" || !strings.HasPrefix(cs.text("reasoning_content"), "Summary: weather") {
		t.Errorf("chat on the compat listener: done %v finish %q reasoning %q", cs.done, f, cs.text("reasoning_content"))
	}
	for _, p := range []string{"/v1/responses", "/v1/services", "/v1/events", "/v1/embeddings"} {
		if code, _ := e.getAt(e.compat, p); code != 404 {
			t.Errorf("compat listener serves %s (%d)", p, code)
		}
	}
	// the main listener has /props and chat too (the same compat service is its default model)
	if code, _ := e.getAt(e.base, "/props"); code != 200 {
		t.Errorf("main /props: %d", code)
	}
	resp = e.post("/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`)
	if b := readBody(t, resp); resp.StatusCode != 200 || !strings.Contains(b, `"model":"qwen3.8-27b"`) {
		t.Errorf("main chat without a model: %d %s", resp.StatusCode, b)
	}
	// down: /props says so (404); /v1/models still names the model; chat is a 503
	resp = e.post("/v1/services/qwen3.8-27b/down", "")
	resp.Body.Close()
	e.waitState("qwen3.8-27b", StateDown, 10*time.Second)
	if code, _ := e.getAt(e.compat, "/props"); code != 404 {
		t.Errorf("compat /props when down: %d", code)
	}
	if code, body := e.getAt(e.compat, "/v1/models"); code != 200 || !strings.Contains(body, "Qwen3.8-27B.gguf") {
		t.Errorf("compat /v1/models when down: %d %s", code, body)
	}
	resp = e.at(e.compat, "POST", "/v1/chat/completions", gatewayBody)
	if resp.StatusCode != 503 {
		t.Errorf("chat when down: %d", resp.StatusCode)
	}
	resp.Body.Close()
	// no display: the service name is the model name
	e2 := startMux(t, epSpec{name: "plain", scenario: "summary", compat: true})
	if code, body := e2.getAt(e2.compat, "/v1/models"); code != 200 || !strings.Contains(body, `"id":"plain.gguf"`) {
		t.Errorf("without display: %d %s", code, body)
	}
}

func TestCompatConfig(t *testing.T) {
	dir := t.TempDir()
	writeService(t, dir, "q", "kind: engine\ndisplay: Qwen3.8-27B\ngguf: /g.gguf\nctx: 1000\ncommand: [x]\n")
	writeService(t, dir, "e", "kind: embed\ncommand: [x]\n")
	writeService(t, dir, "s", "kind: shim\nctx: 1000\ncommand: [x]\n")
	host := func(extra string) string {
		return "socket: /x\nservices_dir: " + dir + "\nservices: {q: {up: true}, e: {up: true}, s: {up: true}}\n" + extra
	}
	c, err := loadConfig([]byte(host("compat_listen: 127.0.0.1:18101\ncompat_service: q\n")), "")
	if err != nil {
		t.Fatal(err)
	}
	if c.CompatListen != "127.0.0.1:18101" || c.CompatService != "q" || c.Services[0].Display != "Qwen3.8-27B" || c.Services[0].LlamaID() != "Qwen3.8-27B.gguf" || c.Services[1].LlamaID() != "e.gguf" {
		t.Errorf("config %+v %+v", c, c.Services[0])
	}
	if _, err := loadConfig([]byte(host("compat_service: s\n")), ""); err != nil { // a shim fronts chat too; no listener is fine
		t.Errorf("compat_service alone: %v", err)
	}
	for name, extra := range map[string]string{
		"listen without service": "compat_listen: 127.0.0.1:1\n",
		"unknown service":        "compat_listen: 127.0.0.1:1\ncompat_service: nope\n",
		"embed service":          "compat_listen: 127.0.0.1:1\ncompat_service: e\n",
	} {
		if _, err := loadConfig([]byte(host(extra)), ""); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
