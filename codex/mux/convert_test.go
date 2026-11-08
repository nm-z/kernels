package mux

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func testEndpoint() *ServiceConfig {
	return &ServiceConfig{Name: "m", Kind: KindEngine, Ctx: 1000, Vision: true, Efforts: defaultEfforts}
}

func prep(t *testing.T, body string) *prepared {
	t.Helper()
	p, err := prepare([]byte(body), testEndpoint())
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return p
}

func msgsJSON(p *prepared) string {
	b, _ := json.Marshal(p.messages)
	return string(b)
}

func TestConvertCodexConversation(t *testing.T) {
	p := prep(t, `{"model":"m","stream":true,"instructions":"be brief","input":[
	 {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev rules"}]},
	 {"type":"message","role":"user","content":[{"type":"input_text","text":"list "},{"type":"input_text","text":"files"}]},
	 {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"need ls"}],"encrypted_content":null},
	 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Running."}]},
	 {"type":"function_call","name":"shell","arguments":"{\"cmd\":\"ls\",\"n\":2}","call_id":"c1"},
	 {"type":"function_call","name":"read","arguments":"{\"p\":\"a\"}","call_id":"c2"},
	 {"type":"function_call_output","call_id":"c1","output":"a b"},
	 {"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"x"},{"type":"input_text","text":"y"}]},
	 {"type":"reasoning","summary":[{"type":"summary_text","text":"s"}],"content":null},
	 {"type":"custom_tool_call","name":"apply_patch","input":"*** Begin","call_id":"c3"},
	 {"type":"custom_tool_call_output","call_id":"c3","output":"done"},
	 {"type":"message","role":"system","content":"sys late"},
	 {"type":"message","role":"user","content":"thanks"}
	]}`)
	want := `[{"role":"system","content":"be brief\n\ndev rules\n\nsys late"},` +
		`{"role":"user","content":"list files"},` +
		`{"role":"assistant","content":"Running.","reasoning_content":"need ls","tool_calls":[` +
		`{"id":"c1","type":"function","function":{"name":"shell","arguments":{"cmd":"ls","n":2}}},` +
		`{"id":"c2","type":"function","function":{"name":"read","arguments":{"p":"a"}}}]},` +
		`{"role":"tool","content":"a b","tool_call_id":"c1"},` +
		`{"role":"tool","content":"x\ny","tool_call_id":"c2"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c3","type":"function","function":{"name":"apply_patch","arguments":{"input":"*** Begin"}}}]},` +
		`{"role":"tool","content":"done","tool_call_id":"c3"},` +
		`{"role":"user","content":"thanks"}]`
	if got := msgsJSON(p); got != want {
		t.Fatalf("messages:\n got %s\nwant %s", got, want)
	}
}

func TestConvertStringInputAndEffort(t *testing.T) {
	for effort, check := range map[string]func(map[string]any) bool{
		"none":   func(v map[string]any) bool { return v["enable_thinking"] == false && v["reasoning_effort"] == nil },
		"low":    func(v map[string]any) bool { return v["reasoning_effort"] == "low" && v["enable_thinking"] == nil },
		"medium": func(v map[string]any) bool { return v["reasoning_effort"] == "medium" },
		"high":   func(v map[string]any) bool { return v["reasoning_effort"] == "xhigh" },
		"xhigh":  func(v map[string]any) bool { return v["reasoning_effort"] == "xhigh" },
		"":       func(v map[string]any) bool { return v["reasoning_effort"] == nil && v["enable_thinking"] == nil },
	} {
		body := `{"model":"m","input":"hi"}`
		if effort != "" {
			body = `{"model":"m","input":"hi","reasoning":{"effort":"` + effort + `"}}`
		}
		p := prep(t, body)
		if !check(p.templateVars()) {
			t.Errorf("effort %q: %v", effort, p.templateVars())
		}
		if v := p.templateVars(); v["add_generation_prompt"] != true || msgsJSON(p) != `[{"role":"user","content":"hi"}]` {
			t.Errorf("vars %v", v)
		}
	}
}

func TestConvertTools(t *testing.T) {
	p := prep(t, `{"model":"m","input":"hi","tools":[
	  {"type":"function","name":"shell","description":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"n":{"type":["integer","null"]}},"required":["cmd"]},"strict":false},
	  {"type":"custom","name":"apply_patch","description":"patch","format":{"type":"grammar","syntax":"lark","definition":"start: x"}},
	  {"type":"namespace","name":"mcp__srv__","description":"d","tools":[{"type":"function","name":"go","parameters":{"type":"object","properties":{}}}]},
	  {"type":"web_search"}
	],"input":[{"type":"additional_tools","role":"developer","tools":[{"type":"function","name":"extra","parameters":{"type":"object","properties":{"a":{"type":"boolean"}}}}]},{"type":"message","role":"user","content":"x"}]}`)
	b, _ := json.Marshal(p.tools)
	got := string(b)
	for _, w := range []string{
		`{"type":"function","function":{"name":"shell","description":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"n":{"type":["integer","null"]}},"required":["cmd"]}}}`,
		`"name":"apply_patch"`, `lark grammar:\nstart: x`, `"required":["input"]`,
		`"name":"mcp__srv__go"`, `"name":"extra"`,
	} {
		if !strings.Contains(got, w) {
			t.Errorf("tools lack %s:\n%s", w, got)
		}
	}
	if strings.Contains(got, "web_search") || len(p.tools) != 4 {
		t.Errorf("a tool ai-mux cannot offer was kept: %d tools", len(p.tools))
	}
	if ti := p.toolInfo["apply_patch"]; ti == nil || !ti.custom {
		t.Errorf("custom tool not recorded: %+v", ti)
	}
	if ti := p.toolInfo["mcp__srv__go"]; ti == nil || ti.name != "go" || ti.namespace != "mcp__srv__" {
		t.Errorf("namespace not recorded: %+v", ti)
	}
	if ti := p.toolInfo["shell"]; ti == nil || ti.params["n"][1] != "null" || ti.params["cmd"][0] != "string" {
		t.Errorf("schema types: %+v", ti)
	}
	// the chat shape is accepted too, and tool_choice none drops the tools
	p = prep(t, `{"model":"m","input":"hi","tool_choice":"none","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}`)
	if len(p.tools) != 0 {
		t.Error("tool_choice none kept tools")
	}
	p = prep(t, `{"model":"m","input":"hi","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}`)
	if len(p.tools) != 1 {
		t.Error("chat-shaped tool not accepted")
	}
}

func TestConvertImages(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNGfake"))
	p := prep(t, `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"see"},{"type":"input_image","image_url":"data:image/png;base64,`+png+`"}]}]}`)
	if len(p.images) != 1 || string(p.images[0].bytes) != "\x89PNGfake" || len(p.images[0].hash) != 64 {
		t.Fatalf("images %+v", p.images)
	}
	if got := msgsJSON(p); got != `[{"role":"user","content":[{"text":"see","type":"text"},{"img":"`+hashBytes([]byte("\x89PNGfake"))+`","type":"image"}]}]` {
		t.Errorf("messages %s", got)
	}
	ep := testEndpoint()
	ep.Vision = false
	if _, err := prepare([]byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,`+png+`"}]}]}`), ep); err == nil || err.Code != "images_not_supported" {
		t.Errorf("image on a text-only endpoint: %v", err)
	}
	for _, bad := range []string{"https://x/y.png", "data:image/png,abc", "data:image/png;base64,!!!"} {
		if _, err := prepare([]byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"`+bad+`"}]}]}`), testEndpoint()); err == nil || err.Code != "invalid_image" {
			t.Errorf("%s: %v", bad, err)
		}
	}
}

func TestConvertRefusals(t *testing.T) {
	for _, f := range []string{"temperature", "top_p", "top_k", "seed", "max_output_tokens", "presence_penalty", "frequency_penalty", "stop"} {
		_, err := prepare([]byte(`{"model":"m","input":"hi","`+f+`":1}`), testEndpoint())
		if err == nil || err.Status != 400 || err.Param != f || err.Code != "unsupported_parameter" {
			t.Errorf("%s: %+v", f, err)
		}
	}
	// null is not set
	if _, err := prepare([]byte(`{"model":"m","input":"hi","temperature":null,"stop":null}`), testEndpoint()); err != nil {
		t.Errorf("null refused: %v", err)
	}
	// the transport fields Codex sends are accepted
	if _, err := prepare([]byte(`{"model":"m","input":"hi","tool_choice":"auto","parallel_tool_calls":true,"store":false,"include":["reasoning.encrypted_content"],"prompt_cache_key":"k","text":{"verbosity":"low"},"client_metadata":{"a":"b"},"service_tier":"priority","stream_options":{"include_obfuscation":false}}`), testEndpoint()); err != nil {
		t.Errorf("transport fields refused: %v", err)
	}
	_, err := prepare([]byte(`{"model":"m","input":"hi","reasoning":{"effort":"max"}}`), testEndpoint())
	if err == nil || err.Status != 400 || err.Code != "unsupported_reasoning_effort" || !strings.Contains(err.Message, "none, low, medium, high, xhigh") {
		t.Errorf("effort: %+v", err)
	}
	for _, bad := range []string{`{"input":"hi"}`, `{"model":"m"}`, `{"model":"m","input":[{"type":"mystery"}]}`, `{"model":"m","input":[]}`, `{"model":"m","input":"x","previous_response_id":"resp_1"}`, `[]`} {
		if _, err := prepare([]byte(bad), testEndpoint()); err == nil || err.Status != 400 {
			t.Errorf("%s accepted: %v", bad, err)
		}
	}
}
