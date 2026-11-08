package mux

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

type memSink struct {
	evs []struct {
		name string
		data map[string]any
	}
}

func (m *memSink) write(typ string, data []byte) {
	var d map[string]any
	json.Unmarshal(data, &d)
	m.evs = append(m.evs, struct {
		name string
		data map[string]any
	}{typ, d})
}
func (m *memSink) comment(string)            {}
func (m *memSink) sinceWrite() time.Duration { return 0 }
func (m *memSink) failed() bool              { return false }
func (m *memSink) deltas(name string) string {
	var sb strings.Builder
	for _, e := range m.evs {
		if e.name == name {
			sb.WriteString(e.data["delta"].(string))
		}
	}
	return sb.String()
}

// feedText runs text through a builder one token (one byte, or a special) at a time.
func feedText(t *testing.T, text string, inThink bool, tools map[string]*toolInfo) (*respBuilder, *memSink) {
	t.Helper()
	ms := &memSink{}
	rb := newRespBuilder("m", fakeModel{}, tools, ms, inThink)
	ids, _ := fakeModel{}.Encode(text)
	for _, id := range ids {
		rb.feed(id)
	}
	rb.finish()
	return rb, ms
}

func itemTypes(rb *respBuilder) []string {
	var out []string
	for _, it := range rb.items {
		out = append(out, it["type"].(string))
	}
	return out
}

func TestParserReasoningAnswerAndCalls(t *testing.T) {
	tools := map[string]*toolInfo{
		"get_weather": {visible: "get_weather", name: "get_weather", params: map[string][]string{"city": {"string"}, "days": {"integer"}}},
		"apply_patch": {visible: "apply_patch", name: "apply_patch", custom: true},
		"mcp__s__t":   {visible: "mcp__s__t", name: "t", namespace: "mcp__s__"},
	}
	rb, ms := feedText(t, fakeAnswer, true, tools)
	if got := strings.Join(itemTypes(rb), ","); got != "reasoning,message,function_call,custom_tool_call" {
		t.Fatalf("items: %s", got)
	}
	wantReason := "The user wants the weather in Paris: café 日本語. I should call the tool."
	if g := ms.deltas("response.reasoning_text.delta"); g != wantReason {
		t.Errorf("reasoning deltas %q", g)
	}
	if g := ms.deltas("response.output_text.delta"); g != "Let me check." {
		t.Errorf("answer deltas %q (tool XML must not stream as text, trailing blanks are dropped)", g)
	}
	fc := rb.items[2]
	if fc["name"] != "get_weather" || fc["arguments"] != `{"city":"Paris","days":3}` || fc["status"] != "completed" {
		t.Errorf("function_call %v", fc)
	}
	if !strings.HasPrefix(fc["call_id"].(string), "call_") || !strings.HasPrefix(fc["id"].(string), "fc_") {
		t.Errorf("ids %v", fc)
	}
	ct := rb.items[3]
	if ct["name"] != "apply_patch" || ct["input"] != "*** Begin Patch\n*** End Patch" {
		t.Errorf("custom_tool_call %v", ct)
	}
	rs := rb.items[0]
	c := rs["content"].([]any)[0].(map[string]any)
	if c["type"] != "reasoning_text" || c["text"] != wantReason {
		t.Errorf("reasoning item %v", rs)
	}
}

func TestParserNamespaceAndTypes(t *testing.T) {
	tools := map[string]*toolInfo{"mcp__s__t": {visible: "mcp__s__t", name: "t", namespace: "mcp__s__"}}
	rb, _ := feedText(t, "<tool_call>\n<function=mcp__s__t>\n<parameter=a>\n12\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=c>\n[1, 2]\n</parameter>\n<parameter=d>\nhello\nworld\n</parameter>\n<parameter=e>\n007\n</parameter>\n<parameter=f>\nnull\n</parameter>\n</function>\n</tool_call><|im_end|>", false, tools)
	if len(rb.items) != 1 {
		t.Fatalf("items %v", rb.items)
	}
	fc := rb.items[0]
	if fc["name"] != "t" || fc["namespace"] != "mcp__s__" {
		t.Errorf("namespace not split: %v", fc)
	}
	// no schema: numbers, booleans, arrays and objects keep their type, everything else is a string
	if fc["arguments"] != `{"a":12,"b":true,"c":[1,2],"d":"hello\nworld","e":"007","f":"null"}` {
		t.Errorf("arguments %v", fc["arguments"])
	}
}

func TestParserSchemaKeepsStrings(t *testing.T) {
	tools := map[string]*toolInfo{"f": {visible: "f", name: "f", params: map[string][]string{"s": {"string"}, "n": {"number"}, "u": {"string", "null"}}}}
	rb, _ := feedText(t, "<tool_call>\n<function=f>\n<parameter=s>\n123\n</parameter>\n<parameter=n>\n1.5e3\n</parameter>\n<parameter=u>\nnull\n</parameter>\n</function>\n</tool_call>", false, tools)
	if rb.items[0]["arguments"] != `{"s":"123","n":1.5e3,"u":null}` {
		t.Errorf("arguments %v", rb.items[0]["arguments"])
	}
}

func TestParserNoThinkAndHoldback(t *testing.T) {
	// the prompt already closed the think block: everything is the answer; "<tool" is held back until it is known not to be a call
	rb, ms := feedText(t, "\n\nUse a < b and <tool and <to>. Done.  \n", false, nil)
	if got := strings.Join(itemTypes(rb), ","); got != "message" {
		t.Fatalf("items %s", got)
	}
	if g := ms.deltas("response.output_text.delta"); g != "Use a < b and <tool and <to>. Done." {
		t.Errorf("answer %q", g)
	}
}

func TestParserTruncatedCall(t *testing.T) {
	tools := map[string]*toolInfo{"f": {visible: "f", name: "f"}}
	// cut off after the function block but before </tool_call>
	rb, _ := feedText(t, "ok\n<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n", false, tools)
	if got := strings.Join(itemTypes(rb), ","); got != "message,function_call" || rb.items[1]["arguments"] != `{"x":1}` {
		t.Errorf("items %v", rb.items)
	}
	// cut off inside a parameter: lenient
	rb, _ = feedText(t, "<tool_call>\n<function=f>\n<parameter=x>\nabc", false, tools)
	if len(rb.items) != 1 || rb.items[0]["arguments"] != `{"x":"abc"}` {
		t.Errorf("items %v", rb.items)
	}
	// not a call at all: shown as text rather than lost
	rb, ms := feedText(t, "<tool_call>garbage", false, nil)
	if got := strings.Join(itemTypes(rb), ","); got != "message" || ms.deltas("response.output_text.delta") != "<tool_call>garbage" {
		t.Errorf("items %v", rb.items)
	}
}

func TestParserUnfinishedReasoning(t *testing.T) {
	// the run ended inside <think> (max_new): reasoning item only, no message
	rb, _ := feedText(t, "still thinking  \n", true, nil)
	if got := strings.Join(itemTypes(rb), ","); got != "reasoning" {
		t.Fatalf("items %s", got)
	}
	// empty reasoning produces no item
	rb, _ = feedText(t, "\n\n</think>\n\nhi", true, nil)
	if got := strings.Join(itemTypes(rb), ","); got != "message" {
		t.Fatalf("items %s", got)
	}
	// the model opens the think block itself
	rb, _ = feedText(t, "<think>hmm</think>answer", false, nil)
	if got := strings.Join(itemTypes(rb), ","); got != "reasoning,message" {
		t.Fatalf("items %s", got)
	}
}

func TestUsageReasoningTokens(t *testing.T) {
	rb, _ := feedText(t, "abc</think>\n\nxy", true, nil)
	if rb.reasoningTokens != 4 { // a b c and </think>
		t.Errorf("reasoning tokens %d", rb.reasoningTokens)
	}
}
