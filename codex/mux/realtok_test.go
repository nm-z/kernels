package mux

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func findGGUF(t *testing.T) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	for _, p := range []string{
		os.Getenv("AI_MUX_TEST_GGUF"),
		home + "/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf",
		"/data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf",
	} {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no Qwen3.8 GGUF found (set AI_MUX_TEST_GGUF, or put one at ~/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf)")
	return ""
}

// TestRealTokenizerEndToEnd serves the fake engine's scripted answer through the real GGUF tokenizer
// and chat template: byte-level tokens that split characters, the real <think> and <tool_call> ids,
// the real template's rendering of tools, tool calls and tool results.
func TestRealTokenizerEndToEnd(t *testing.T) {
	gguf := findGGUF(t)
	e := startMuxOpts(t, Options{LoadModel: LoadTokenizer}, epSpec{name: "q", scenario: "normal", gguf: gguf})
	e.waitState("q", StateUp, 20*time.Second)

	resp := e.post("/v1/responses", codexBody("q", ""))
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	var reasoning, answer strings.Builder
	for _, ev := range res.named("response.reasoning_text.delta") {
		reasoning.WriteString(ev.obj()["delta"].(string))
	}
	for _, ev := range res.named("response.output_text.delta") {
		answer.WriteString(ev.obj()["delta"].(string))
	}
	if want := "The user wants the weather in Paris: caf\u00e9 \u65e5\u672c\u8a9e. I should call the tool."; reasoning.String() != want {
		t.Errorf("reasoning %q", reasoning.String())
	}
	if answer.String() != "Let me check." {
		t.Errorf("answer %q", answer.String())
	}
	var items []map[string]any
	for _, ev := range res.named("response.output_item.done") {
		items = append(items, ev.obj()["item"].(map[string]any))
	}
	if len(items) != 4 || items[2]["arguments"] != `{"city":"Paris","days":3}` || items[3]["input"] != "*** Begin Patch\n*** End Patch" || len(res.named("response.completed")) != 1 {
		t.Fatalf("items %v", items)
	}

	prompt := strings.SplitN(e.readOut("prompt"), "\n====\n", 2)[0]
	for _, w := range []string{"<|im_start|>system\nReasoning effort is set to low.", "# Tools", `"name": "get_weather"`, `"name": "apply_patch"`,
		"You are Codex.", "<|im_start|>user\nweather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n"} {
		if !strings.Contains(prompt, w) {
			t.Errorf("prompt lacks %q:\n%s", w, prompt)
		}
	}
	if !strings.HasSuffix(prompt, "<think>\n") {
		t.Errorf("prompt does not end inside <think>: %q", prompt[len(prompt)-40:])
	}

	// the next turn: the model's output goes back as items, with the tool results
	var input []string
	input = append(input, `{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]}`)
	for _, it := range items {
		b, _ := json.Marshal(it)
		input = append(input, string(b))
	}
	input = append(input,
		`{"type":"function_call_output","call_id":"`+items[2]["call_id"].(string)+`","output":"sunny, 21C"}`,
		`{"type":"custom_tool_call_output","call_id":"`+items[3]["call_id"].(string)+`","output":"patched"}`)
	next := strings.Replace(codexBody("q", ""), `"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]}]`, `"input":[`+strings.Join(input, ",")+`]`, 1)
	resp = e.post("/v1/responses", next)
	readSSE(t, resp.Body)
	resp.Body.Close()
	prompts := strings.Split(e.readOut("prompt"), "\n====\n")
	p2 := prompts[1]
	for _, w := range []string{
		"<|im_start|>assistant\n<think>\n" + "The user wants the weather in Paris: caf\u00e9 \u65e5\u672c\u8a9e. I should call the tool." + "\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
		"<|im_start|>user\n<tool_response>\nsunny, 21C\n</tool_response>\n<tool_response>\npatched\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n",
	} {
		if !strings.Contains(p2, w) {
			t.Errorf("second prompt lacks %q:\n%s", w, p2)
		}
	}
}
