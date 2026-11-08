package tmpl

import (
	"os"
	"strings"
	"sync"
	"testing"
)

var (
	once   sync.Once
	loaded *Tokenizer
	lerr   error
	lpath  string
)

// testGGUF finds a Qwen3.8 GGUF with tokenizer and chat template; tests that need it skip without one.
func testGGUF(t *testing.T) *Tokenizer {
	t.Helper()
	once.Do(func() {
		home, _ := os.UserHomeDir()
		for _, p := range []string{
			os.Getenv("MUX_TEST_GGUF"),
			home + "/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf",
			"/data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf",
		} {
			if p == "" {
				continue
			}
			if _, err := os.Stat(p); err != nil {
				continue
			}
			lpath = p
			loaded, lerr = Load(p)
			return
		}
	})
	if lpath == "" {
		t.Skip("no Qwen3.8 GGUF found (set MUX_TEST_GGUF, or put one at ~/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf)")
	}
	if lerr != nil {
		t.Fatal(lerr)
	}
	return loaded
}

func msgs(m ...map[string]any) []any {
	out := make([]any, len(m))
	for i := range m {
		out[i] = m[i]
	}
	return out
}

func TestRenderThinking(t *testing.T) {
	tk := testGGUF(t)
	got, err := tk.Render(map[string]any{
		"messages": msgs(
			map[string]any{"role": "system", "content": "S"},
			map[string]any{"role": "user", "content": "hi"}),
		"add_generation_prompt": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestRenderNoThinkingAndEffort(t *testing.T) {
	tk := testGGUF(t)
	m := msgs(map[string]any{"role": "user", "content": "hi"})
	got, err := tk.Render(map[string]any{"messages": m, "add_generation_prompt": true, "enable_thinking": false})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	got, err = tk.Render(map[string]any{"messages": m, "add_generation_prompt": true, "reasoning_effort": "medium"})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	got, err = tk.Render(map[string]any{"messages": m, "add_generation_prompt": true, "reasoning_effort": "low"})
	if err != nil || !strings.Contains(got, "Reasoning effort is set to low.") {
		t.Fatalf("low: %q %v", got, err)
	}
	if _, err = tk.Render(map[string]any{"messages": m, "reasoning_effort": "bogus"}); err == nil {
		t.Fatal("an unsupported effort rendered")
	}
}

func TestRenderToolsAndToolCalls(t *testing.T) {
	tk := testGGUF(t)
	tools := []any{map[string]any{"type": "function", "function": map[string]any{
		"name": "get_weather", "description": "weather",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
	}}}
	got, err := tk.Render(map[string]any{
		"tools": tools,
		"messages": msgs(
			map[string]any{"role": "user", "content": "weather in Paris?"},
			map[string]any{"role": "assistant", "content": "", "reasoning_content": "need the tool",
				"tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{
					"name": "get_weather", "arguments": map[string]any{"city": "Paris", "days": 3}}}}},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "sunny"}),
		"add_generation_prompt": true,
		"enable_thinking":       false,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{
		"<tools>\n{\"function\": {\"description\": \"weather\"",
		"<|im_start|>assistant\n<think>\nneed the tool\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
		"<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
	} {
		if !strings.Contains(got, w) {
			t.Fatalf("missing %q in %q", w, got)
		}
	}
}

func TestEncodePieceRoundTrip(t *testing.T) {
	tk := testGGUF(t)
	text := "<|im_start|>user\nHello, wörld! 日本語 \U0001F600 <tool_call>\n<function=x>\n</function>\n</tool_call><|im_end|>\n"
	ids, err := tk.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	for _, id := range ids {
		sb.Write(tk.Piece(id))
	}
	if sb.String() != text {
		t.Fatalf("round trip: got %q want %q", sb.String(), text)
	}
	imEnd, thinkEnd, tc := tk.Find("<|im_end|>"), tk.Find("</think>"), tk.Find("<tool_call>")
	if imEnd < 0 || thinkEnd < 0 || tc < 0 {
		t.Fatalf("special tokens missing: im_end %d </think> %d <tool_call> %d", imEnd, thinkEnd, tc)
	}
	if ids[0] != uint32(tk.Find("<|im_start|>")) {
		t.Fatalf("first id %d is not <|im_start|>", ids[0])
	}
	if !tk.Hidden(uint32(imEnd)) {
		t.Error("<|im_end|> should be hidden")
	}
	if tk.Hidden(uint32(tc)) {
		t.Error("<tool_call> should be shown")
	}
	if tk.EOS() == 0xffffffff {
		t.Error("no eos")
	}
	if tk.Count() < 100000 {
		t.Errorf("vocab size %d", tk.Count())
	}
}

func TestRenderImage(t *testing.T) {
	tk := testGGUF(t)
	got, err := tk.Render(map[string]any{
		"messages": msgs(map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "image"}, map[string]any{"type": "text", "text": "what is this?"}}}),
		"add_generation_prompt": true, "enable_thinking": false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "<|vision_start|><|image_pad|><|vision_end|>what is this?") {
		t.Fatalf("no image pad: %q", got)
	}
	if tk.Find("<|image_pad|>") < 0 {
		t.Error("no <|image_pad|> token")
	}
}

func TestConcurrentUse(t *testing.T) {
	tk := testGGUF(t)
	m := msgs(map[string]any{"role": "user", "content": "hello wörld 日本語"})
	vars := map[string]any{"messages": m, "add_generation_prompt": true}
	want, err := tk.Render(vars)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs, _ := tk.Encode(want)
	var wg sync.WaitGroup
	errs := make(chan string, 16)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				got, err := tk.Render(vars)
				if err != nil || got != want {
					errs <- "render differs"
					return
				}
				ids, _ := tk.Encode(got)
				if len(ids) != len(wantIDs) {
					errs <- "encode differs"
					return
				}
				for j := range ids {
					if ids[j] != wantIDs[j] {
						errs <- "encode differs"
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}
