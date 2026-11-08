package mux

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// A conversation of three turns, long enough that the whole of it plus the summarization prompt does not
// fit a 1500-token window.
func longConversation() string {
	a, b := strings.Repeat("A", 500), strings.Repeat("B", 500)
	return `[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"turn one ` + a + `"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer one ` + strings.Repeat("a", 200) + `"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"turn two ` + b + `"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer two ` + strings.Repeat("b", 200) + `"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"turn three: weather in Paris, 3 days?"}]}`
}

func TestContextLengthExceeded(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "summary", ctx: 1500})
	e.waitState("m", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":`+longConversation()+`]}`)
	body := readBody(t, resp)
	var er struct {
		Error struct{ Message, Type, Code, Param string } `json:"error"`
	}
	json.Unmarshal([]byte(body), &er)
	if resp.StatusCode != 400 || er.Error.Code != "context_length_exceeded" || er.Error.Param != "input" || er.Error.Type != "invalid_request_error" ||
		!strings.Contains(er.Error.Message, "1500") {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	if e.readOut("prompt") != "" {
		t.Error("the oversized prompt reached the engine")
	}
}

func TestCompactionRoundTrip(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "summary", ctx: 1500})
	e.waitState("m", StateUp, 10*time.Second)
	sub := e.srv.Hub().Subscribe()
	defer e.srv.Hub().Unsubscribe(sub)

	// the client compacts: the conversation plus compaction_trigger
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":`+longConversation()+`,{"type":"compaction_trigger"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, readBody(t, resp))
	}
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	// the oldest turns were dropped to make the summarization prompt fit
	prompt := strings.SplitN(e.readOut("prompt"), "\n====\n", 2)[0]
	if strings.Contains(prompt, "AAAAAAAAAA") || strings.Contains(prompt, "BBBBBBBBBB") || !strings.Contains(prompt, "turn three: weather in Paris, 3 days?") {
		t.Errorf("the wrong turns were kept:\n%s", prompt)
	}
	if !strings.HasSuffix(prompt, "<|im_start|>user\n"+compactInstruction+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Errorf("the summarization instruction is not the last user message (and thinking is not off):\n%s", prompt[len(prompt)-300:])
	}
	var dropped any
	for stop := false; !stop; {
		select {
		case ev := <-sub.C:
			if ev.Type == "run.start" {
				var m map[string]any
				json.Unmarshal(ev.Data, &m)
				dropped = m["compaction"].(map[string]any)["dropped_turns"]
			}
		default:
			stop = true
		}
	}
	if dropped != float64(2) {
		t.Errorf("dropped_turns %v", dropped)
	}

	// the stream: one compaction item, added and done, then completed; nothing else
	var kinds []string
	var item map[string]any
	for _, ev := range res.events {
		switch ev.Name {
		case "response.output_item.added", "response.output_item.done":
			it := ev.obj()["item"].(map[string]any)
			kinds = append(kinds, ev.Name+":"+it["type"].(string))
			if ev.Name == "response.output_item.done" {
				item = it
			}
		case "response.output_text.delta", "response.reasoning_text.delta":
			t.Errorf("a compaction streams no text: %s", ev.Name)
		}
	}
	if strings.Join(kinds, ",") != "response.output_item.added:compaction,response.output_item.done:compaction" ||
		res.events[len(res.events)-1].Name != "response.completed" {
		t.Fatalf("events %v (last %s)", kinds, res.events[len(res.events)-1].Name)
	}
	if !strings.HasPrefix(item["id"].(string), "cmp_") {
		t.Errorf("id %v", item["id"])
	}
	enc := item["encrypted_content"].(string)
	raw, err := base64.StdEncoding.DecodeString(enc)
	var payload struct {
		V       int
		Summary string
	}
	if err != nil || json.Unmarshal(raw, &payload) != nil || payload.V != 1 || payload.Summary != "Summary: weather in Paris, 3 days." {
		t.Fatalf("encrypted_content %q: %s %v", enc, raw, err)
	}
	comp := res.named("response.completed")[0].obj()["response"].(map[string]any)
	if out := comp["output"].([]any); len(out) != 1 || comp["status"] != "completed" || comp["usage"].(map[string]any)["input_tokens"].(float64) < 100 {
		t.Errorf("completed %v", comp)
	}

	// later requests expand the item into the summary, at its place; the client keeps what it retains after it
	itemJSON, _ := json.Marshal(item)
	next := `{"model":"m","stream":false,"input":[` + string(itemJSON) + `,{"type":"message","role":"user","content":"and tomorrow?"}]}`
	resp = e.post("/v1/responses", next)
	if resp.StatusCode != 200 {
		t.Fatalf("expand: %d %s", resp.StatusCode, readBody(t, resp))
	}
	resp.Body.Close()
	p2 := strings.Split(e.readOut("prompt"), "\n====\n")[1]
	want := "<|im_start|>user\nSummary of the earlier conversation: Summary: weather in Paris, 3 days.<|im_end|>\n<|im_start|>user\nand tomorrow?<|im_end|>\n"
	if !strings.Contains(p2, want) {
		t.Errorf("second prompt lacks the expanded summary:\n%s", p2)
	}
	// the same with context_compaction, and in the middle of a conversation
	ctxItem := strings.Replace(string(itemJSON), `"type":"compaction"`, `"type":"context_compaction"`, 1)
	resp = e.post("/v1/responses", `{"model":"m","stream":false,"input":[{"type":"message","role":"user","content":"first"},`+ctxItem+`,{"type":"message","role":"user","content":"last"}]}`)
	resp.Body.Close()
	p3 := strings.Split(e.readOut("prompt"), "\n====\n")[2]
	if !strings.Contains(p3, "first<|im_end|>\n<|im_start|>user\nSummary of the earlier conversation: Summary: weather in Paris, 3 days.<|im_end|>\n<|im_start|>user\nlast") {
		t.Errorf("third prompt:\n%s", p3)
	}

	// non-streaming compaction
	resp = e.post("/v1/responses", `{"model":"m","stream":false,"input":`+longConversation()+`,{"type":"compaction_trigger"}]}`)
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	out, _ := r["output"].([]any)
	if r["status"] != "completed" || len(out) != 1 || out[0].(map[string]any)["type"] != "compaction" || out[0].(map[string]any)["encrypted_content"] == "" {
		t.Errorf("non-streaming %v", r)
	}
}

func TestCompactionRefusals(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "summary", ctx: 1500})
	e.waitState("m", StateUp, 10*time.Second)
	for name, c := range map[string]struct{ body, code string }{
		"foreign compaction":   {`{"model":"m","input":[{"type":"compaction","encrypted_content":"gAAAAABopenai-encrypted"},{"type":"message","role":"user","content":"x"}]}`, "invalid_compaction"},
		"not base64 json":      {`{"model":"m","input":[{"type":"compaction","encrypted_content":"` + base64.StdEncoding.EncodeToString([]byte("plain text")) + `"},{"type":"message","role":"user","content":"x"}]}`, "invalid_compaction"},
		"no encrypted_content": {`{"model":"m","input":[{"type":"compaction"},{"type":"message","role":"user","content":"x"}]}`, "invalid_compaction"},
		"trigger not last":     {`{"model":"m","input":[{"type":"message","role":"user","content":"x"},{"type":"compaction_trigger"},{"type":"message","role":"user","content":"y"}]}`, "invalid_compaction"},
		"nothing to compact":   {`{"model":"m","input":[{"type":"compaction_trigger"}]}`, "empty_input"},
		"latest turn too big":  {`{"model":"m","input":[{"type":"message","role":"user","content":"` + strings.Repeat("Z", 2000) + `"},{"type":"compaction_trigger"}]}`, "context_length_exceeded"},
	} {
		resp := e.post("/v1/responses", c.body)
		body := readBody(t, resp)
		if resp.StatusCode != 400 || !strings.Contains(body, `"code":"`+c.code+`"`) {
			t.Errorf("%s: %d %s", name, resp.StatusCode, body)
		}
	}
	// a compaction_trigger on a request that sets sampling is still refused
	resp := e.post("/v1/responses", `{"model":"m","temperature":0.2,"input":[{"type":"message","role":"user","content":"x"},{"type":"compaction_trigger"}]}`)
	if resp.StatusCode != 400 {
		t.Errorf("temperature with a trigger: %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestCompactionOnAShim(t *testing.T) {
	e := startMux(t, epSpec{name: "s", scenario: "shim_summary", kind: KindShim, ctx: 128000})
	e.waitState("s", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"s","stream":true,"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{}}}],"input":[`+
		`{"type":"message","role":"user","content":"weather in Paris?"},{"type":"function_call","name":"get_weather","arguments":"{}","call_id":"c1"},`+
		`{"type":"function_call_output","call_id":"c1","output":"sunny"},{"type":"compaction_trigger"}]}`)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	var item map[string]any
	for _, ev := range res.named("response.output_item.done") {
		item = ev.obj()["item"].(map[string]any)
	}
	if len(res.named("response.output_item.done")) != 1 || item["type"] != "compaction" {
		t.Fatalf("events %v", res.events)
	}
	raw, _ := base64.StdEncoding.DecodeString(item["encrypted_content"].(string))
	var payload struct{ Summary string }
	json.Unmarshal(raw, &payload)
	if payload.Summary != "The user wants weather data for Paris; the plan is to call get_weather." {
		t.Errorf("summary %q (reasoning must not be in it)", payload.Summary)
	}
	var run struct {
		Messages []map[string]any `json:"messages"`
		Tools    []any            `json:"tools"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("shim.runs"), "\n", 2)[0]), &run)
	last := run.Messages[len(run.Messages)-1]
	if last["role"] != "user" || last["content"] != compactInstruction || len(run.Tools) != 0 || len(run.Messages) != 4 {
		t.Errorf("run %+v", run)
	}
}

func TestFitCompactionDropsOldestFirst(t *testing.T) {
	mk := func(role, text string) chatMsg { return chatMsg{Role: role, Content: text} }
	pr := &prepared{messages: []chatMsg{
		mk("system", "sys"),
		mk("user", strings.Repeat("1", 300)), mk("assistant", strings.Repeat("a", 100)),
		mk("user", strings.Repeat("2", 300)), mk("assistant", strings.Repeat("b", 100)),
		mk("user", strings.Repeat("3", 300)), mk("assistant", strings.Repeat("c", 100)),
		mk("user", "four"),
	}}
	var m Model = fakeModel{}
	turns := splitTurns(pr.messages)
	size := func(drop int) int {
		v := pr.varsFor(turns.messages(drop), nil)
		v["enable_thinking"] = false
		p, _ := m.Render(v)
		ids, _ := m.Encode(p)
		return len(ids)
	}
	if turns.maxDrop() != 3 || !(size(0) > size(1) && size(1) > size(2) && size(2) > size(3)) {
		t.Fatalf("turns: max drop %d, sizes %d %d %d %d", turns.maxDrop(), size(0), size(1), size(2), size(3))
	}
	for want := 0; want <= 3; want++ {
		// the smallest window in which `want` dropped turns is enough (and one fewer is not)
		ctx := size(want) + 64
		for i := 0; i < 6; i++ {
			ctx = size(want) + compactReserve(ctx)
		}
		prompt, ids, dropped, err := fitCompaction(m, pr, ctx)
		if err != nil || dropped != want {
			t.Fatalf("ctx %d: dropped %d err %v (want %d)", ctx, dropped, err, want)
		}
		if len(ids)+compactReserve(ctx) > ctx || !strings.Contains(prompt, "four") || !strings.Contains(prompt, "<|im_start|>system\nsys") {
			t.Errorf("ctx %d: %d tokens", ctx, len(ids))
		}
		for i, mark := range []string{"1111111111", "2222222222", "3333333333"} {
			if has := strings.Contains(prompt, mark); has != (i >= want) {
				t.Errorf("ctx %d: turn %d kept = %v", ctx, i+1, has)
			}
		}
	}
	if _, _, _, err := fitCompaction(m, pr, size(3)+compactReserve(0)-1); err == nil || err.Code != "context_length_exceeded" {
		t.Errorf("a window smaller than the latest turn and the summary: %v", err)
	}
}
