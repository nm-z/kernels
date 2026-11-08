package mux

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func codexBody(model string, extra string) string {
	return `{"model":"` + model + `","stream":true,"instructions":"You are Codex.","reasoning":{"effort":"low"},` +
		`"tool_choice":"auto","parallel_tool_calls":true,"store":false,"include":[],"prompt_cache_key":"k","text":{"verbosity":"low"},` +
		`"tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}},` +
		`{"type":"custom","name":"apply_patch","description":"p","format":{"type":"grammar","syntax":"lark","definition":"start: x"}}],` +
		`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]}]` + extra + `}`
}

func TestResponsesStreaming(t *testing.T) {
	e := startMux(t, epSpec{name: "m1", scenario: "normal"})
	e.waitState("m1", StateUp, 10*time.Second)

	sub := e.srv.Hub().Subscribe()
	defer e.srv.Hub().Unsubscribe(sub)

	resp := e.post("/v1/responses", codexBody("m1", ""))
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status %d %s: %s", resp.StatusCode, resp.Header.Get("Content-Type"), readBody(t, resp))
	}
	res := readSSE(t, resp.Body)
	resp.Body.Close()

	var names []string
	for _, ev := range res.events {
		names = append(names, ev.Name)
	}
	t.Logf("events: %d, stats: %d", len(names), len(res.named("response.ai_mux.stats")))
	if names[0] != "response.created" || names[1] != "response.in_progress" || names[len(names)-1] != "response.completed" {
		t.Fatalf("first/last events: %v … %v", names[:2], names[len(names)-3:])
	}
	// every event's JSON carries its own type
	for _, ev := range res.events {
		if ev.obj()["type"] != ev.Name {
			t.Fatalf("event %s has type %v", ev.Name, ev.obj()["type"])
		}
	}

	// reasoning
	var reasoning, answer strings.Builder
	for _, ev := range res.named("response.reasoning_text.delta") {
		o := ev.obj()
		if o["content_index"] != float64(0) || o["item_id"] == "" {
			t.Fatalf("reasoning delta %v", o)
		}
		reasoning.WriteString(o["delta"].(string))
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

	// items, in order
	var types []string
	var items []map[string]any
	for _, ev := range res.named("response.output_item.done") {
		it := ev.obj()["item"].(map[string]any)
		types = append(types, it["type"].(string))
		items = append(items, it)
	}
	if strings.Join(types, ",") != "reasoning,message,function_call,custom_tool_call" {
		t.Fatalf("items %v", types)
	}
	for i, ev := range res.named("response.output_item.done") {
		if ev.obj()["output_index"] != float64(i) {
			t.Errorf("output_index %v for item %d", ev.obj()["output_index"], i)
		}
	}
	if rs := items[0]["content"].([]any)[0].(map[string]any); rs["type"] != "reasoning_text" || !strings.Contains(rs["text"].(string), "caf\u00e9") {
		t.Errorf("reasoning item %v", items[0])
	}
	if items[1]["role"] != "assistant" || items[1]["content"].([]any)[0].(map[string]any)["text"] != "Let me check." {
		t.Errorf("message item %v", items[1])
	}
	if items[2]["name"] != "get_weather" || items[2]["arguments"] != `{"city":"Paris","days":3}` {
		t.Errorf("function_call %v", items[2])
	}
	if items[3]["name"] != "apply_patch" || items[3]["input"] != "*** Begin Patch\n*** End Patch" {
		t.Errorf("custom_tool_call %v", items[3])
	}
	// no tool XML reached the client as text
	for _, ev := range res.named("response.output_text.delta") {
		if strings.Contains(string(ev.Data), "tool_call") {
			t.Errorf("tool XML leaked: %s", ev.Data)
		}
	}

	// stats: at most every 250 ms, at least every 500 ms, final one with idle_pct before completed
	stats := res.named("response.ai_mux.stats")
	if len(stats) < 4 {
		t.Fatalf("only %d stats events", len(stats))
	}
	for i := 1; i < len(stats)-1; i++ {
		gap := stats[i].At.Sub(stats[i-1].At)
		if gap < 200*time.Millisecond || gap > 750*time.Millisecond {
			t.Errorf("stats gap %d: %s", i, gap)
		}
	}
	for i, ev := range stats {
		o := ev.obj()
		pf, dc := o["prefill"].(map[string]any), o["decode"].(map[string]any)
		if (o["phase"] != "prefill" && o["phase"] != "decode") || pf["total"] == nil || pf["done"] == nil || pf["tps"] == nil || dc["tokens"] == nil || dc["tps"] == nil || o["ctx"] != float64(100000) {
			t.Errorf("stats %d: %s", i, ev.Data)
		}
		// idle_pct is on every update: null until the engine has one, then live, then the final value
		if v, has := o["idle_pct"]; !has || (i == len(stats)-1 && v != 4.2) {
			t.Errorf("stats %d idle_pct = %v (present %v): %s", i, v, has, ev.Data)
		}
	}
	sawLive := false
	for _, ev := range stats[1 : len(stats)-1] {
		if ev.obj()["idle_pct"] == 3.5 {
			sawLive = true
		}
	}
	if stats[0].obj()["idle_pct"] != nil || !sawLive {
		t.Errorf("idle_pct should be null at first and then live (3.5): first %s", stats[0].Data)
	}
	lastStats := stats[len(stats)-1].obj()
	if lastStats["idle_pct"] != 4.2 || lastStats["decode"].(map[string]any)["tps"] != 67.8 || lastStats["prefill"].(map[string]any)["tps"] != 1234.5 || lastStats["phase"] != "decode" {
		t.Errorf("final stats %s", stats[len(stats)-1].Data)
	}
	if idx := len(res.events) - 2; res.events[idx].Name != "response.ai_mux.stats" {
		t.Errorf("final stats is not right before response.completed: %v", names[len(names)-3:])
	}
	// the first stats event has the prompt total and decode 0
	if st0 := stats[0].obj(); st0["phase"] != "prefill" || st0["prefill"].(map[string]any)["total"].(float64) < 10 {
		t.Errorf("first stats %s", stats[0].Data)
	}

	// completed
	comp := res.named("response.completed")[0].obj()["response"].(map[string]any)
	usage := comp["usage"].(map[string]any)
	genWant := float64(len(mustEncode(fakeAnswer)))
	if usage["output_tokens"] != genWant || usage["input_tokens"].(float64) < 20 || usage["total_tokens"] != "" {
		t.Errorf("usage %v (want %v output tokens)", usage, genWant)
	}
	if rt := usage["output_tokens_details"].(map[string]any)["reasoning_tokens"]; rt != "" {
		t.Errorf("reasoning tokens %v", rt)
	}
	if comp["status"] != "completed" || len(comp["output"].([]any)) != 4 || !strings.HasPrefix(comp["id"].(string), "resp_") || comp["model"] != "m1" {
		t.Errorf("response %v", comp)
	}

	// what the engine was asked: the rendered prompt, the stop ids, max_new
	prompt := e.readOut("prompt")
	for _, w := range []string{"<|im_start|>system\n[effort:low]\n[tool:{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"", "[tool:{\"type\":\"function\",\"function\":{\"name\":\"apply_patch\"",
		"You are Codex.", "<|im_start|>user\nweather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n"} {
		if !strings.Contains(prompt, w) {
			t.Errorf("prompt lacks %q:\n%s", w, prompt)
		}
	}
	var rj struct {
		IDs    int      `json:"ids"`
		MaxNew int      `json:"max_new"`
		Stop   []uint32 `json:"stop"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	if rj.MaxNew != 8192 || len(rj.Stop) != 1 || rj.Stop[0] != 256 || rj.IDs != int(usage["input_tokens"].(float64)) {
		t.Errorf("run %+v", rj)
	}

	// the endpoint shows the last run
	waitFor(t, 2*time.Second, "endpoint back to up", func() bool { return e.srv.Service("m1").Info().State == StateUp })
	var eps struct{ Data []ServiceInfo }
	e.getJSON("/v1/services", &eps)
	in := eps.Data[0]
	if in.Name != "m1" || in.State != "up" || in.Last == nil || in.Last.PrefillTPS == nil || *in.Last.PrefillTPS != 1234.5 || *in.Last.DecodeTPS != 67.8 || *in.Last.IdlePct != 4.2 ||
		in.PID == 0 || in.Ctx != 100000 || in.Slots != 1 || in.Device != "test device" || in.Since == "" || !strings.HasSuffix(in.Log, "m1.log") {
		t.Errorf("endpoint info %+v", in)
	}
	if lg, _ := os.ReadFile(in.Log); !strings.Contains(string(lg), "mux:") || !strings.Contains(string(lg), "start pid=") {
		t.Errorf("log %s: %q", in.Log, lg)
	}

	// the observer saw the run
	seen := map[string]int{}
drain:
	for {
		select {
		case ev := <-sub.C:
			seen[ev.Type]++
		default:
			break drain
		}
	}
	for _, typ := range []string{"run.start", "run.prefill", "run.stats", "run.tokens", "run.end", "service.state"} {
		if seen[typ] == 0 {
			t.Errorf("no %s event (saw %v)", typ, seen)
		}
	}
}

func mustEncode(s string) []uint32 { ids, _ := fakeModel{}.Encode(s); return ids }

func TestResponsesNonStreaming(t *testing.T) {
	e := startMux(t, epSpec{name: "m1", scenario: "normal"})
	e.waitState("m1", StateUp, 10*time.Second)
	body := strings.Replace(codexBody("m1", ""), `"stream":true`, `"stream":false`, 1)
	resp := e.post("/v1/responses", body)
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("status %d: %s", resp.StatusCode, readBody(t, resp))
	}
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	out := r["output"].([]any)
	if r["status"] != "completed" || r["object"] != "response" || len(out) != 4 || out[0].(map[string]any)["type"] != "reasoning" || out[3].(map[string]any)["type"] != "custom_tool_call" {
		t.Fatalf("response %v", r)
	}
	if am := r["mux"].(map[string]any); am["decode_tps"] != 67.8 || am["idle_pct"] != 4.2 {
		t.Errorf("mux %v", am)
	}
	// no stats events and no SSE in non-streaming mode
	if r["usage"].(map[string]any)["output_tokens"] != float64(len(mustEncode(fakeAnswer))) {
		t.Errorf("usage %v", r["usage"])
	}
}

func TestResponsesErrors(t *testing.T) {
	e := startMux(t, epSpec{name: "m1", scenario: "normal"}, epSpec{name: "m2", scenario: "normal", efforts: []string{"low"}})
	e.waitState("m1", StateUp, 10*time.Second)
	e.waitState("m2", StateUp, 10*time.Second)
	cases := []struct {
		name   string
		body   string
		status int
		code   string
		param  string
	}{}
	for _, f := range []string{"temperature", "top_p", "top_k", "seed", "max_output_tokens", "presence_penalty", "frequency_penalty", "stop"} {
		v := "0.5"
		if f == "stop" {
			v = `["x"]`
		}
		cases = append(cases, struct {
			name   string
			body   string
			status int
			code   string
			param  string
		}{f, `{"model":"m1","input":"hi","` + f + `":` + v + `}`, 400, "unsupported_parameter", f})
	}
	cases = append(cases, []struct {
		name   string
		body   string
		status int
		code   string
		param  string
	}{
		{"effort not offered", `{"model":"m2","input":"hi","reasoning":{"effort":"high"}}`, 400, "unsupported_reasoning_effort", "reasoning.effort"},
		{"unknown model", `{"model":"nope","input":"hi"}`, 404, "model_not_found", "model"},
		{"not json", `{`, 400, "invalid_json", ""},
		{"too long", `{"model":"m1","input":"` + strings.Repeat("x", 100001) + `"}`, 400, "context_length_exceeded", "input"},
	}...)
	for _, c := range cases {
		resp := e.post("/v1/responses", c.body)
		b := readBody(t, resp)
		if resp.StatusCode != c.status {
			t.Errorf("%s: status %d: %s", c.name, resp.StatusCode, b)
			continue
		}
		if c.code == "" {
			continue
		}
		var er struct {
			Error struct {
				Message string  `json:"message"`
				Type    string  `json:"type"`
				Code    string  `json:"code"`
				Param   *string `json:"param"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(b), &er); err != nil || er.Error.Message == "" || er.Error.Type == "" || er.Error.Code != c.code {
			t.Errorf("%s: error body %s", c.name, b)
		}
		if c.param != "" && (er.Error.Param == nil || *er.Error.Param != c.param) {
			t.Errorf("%s: param in %s", c.name, b)
		}
	}
	// a refused request never reached the engine
	if e.readOut("prompt") != "" {
		t.Errorf("a refused request reached the engine: %s", e.readOut("prompt"))
	}
	// GET /v1/models
	var models struct{ Data []map[string]any }
	e.getJSON("/v1/models", &models)
	if len(models.Data) != 2 || models.Data[0]["id"] != "m1" || models.Data[0]["ctx"] != float64(100000) || models.Data[0]["state"] != "up" || models.Data[0]["vision"] != true || len(models.Data[1]["efforts"].([]any)) != 1 {
		t.Errorf("models %v", models.Data)
	}
}

func TestCancelOnClientDisconnect(t *testing.T) {
	e := startMux(t, epSpec{name: "h", scenario: "hang"})
	e.waitState("h", StateUp, 10*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	resp := e.postCtx(ctx, "/v1/responses", `{"model":"h","stream":true,"input":"hi"}`)
	// read until the first output text, then walk away
	got := make(chan struct{})
	go func() {
		defer close(got)
		buf := make([]byte, 4096)
		var acc string
		for !strings.Contains(acc, "response.output_item.added") && !strings.Contains(acc, "output_text.delta") {
			n, err := resp.Body.Read(buf)
			acc += string(buf[:n])
			if err != nil {
				return
			}
		}
	}()
	select {
	case <-got:
	case <-time.After(5 * time.Second):
		t.Fatal("no output")
	}
	waitFor(t, 3*time.Second, "endpoint busy", func() bool { return e.srv.Service("h").Info().State == StateBusy })
	cancel()
	resp.Body.Close()
	waitFor(t, 5*time.Second, "engine to receive cancel", func() bool { return strings.Contains(e.readOut("cancel"), "r1") })
	waitFor(t, 5*time.Second, "engine to stop the run", func() bool { return strings.Contains(e.readOut("cancelled"), "yes") })
	waitFor(t, 5*time.Second, "endpoint idle again", func() bool {
		in := e.srv.Service("h").Info()
		return in.State == StateUp && in.Active == 0
	})
	if st := e.srv.Service("h").Info(); st.Reason != "" {
		t.Errorf("a cancelled run marked the endpoint: %+v", st)
	}
	// the endpoint takes the next request
	ctx2, cancel2 := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel2()
	resp2 := e.postCtx(ctx2, "/v1/responses", `{"model":"h","stream":true,"input":"again"}`)
	buf := make([]byte, 64)
	if n, _ := resp2.Body.Read(buf); n == 0 {
		t.Error("no answer to the next request")
	}
	resp2.Body.Close()
}

func TestRule5(t *testing.T) {
	for _, c := range []struct{ scenario, why string }{
		{"nostats", "no stats line during generation"},
		{"noprefill", "no prefill line during the prompt"},
		{"noidle", "done has no idle_pct"},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			e := startMux(t, epSpec{name: "bad", scenario: c.scenario}, epSpec{name: "good", scenario: "slow"})
			e.waitState("bad", StateUp, 10*time.Second)
			sub := e.srv.Hub().Subscribe()
			defer e.srv.Hub().Unsubscribe(sub)
			pid := e.srv.Service("bad").Info().PID

			resp := e.post("/v1/responses", `{"model":"bad","stream":true,"input":"hi"}`)
			res := readSSE(t, resp.Body)
			resp.Body.Close()
			if len(res.named("response.completed")) != 1 {
				t.Errorf("the client's response was not completed")
			}
			in := e.waitState("bad", StateFailed, 5*time.Second)
			if !strings.Contains(in.Reason, "rule 5") || !strings.Contains(in.Reason, c.why) {
				t.Errorf("reason %q", in.Reason)
			}
			// visible in /v1/services …
			var eps struct{ Data []ServiceInfo }
			e.getJSON("/v1/services", &eps)
			if eps.Data[0].State != "failed" || !strings.Contains(eps.Data[0].Reason, c.why) || eps.Data[0].Slots != 0 {
				t.Errorf("/v1/services: %+v", eps.Data[0])
			}
			// … and in /v1/events
			found := false
		drain:
			for {
				select {
				case ev := <-sub.C:
					if ev.Type == "service.state" && strings.Contains(string(ev.Data), `"state":"failed"`) && strings.Contains(string(ev.Data), c.why) {
						found = true
					}
				default:
					break drain
				}
			}
			if !found {
				t.Error("no endpoint.state failed event with the reason")
			}
			// the process is gone and the endpoint takes no requests
			waitFor(t, 5*time.Second, "engine process to be killed", func() bool { return !pidAlive(pid) })
			resp = e.post("/v1/responses", `{"model":"bad","input":"hi"}`)
			if resp.StatusCode != 503 {
				t.Errorf("request to a failed endpoint: %d %s", resp.StatusCode, readBody(t, resp))
			}
			// the other endpoint is unaffected
			if e.srv.Service("good").Info().State != StateUp {
				t.Error("other endpoint affected")
			}
		})
	}
}

func TestMaxNewIsIncomplete(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "maxnew", maxNew: 40})
	e.waitState("m", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":"hi","reasoning":{"effort":"none"}}`)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	if len(res.named("response.completed")) != 0 || len(res.named("response.failed")) != 0 {
		t.Fatalf("events %v", res.events[len(res.events)-1].Name)
	}
	inc := res.named("response.incomplete")
	if len(inc) != 1 {
		t.Fatalf("no response.incomplete")
	}
	r := inc[0].obj()["response"].(map[string]any)
	if r["status"] != "incomplete" || r["incomplete_details"].(map[string]any)["reason"] != "max_output_tokens" || r["usage"].(map[string]any)["output_tokens"] != float64(40) {
		t.Errorf("response %v", r)
	}
	// with the think block closed by the prompt (effort none) the 40 a's are the answer
	var rj struct {
		MaxNew int `json:"max_new"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	if rj.MaxNew != 40 {
		t.Errorf("max_new %d", rj.MaxNew)
	}
	if !strings.Contains(e.readOut("prompt"), "<think>\n\n</think>\n\n") {
		t.Errorf("effort none did not close the think block:\n%s", e.readOut("prompt"))
	}
	// non-streaming
	resp = e.post("/v1/responses", `{"model":"m","input":"hi"}`)
	var r2 map[string]any
	json.NewDecoder(resp.Body).Decode(&r2)
	resp.Body.Close()
	if r2["status"] != "incomplete" {
		t.Errorf("non-streaming: %v", r2["status"])
	}
}

func TestMaxNewIsBoundedByContext(t *testing.T) {
	// ctx 120: the prompt takes about 50, so max_new = ctx - prompt < the engine's 8192
	e := startMux(t, epSpec{name: "m", scenario: "maxnew", ctx: 120})
	e.waitState("m", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":"hi"}`)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	var rj struct {
		IDs    int `json:"ids"`
		MaxNew int `json:"max_new"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	if rj.MaxNew != 120-rj.IDs || rj.MaxNew <= 0 || len(res.named("response.incomplete")) != 1 {
		t.Errorf("run %+v, events %d", rj, len(res.events))
	}
}

func TestEngineErrorFailsResponse(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "error"})
	e.waitState("m", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":"hi"}`)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	f := res.named("response.failed")
	if len(f) != 1 || !strings.Contains(string(f[0].Data), "kernel crashed") || !strings.Contains(string(f[0].Data), `"status":"failed"`) {
		t.Fatalf("events %v", res.events)
	}
	// a run error does not drop the engine
	if st := e.srv.Service("m").Info().State; st != StateUp {
		t.Errorf("state %s", st)
	}
	resp = e.post("/v1/responses", `{"model":"m","input":"hi"}`)
	if resp.StatusCode != 502 {
		t.Errorf("non-streaming status %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestQueueOneSlot(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "slow"})
	e.waitState("q", StateUp, 10*time.Second)
	var wg sync.WaitGroup
	results := make([]sseResult, 3)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := e.post("/v1/responses", `{"model":"q","stream":true,"input":"hi"}`)
			results[i] = readSSE(t, resp.Body)
			resp.Body.Close()
		}()
	}
	waitFor(t, 3*time.Second, "two requests to queue", func() bool { return e.srv.Service("q").Info().Queued == 2 })
	if in := e.srv.Service("q").Info(); in.Active != 1 || in.State != StateBusy {
		t.Errorf("info %+v", in)
	}
	wg.Wait()
	for i, r := range results {
		// the fake engine fails a run that overlaps another on its one slot
		if len(r.named("response.completed")) != 1 {
			t.Errorf("request %d: %v", i, r.events[len(r.events)-1].Name)
		}
	}
}

func TestTwoSlotsRunTogether(t *testing.T) {
	e := startMux(t, epSpec{name: "q", scenario: "slow", slots: 2})
	e.waitState("q", StateUp, 10*time.Second)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := e.post("/v1/responses", `{"model":"q","stream":true,"input":"hi"}`)
			readSSE(t, resp.Body)
			resp.Body.Close()
		}()
	}
	waitFor(t, 3*time.Second, "two active runs", func() bool { return e.srv.Service("q").Info().Active == 2 })
	wg.Wait()
	if e.srv.Service("q").Info().Slots != 2 {
		t.Error("slots not reported")
	}
}

func TestUnknownPeerCannotRegister(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "normal"})
	e.waitState("m", StateUp, 10*time.Second)
	// this test process is not in the endpoint's process tree
	for _, name := range []string{"m", "nope"} {
		resp := rogueRegister(t, filepath.Join(e.dir, "s.sock"), name)
		if !strings.Contains(resp, `"op":"error"`) || !strings.Contains(resp, "register refused") {
			t.Errorf("rogue register for %s: %q", name, resp)
		}
	}
	// and the real engine is still the endpoint's
	if st := e.srv.Service("m").Info(); st.State != StateUp {
		t.Errorf("state %s", st.State)
	}
}

func TestPositionsBelowCtxRefused(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "smallpos"})
	in := e.waitState("m", StateFailed, 10*time.Second)
	if !strings.Contains(in.Reason, "exit code 3") {
		t.Errorf("reason %q", in.Reason)
	}
	if got := e.readOut("refused"); !strings.Contains(got, "positions 10 is less than the service's ctx") {
		t.Errorf("engine was told %q", got)
	}
}

func TestProcessExitFailsAtOnceAndRestarts(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "normal", restart: "on-failure"})
	in := e.waitState("m", StateUp, 10*time.Second)
	old := in.PID
	syscallKill(old)
	in = e.waitState("m", StateFailed, 3*time.Second)
	if !strings.Contains(in.Reason, "killed by signal") || in.PID != 0 {
		t.Errorf("info %+v", in)
	}
	// restart policy: back up after the (shortened) backoff, as a new process
	in = e.waitState("m", StateUp, 10*time.Second)
	if in.PID == old || in.PID == 0 {
		t.Errorf("not restarted: %+v", in)
	}
	lg, _ := os.ReadFile(in.Log)
	if strings.Count(string(lg), "start pid=") != 2 || !strings.Contains(string(lg), "exit pid="+strconv.Itoa(old)+": killed by signal") {
		t.Errorf("log:\n%s", lg)
	}
}

func TestRestartNoStaysFailedUntilUp(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "normal", restart: "no"})
	in := e.waitState("m", StateUp, 10*time.Second)
	syscallKill(in.PID)
	e.waitState("m", StateFailed, 3*time.Second)
	time.Sleep(600 * time.Millisecond) // longer than the backoff
	if st := e.srv.Service("m").Info().State; st != StateFailed {
		t.Fatalf("restarted with restart: no (%s)", st)
	}
	resp := e.post("/v1/services/m/up", "")
	resp.Body.Close()
	e.waitState("m", StateUp, 10*time.Second)
}

func TestDownKillsEverythingAndRewritesYAML(t *testing.T) {
	dir := shortDir(t)
	cfgPath := filepath.Join(dir, "mux.yaml")
	out := filepath.Join(dir, "out")
	os.MkdirAll(out, 0o755)
	svc := "kind: engine\ngguf: fake.gguf\nctx: 1000\ncommand: [" + os.Args[0] + "]\nenv: {MUX_FAKE_ENGINE: spawn, MUX_FAKE_OUT: " + out + "}\npost_stop: [sh, -c, \"echo stopped > " + filepath.Join(out, "post_stop") + "\"]\nrestart: no\n"
	writeService(t, filepath.Join(dir, "svcs"), "m", svc)
	yaml := "# test\nlisten: 127.0.0.1:0\nsocket: " + filepath.Join(dir, "s.sock") + "\nservices_dir: svcs\nservices:\n  m: {up: true} # keep\n"
	os.WriteFile(cfgPath, []byte(yaml), 0o644)
	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	e := startMuxOpts(t, Options{Config: cfg})
	e.dir, e.out = dir, out
	in := e.waitState("m", StateUp, 10*time.Second)
	waitFor(t, 3*time.Second, "child pid", func() bool { return e.readOut("child.pid") != "" })
	child, _ := strconv.Atoi(strings.TrimSpace(e.readOut("child.pid")))
	if !pidAlive(child) || !pidAlive(in.PID) {
		t.Fatalf("engine %d / child %d not running", in.PID, child)
	}
	resp := e.post("/v1/services/m/down", "")
	if resp.StatusCode != 200 {
		t.Fatalf("down: %d %s", resp.StatusCode, readBody(t, resp))
	}
	resp.Body.Close()
	e.waitState("m", StateDown, 10*time.Second)
	if pidAlive(in.PID) || pidAlive(child) {
		t.Errorf("engine %v / child %v survived down", pidAlive(in.PID), pidAlive(child))
	}
	waitFor(t, 3*time.Second, "post_stop", func() bool { return strings.Contains(e.readOut("post_stop"), "stopped") })
	b, _ := os.ReadFile(cfgPath)
	if !strings.Contains(string(b), "{up: false} # keep") || !strings.Contains(string(b), "# test") {
		t.Errorf("yaml:\n%s", b)
	}
	if st := e.srv.Service("m").Info(); st.Up || st.PID != 0 {
		t.Errorf("info %+v", st)
	}
	// requests to a down endpoint: 503
	resp = e.post("/v1/responses", `{"model":"m","input":"hi"}`)
	if resp.StatusCode != 503 {
		t.Errorf("status %d", resp.StatusCode)
	}
	resp.Body.Close()
	// and up brings it back
	resp = e.post("/v1/services/m/up", "")
	resp.Body.Close()
	in2 := e.waitState("m", StateUp, 10*time.Second)
	if in2.PID == in.PID {
		t.Error("same pid after up")
	}
	b, _ = os.ReadFile(cfgPath)
	if !strings.Contains(string(b), "{up: true} # keep") {
		t.Errorf("yaml after up:\n%s", b)
	}
	// shutdown stops everything
	e.srv.Shutdown()
	if pidAlive(in2.PID) {
		t.Error("engine survived shutdown")
	}
	for _, r := range []struct{ method, path string }{{"GET", "/v1/services/zzz/up"}, {"POST", "/v1/services/zzz/up"}, {"POST", "/v1/services/m/sideways"}} {
		req, _ := http.NewRequest(r.method, e.base+r.path, nil)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	}
}

func TestEventsStream(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "slow"})
	e.waitState("m", StateUp, 10*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var results [2]chan sseResult
	for i := range results {
		results[i] = make(chan sseResult, 1)
		req, _ := http.NewRequestWithContext(ctx, "GET", e.base+"/v1/events", nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		go func() { results[i] <- readSSE(t, resp.Body) }()
	}
	time.Sleep(100 * time.Millisecond)
	resp := e.post("/v1/responses", `{"model":"m","stream":true,"input":"hi"}`)
	readSSE(t, resp.Body)
	resp.Body.Close()
	time.Sleep(200 * time.Millisecond)
	cancel()
	for i := range results {
		r := <-results[i]
		have := map[string]int{}
		for _, ev := range r.events {
			have[ev.Name]++
			if ev.obj()["time"] == nil || ev.obj()["type"] != ev.Name {
				t.Errorf("observer %d: event %s %s", i, ev.Name, ev.Data)
			}
			if strings.Contains(string(ev.Data), "quick answer") {
				t.Errorf("token text leaked into /v1/events: %s", ev.Data)
			}
		}
		if have["snapshot"] != 1 || have["run.start"] != 1 || have["run.end"] != 1 || have["run.prefill"] < 2 || have["run.tokens"] < 1 || have["service.state"] < 2 {
			t.Errorf("observer %d saw %v", i, have)
		}
	}
}

func TestShutdownStopsEndpoints(t *testing.T) {
	e := startMux(t, epSpec{name: "a", scenario: "spawn"}, epSpec{name: "b", scenario: "normal"})
	a := e.waitState("a", StateUp, 10*time.Second)
	b := e.waitState("b", StateUp, 10*time.Second)
	e.srv.Shutdown()
	for _, pid := range []int{a.PID, b.PID} {
		if pidAlive(pid) {
			t.Errorf("pid %d survived Shutdown", pid)
		}
	}
	if _, err := os.Stat(filepath.Join(e.dir, "s.sock")); err == nil {
		t.Error("socket left behind")
	}
}

func TestStatsKeepComing(t *testing.T) {
	e := startMux(t, epSpec{name: "s", scenario: "silent"})
	e.waitState("s", StateUp, 10*time.Second)
	resp := e.post("/v1/responses", `{"model":"s","stream":true,"input":"hi"}`)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	stats := res.named("response.ai_mux.stats")
	if len(stats) < 4 {
		t.Fatalf("%d stats events over a 1.7 s silence", len(stats))
	}
	for i := 1; i < len(stats); i++ {
		if gap := stats[i].At.Sub(stats[i-1].At); gap > 750*time.Millisecond {
			t.Errorf("gap %d between stats events: %s", i, gap)
		}
	}
	if len(res.named("response.completed")) != 1 {
		t.Error("not completed")
	}
	if p := stats[1].obj()["prefill"].(map[string]any); p["done"] != float64(0) || p["total"].(float64) < 10 {
		t.Errorf("during the silence: %s", stats[1].Data)
	}
}

func TestImagesReachTheEngine(t *testing.T) {
	e := startMux(t, epSpec{name: "v", scenario: "slow"})
	e.waitState("v", StateUp, 10*time.Second)
	img := func(s string) string { return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(s)) }
	body := `{"model":"v","stream":true,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"compare "},` +
		`{"type":"input_image","image_url":"` + img("first") + `"},{"type":"input_image","image_url":"` + img("second") + `"}]}]}`
	resp := e.post("/v1/responses", body)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	if len(res.named("response.completed")) != 1 {
		t.Fatalf("not completed: %v", res.events[len(res.events)-1])
	}
	// each <|image_pad|> is expanded to the positions the engine said the image takes (4 in the fake)
	var rj struct {
		IDs    int `json:"ids"`
		Images []struct {
			Img string `json:"img"`
			At  int    `json:"at"`
		} `json:"images"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("run.json"), "\n", 2)[0]), &rj)
	prompt := strings.SplitN(e.readOut("prompt"), "\n====\n", 2)[0]
	pad := "<|vision_start|>" + strings.Repeat("<|image_pad|>", 4) + "<|vision_end|>"
	if strings.Count(prompt, pad) != 2 {
		t.Errorf("prompt:\n%s", prompt)
	}
	if len(rj.Images) != 2 || rj.Images[0].Img == rj.Images[1].Img || rj.Images[0].Img != hashBytes([]byte("first")) || rj.Images[1].At != rj.Images[0].At+6 {
		t.Errorf("images %+v", rj.Images)
	}
	// `at` is the position of the first pad of its image
	ids := mustEncode(prompt)
	if len(ids) != rj.IDs || ids[rj.Images[0].At] != 261 || ids[rj.Images[0].At-1] != 263 || ids[rj.Images[1].At+3] != 261 || ids[rj.Images[1].At+4] != 264 {
		t.Errorf("positions: ids %d, %+v", len(ids), rj.Images)
	}
}
