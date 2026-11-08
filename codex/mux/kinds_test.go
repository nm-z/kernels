package mux

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

const shimBody = `{"model":"s","stream":true,"instructions":"You are Codex.","reasoning":{"effort":"low"},` +
	`"tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}},` +
	`{"type":"custom","name":"apply_patch","description":"p","format":{"type":"grammar","syntax":"lark","definition":"start: x"}}],` +
	`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]}]}`

func startShim(t *testing.T, scenario string) *tenv {
	e := startMux(t, epSpec{name: "s", scenario: scenario, kind: KindShim, ctx: 128000})
	e.waitState("s", StateUp, 10*time.Second)
	return e
}

func TestShimStreaming(t *testing.T) {
	e := startShim(t, "shim")
	resp := e.post("/v1/responses", shimBody)
	res := readSSE(t, resp.Body)
	resp.Body.Close()

	var reasoning, answer strings.Builder
	for _, ev := range res.named("response.reasoning_text.delta") {
		reasoning.WriteString(ev.obj()["delta"].(string))
	}
	for _, ev := range res.named("response.output_text.delta") {
		answer.WriteString(ev.obj()["delta"].(string))
	}
	if reasoning.String() != "Let me think. café" || answer.String() != "Hello world." {
		t.Errorf("reasoning %q, answer %q", reasoning.String(), answer.String())
	}
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
	if items[2]["name"] != "get_weather" || items[2]["call_id"] != "call_shim1" || items[2]["arguments"] != `{"city":"Paris","days":3}` {
		t.Errorf("function_call %v", items[2])
	}
	if items[3]["name"] != "apply_patch" || items[3]["input"] != "*** Begin Patch\n*** End Patch" || items[3]["call_id"] != "call_shim2" {
		t.Errorf("custom_tool_call %v", items[3])
	}
	// a shim has no engine stats: one event at the first token, with ttft and the window, and no route (not an auto shim)
	stats := res.named("response.ai_mux.stats")
	if len(stats) != 1 {
		t.Fatalf("%d stats events: %v", len(stats), stats)
	}
	so := stats[0].obj()
	if so["phase"] != "output" || so["ttft_ms"] != 123.5 || so["ctx"] != float64(128000) || so["route"] != nil || so["prefill"] != nil {
		t.Errorf("stats %s", stats[0].Data)
	}
	// ... and it comes before the first delta
	first := -1
	for i, ev := range res.events {
		if ev.Name == "response.reasoning_text.delta" && first < 0 {
			first = i
		}
		if ev.Name == "response.ai_mux.stats" && first >= 0 {
			t.Errorf("stats came after the first delta")
		}
	}
	comp := res.named("response.completed")[0].obj()["response"].(map[string]any)
	if u := comp["usage"].(map[string]any); u["input_tokens"] != float64(11) || u["output_tokens"] != float64(22) || u["total_tokens"] != "" {
		t.Errorf("usage %v", u)
	}
	if comp["status"] != "completed" || comp["model"] != "s" {
		t.Errorf("response %v", comp)
	}

	// what the shim was sent: the chat form
	var run struct {
		Op       string           `json:"op"`
		Messages []map[string]any `json:"messages"`
		Tools    []map[string]any `json:"tools"`
		Effort   string           `json:"effort"`
		Images   []any            `json:"images"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("shim.runs"), "\n", 2)[0]), &run)
	if run.Op != "run" || run.Effort != "low" || len(run.Tools) != 2 || len(run.Messages) != 2 || len(run.Images) != 0 ||
		run.Messages[0]["role"] != "system" || run.Messages[0]["content"] != "You are Codex." ||
		run.Messages[1]["role"] != "user" || run.Messages[1]["content"] != "weather in Paris?" {
		t.Errorf("run %+v", run)
	}
	if run.Tools[0]["type"] != "function" || run.Tools[0]["function"].(map[string]any)["name"] != "get_weather" ||
		run.Tools[1]["function"].(map[string]any)["parameters"].(map[string]any)["required"].([]any)[0] != "input" {
		t.Errorf("tools %v", run.Tools)
	}

	waitFor(t, 2*time.Second, "service idle", func() bool { return e.srv.Service("s").Info().State == StateUp })
	in := e.srv.Service("s").Info()
	if in.Kind != KindShim || in.Last == nil || in.Last.TTFTMS == nil || *in.Last.TTFTMS != 123.5 || in.Last.InputTokens != 11 || in.Slots != 2 {
		t.Errorf("info %+v last %+v", in, in.Last)
	}
	// non-streaming
	resp = e.post("/v1/responses", strings.Replace(shimBody, `"stream":true`, `"stream":false`, 1))
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	resp.Body.Close()
	if r["status"] != "completed" || len(r["output"].([]any)) != 4 || r["mux"].(map[string]any)["ttft_ms"] != 123.5 {
		t.Errorf("non-streaming %v", r)
	}
}

func TestShimAutoRoute(t *testing.T) {
	e := startMux(t, epSpec{name: "s", scenario: "shimauto", kind: KindShim, ctx: 128000})
	e.waitState("s", StateUp, 10*time.Second)
	if !e.srv.Service("s").Info().Auto {
		t.Error("auto shim not reported")
	}
	sub := e.srv.Hub().Subscribe()
	defer e.srv.Hub().Unsubscribe(sub)
	resp := e.post("/v1/responses", shimBody)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	stats := res.named("response.ai_mux.stats")
	if len(stats) != 3 {
		t.Fatalf("%d stats events: %v", len(stats), res.events)
	}
	route := func(i int) (string, string, any) {
		o := stats[i].obj()
		r := o["route"].(map[string]any)
		b1, _ := json.Marshal(r["trying"])
		b2, _ := json.Marshal(r["failed"])
		return string(b1), string(b2), o["ttft_ms"]
	}
	if tr, fl, tt := route(0); stats[0].obj()["phase"] != "route" || tr != `["vertex/gemini-3-pro"]` || fl != `[]` || tt != nil {
		t.Errorf("first route %s", stats[0].Data)
	}
	if tr, fl, tt := route(1); stats[1].obj()["phase"] != "route" || tr != `["openrouter/qwen"]` || fl != `["vertex/gemini-3-pro"]` || tt != nil {
		t.Errorf("second route %s", stats[1].Data)
	}
	if tr, fl, tt := route(2); stats[2].obj()["phase"] != "output" || tr != `["openrouter/qwen"]` || fl != `["vertex/gemini-3-pro"]` || tt != 123.5 {
		t.Errorf("output stats %s", stats[2].Data)
	}
	if len(res.named("response.completed")) != 1 {
		t.Error("not completed")
	}
	n := map[string]int{}
	for done := false; !done; {
		select {
		case ev := <-sub.C:
			n[ev.Type]++
		default:
			done = true
		}
	}
	if n["run.route"] != 2 || n["run.ttft"] != 1 || n["run.start"] != 1 || n["run.end"] != 1 {
		t.Errorf("events %v", n)
	}
	waitFor(t, 2*time.Second, "idle", func() bool { return e.srv.Service("s").Info().State == StateUp })
	if l := e.srv.Service("s").Info().Last; l == nil || len(l.Route) != 1 || l.Route[0] != "openrouter/qwen" {
		t.Errorf("last %+v", l)
	}
}

func TestShimRule5(t *testing.T) {
	for _, c := range []struct{ scenario, why string }{
		{"shim_nottft", "must send ttft before its first delta"},
		{"shim_noroute", "an auto shim must send route"},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			e := startMux(t, epSpec{name: "s", scenario: c.scenario, kind: KindShim, ctx: 128000})
			e.waitState("s", StateUp, 10*time.Second)
			pid := e.srv.Service("s").Info().PID
			sub := e.srv.Hub().Subscribe()
			defer e.srv.Hub().Unsubscribe(sub)
			resp := e.post("/v1/responses", shimBody)
			res := readSSE(t, resp.Body)
			resp.Body.Close()
			if c.scenario == "shim_nottft" && (len(res.named("response.failed")) != 1 || len(res.named("response.output_text.delta")) != 0) {
				t.Errorf("the delta before ttft reached the client or the response did not fail: %d events", len(res.events))
			}
			in := e.waitState("s", StateFailed, 5*time.Second)
			if !strings.Contains(in.Reason, "rule 5") || !strings.Contains(in.Reason, c.why) {
				t.Errorf("reason %q", in.Reason)
			}
			waitFor(t, 5*time.Second, "shim process to be killed", func() bool { return !pidAlive(pid) })
			found := false
			for done := false; !done; {
				select {
				case ev := <-sub.C:
					if ev.Type == "service.state" && strings.Contains(string(ev.Data), `"failed"`) && strings.Contains(string(ev.Data), c.why) {
						found = true
					}
				default:
					done = true
				}
			}
			if !found {
				t.Error("no service.state failed event with the reason")
			}
		})
	}
}

func TestShimCancelAndError(t *testing.T) {
	e := startShim(t, "shim_hang")
	ctx, cancel := context.WithCancel(context.Background())
	resp := e.postCtx(ctx, "/v1/responses", strings.Replace(shimBody, `"tools":[`, `"x":[`, 1))
	buf := make([]byte, 4096)
	var acc string
	for !strings.Contains(acc, "output_text.delta") {
		n, err := resp.Body.Read(buf)
		acc += string(buf[:n])
		if err != nil {
			t.Fatal(err)
		}
	}
	cancel()
	resp.Body.Close()
	waitFor(t, 5*time.Second, "the shim to get cancel", func() bool { return strings.Contains(e.readOut("cancel"), "r1") })
	waitFor(t, 5*time.Second, "the shim to stop", func() bool { return strings.Contains(e.readOut("cancelled"), "yes") })
	waitFor(t, 5*time.Second, "idle", func() bool { in := e.srv.Service("s").Info(); return in.State == StateUp && in.Active == 0 })

	e2 := startShim(t, "shim_error")
	resp = e2.post("/v1/responses", shimBody)
	res := readSSE(t, resp.Body)
	resp.Body.Close()
	f := res.named("response.failed")
	if len(f) != 1 || !strings.Contains(string(f[0].Data), "provider said 429") || !strings.Contains(string(f[0].Data), "provider_error") {
		t.Errorf("events %v", res.events)
	}
	if e2.srv.Service("s").Info().State != StateUp {
		t.Error("a provider error dropped the shim")
	}
}

func TestShimImages(t *testing.T) {
	e := startShim(t, "shim")
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNGfake"))
	resp := e.post("/v1/responses", `{"model":"s","stream":false,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"see"},{"type":"input_image","image_url":"data:image/png;base64,`+png+`"}]}]}`)
	readBody(t, resp)
	var run struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
		Images []map[string]any `json:"images"`
	}
	json.Unmarshal([]byte(strings.SplitN(e.readOut("shim.runs"), "\n", 2)[0]), &run)
	h := hashBytes([]byte("\x89PNGfake"))
	if len(run.Images) != 1 || run.Images[0]["img"] != h || run.Images[0]["mime"] != "image/png" || run.Images[0]["bytes"] != png {
		t.Errorf("images %v", run.Images)
	}
	if len(run.Messages) != 1 || len(run.Messages[0].Content) != 2 || run.Messages[0].Content[1]["type"] != "image" || run.Messages[0].Content[1]["img"] != h {
		t.Errorf("messages %+v", run.Messages)
	}
}

// ---- embed ---------------------------------------------------------------------------------------------------

func startEmbed(t *testing.T, scenario string) *tenv {
	e := startMux(t, epSpec{name: "emb", scenario: scenario, kind: KindEmbed})
	e.waitState("emb", StateUp, 10*time.Second)
	return e
}

func TestEmbeddings(t *testing.T) {
	e := startEmbed(t, "embed")
	in := e.srv.Service("emb").Info()
	if in.Kind != KindEmbed || in.Dims != 4 || in.Device != "test device" && in.Device != "fake embed gpu" {
		t.Errorf("info %+v", in)
	}
	texts := []string{"alpha", "bb", "c", "dddd", "eeeee", "ffffff", "g"}
	b, _ := json.Marshal(map[string]any{"model": "emb", "input": texts})
	resp := e.post("/v1/embeddings", string(b))
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, readBody(t, resp))
	}
	var out struct {
		Object string `json:"object"`
		Model  string `json:"model"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage map[string]int `json:"usage"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if out.Object != "list" || out.Model != "emb" || len(out.Data) != 7 || out.Usage["prompt_tokens"] < 7 || out.Usage["total_tokens"] != out.Usage["prompt_tokens"] {
		t.Fatalf("response %+v", out)
	}
	for i, d := range out.Data {
		want := []float64{float64(len(texts[i])), 0, float64(texts[i][0]), float64(texts[i][len(texts[i])-1])}
		sum := 0
		for _, c := range []byte(texts[i]) {
			sum += int(c)
		}
		want[1] = float64(sum % 100)
		if d.Object != "embedding" || d.Index != i || len(d.Embedding) != 4 || d.Embedding[0] != want[0] || d.Embedding[1] != want[1] || d.Embedding[2] != want[2] || d.Embedding[3] != want[3] {
			t.Errorf("data %d: %+v want %v", i, d, want)
		}
	}
	// max_batch is 3: seven texts went as 3, 3, 1
	var sizes []int
	for _, line := range strings.Split(strings.TrimSpace(e.readOut("embed.runs")), "\n") {
		var r struct {
			Texts []string `json:"texts"`
		}
		json.Unmarshal([]byte(line), &r)
		sizes = append(sizes, len(r.Texts))
	}
	if len(sizes) != 3 || sizes[0] != 3 || sizes[1] != 3 || sizes[2] != 1 {
		t.Errorf("batches %v", sizes)
	}

	// a single string, and base64
	resp = e.post("/v1/embeddings", `{"model":"emb","input":"hello","encoding_format":"base64"}`)
	var b64 struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&b64)
	resp.Body.Close()
	raw, err := base64.StdEncoding.DecodeString(b64.Data[0].Embedding)
	if err != nil || len(raw) != 16 || math.Float32frombits(binary.LittleEndian.Uint32(raw)) != 5 {
		t.Errorf("base64 embedding %q: %v", b64.Data[0].Embedding, err)
	}

	for name, body := range map[string]string{
		"dimensions":   `{"model":"emb","input":"x","dimensions":256}`,
		"token arrays": `{"model":"emb","input":[[1,2,3]]}`,
		"empty list":   `{"model":"emb","input":[]}`,
		"empty text":   `{"model":"emb","input":[""]}`,
		"no input":     `{"model":"emb"}`,
		"bad format":   `{"model":"emb","input":"x","encoding_format":"hex"}`,
		"not json":     `{`,
	} {
		resp := e.post("/v1/embeddings", body)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status %d", name, resp.StatusCode)
		}
		resp.Body.Close()
	}
	resp = e.post("/v1/embeddings", `{"model":"nope","input":"x"}`)
	if resp.StatusCode != 404 {
		t.Errorf("unknown model: %d", resp.StatusCode)
	}
	resp.Body.Close()
	resp = e.post("/v1/responses", `{"model":"emb","input":"x"}`)
	if body := readBody(t, resp); resp.StatusCode != 400 || !strings.Contains(body, "/v1/embeddings") {
		t.Errorf("responses on an embed model: %d %s", resp.StatusCode, body)
	}
	// /v1/models
	var models struct{ Data []map[string]any }
	e.getJSON("/v1/models", &models)
	if len(models.Data) != 1 || models.Data[0]["id"] != "emb" || models.Data[0]["kind"] != "embed" || models.Data[0]["dims"] != float64(4) || models.Data[0]["state"] != "up" {
		t.Errorf("models %v", models.Data)
	}
	if l := e.srv.Service("emb").Info().Last; l == nil || l.Texts != 1 {
		t.Errorf("last %+v", l)
	}
}

func TestEmbedRule5AndErrors(t *testing.T) {
	for _, c := range []struct{ scenario, why string }{
		{"embed_bad", "vectors for"},
		{"embed_dims", "registered dims 4"},
	} {
		e := startEmbed(t, c.scenario)
		resp := e.post("/v1/embeddings", `{"model":"emb","input":["a","b"]}`)
		body := readBody(t, resp)
		if resp.StatusCode != 502 {
			t.Errorf("%s: status %d %s", c.scenario, resp.StatusCode, body)
		}
		in := e.waitState("emb", StateFailed, 5*time.Second)
		if !strings.Contains(in.Reason, "rule 5") || !strings.Contains(in.Reason, c.why) {
			t.Errorf("%s: reason %q", c.scenario, in.Reason)
		}
	}
	e := startEmbed(t, "embed_err")
	resp := e.post("/v1/embeddings", `{"model":"emb","input":["a"]}`)
	if body := readBody(t, resp); resp.StatusCode != 502 || !strings.Contains(body, "model overloaded") {
		t.Errorf("status %d %s", resp.StatusCode, body)
	}
	if e.srv.Service("emb").Info().State != StateUp {
		t.Error("an error dropped the embed service")
	}
}

// ---- jobs ----------------------------------------------------------------------------------------------------

func TestJobRunsToDone(t *testing.T) {
	e := startMux(t, epSpec{name: "j", scenario: "job", kind: KindJob})
	sub := e.srv.Hub().Subscribe()
	defer e.srv.Hub().Unsubscribe(sub)
	in := e.waitState("j", StateUp, 10*time.Second)
	if in.Kind != KindJob || in.Progress == nil || in.Progress.Title != "backfill embeddings" {
		t.Errorf("up: %+v", in)
	}
	// a running job reports its progress in /v1/services
	waitFor(t, 3*time.Second, "progress 50", func() bool {
		p := e.srv.Service("j").Info().Progress
		return p != nil && p.Pct >= 50 && strings.HasPrefix(p.Detail, "batch")
	})
	done := e.waitState("j", StateDone, 10*time.Second)
	if !strings.Contains(done.Reason, "1200 vectors") || done.Progress == nil || done.Progress.Pct != 100 || done.PID != 0 {
		t.Errorf("done: %+v progress %+v", done, done.Progress)
	}
	var pcts []float64
	var finished bool
	for stop := false; !stop; {
		select {
		case ev := <-sub.C:
			var m map[string]any
			json.Unmarshal(ev.Data, &m)
			switch ev.Type {
			case "job.progress":
				pcts = append(pcts, m["pct"].(float64))
			case "job.finished":
				finished = m["ok"] == true
			}
		default:
			stop = true
		}
	}
	if len(pcts) != 4 || pcts[0] != 0 || pcts[3] != 75 || !finished {
		t.Errorf("job events: pcts %v finished %v", pcts, finished)
	}
	// it stays done: no restart
	time.Sleep(500 * time.Millisecond)
	if st := e.srv.Service("j").Info().State; st != StateDone {
		t.Fatalf("state %s", st)
	}
	// up runs it again
	resp := e.post("/v1/services/j/up", "")
	resp.Body.Close()
	e.waitState("j", StateUp, 10*time.Second)
	e.waitState("j", StateDone, 10*time.Second)
	// jobs are not models
	var models struct{ Data []map[string]any }
	e.getJSON("/v1/models", &models)
	if len(models.Data) != 0 {
		t.Errorf("a job is listed as a model: %v", models.Data)
	}
	resp = e.post("/v1/responses", `{"model":"j","input":"hi"}`)
	if resp.StatusCode != 404 {
		t.Errorf("responses on a job: %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestJobRule5(t *testing.T) {
	for _, c := range []struct{ scenario, why string }{
		{"job_stall", "no progress for"},
		{"job_silent", "no progress for"},
		{"job_badpct", "pct between 0 and 100"},
		{"job_fail", "job failed: disk full"},
		{"job_exit", "exited without finishing"},
		{"job_garbage", "malformed line from the service"},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			e := startMux(t, epSpec{name: "j", scenario: c.scenario, kind: KindJob})
			begin := time.Now()
			in := e.waitState("j", StateFailed, 10*time.Second)
			if !strings.Contains(in.Reason, c.why) {
				t.Errorf("reason %q", in.Reason)
			}
			if strings.HasPrefix(c.scenario, "job_s") && !strings.Contains(in.Reason, "rule 5") {
				t.Errorf("a stalled job is a rule 5 failure: %q", in.Reason)
			}
			if c.scenario == "job_stall" || c.scenario == "job_silent" {
				// progress every 400 ms is the rule in tests; it is failed soon after 500 ms, not minutes later
				if d := time.Since(begin); d > 5*time.Second {
					t.Errorf("failed after %s", d)
				}
			}
			waitFor(t, 5*time.Second, "the process to be gone", func() bool { return e.srv.Service("j").Info().PID == 0 })
		})
	}
}

func TestJobLingeringProcessIsStopped(t *testing.T) {
	e := startMuxOpts(t, Options{JobFinishGrace: 400 * time.Millisecond}, epSpec{name: "j", scenario: "job_linger", kind: KindJob})
	in := e.waitState("j", StateUp, 10*time.Second)
	pid := in.PID
	e.waitState("j", StateDone, 10*time.Second)
	waitFor(t, 5*time.Second, "the lingering job process to be stopped", func() bool { return !pidAlive(pid) })
}
