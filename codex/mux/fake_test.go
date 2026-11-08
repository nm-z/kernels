package mux

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nm-z/kernels/codex/tmpl"
)

// The test binary doubles as the engine: an endpoint's command is the test binary itself with
// MUX_FAKE_ENGINE=<scenario>; TestMain turns that into fakeEngine().
func TestMain(m *testing.M) {
	if sc := os.Getenv("MUX_FAKE_ENGINE"); sc != "" {
		switch os.Getenv("MUX_FAKE_KIND") {
		case KindShim:
			fakeShim(sc)
		case KindEmbed:
			fakeEmbed(sc)
		case KindJob:
			fakeJob(sc)
		default:
			fakeEngine(sc)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// ---- fakeModel: a byte vocabulary with a few special tokens and a ChatML template ----------------

type special struct {
	text   string
	id     uint32
	hidden bool
}

var fakeSpecials = []special{
	{"<|im_start|>", 262, true}, {"<|im_end|>", 256, true}, {"</think>", 257, true}, {"<think>", 258, true},
	{"<tool_call>", 259, false}, {"</tool_call>", 260, false}, {"<|image_pad|>", 261, true},
	{"<|vision_start|>", 263, true}, {"<|vision_end|>", 264, true},
}

type fakeModel struct{}

func (fakeModel) Find(text string) int {
	for _, s := range fakeSpecials {
		if s.text == text {
			return int(s.id)
		}
	}
	return -1
}
func (fakeModel) EOS() uint32 { return 256 }
func (fakeModel) Hidden(id uint32) bool {
	for _, s := range fakeSpecials {
		if s.id == id {
			return s.hidden
		}
	}
	return false
}
func (fakeModel) Piece(id uint32) []byte {
	if id < 256 {
		return []byte{byte(id)}
	}
	for _, s := range fakeSpecials {
		if s.id == id {
			return []byte(s.text)
		}
	}
	return nil
}
func (fakeModel) Encode(text string) ([]uint32, error) {
	var out []uint32
	for len(text) > 0 {
		best, at := -1, -1
		for i, s := range fakeSpecials {
			if j := strings.Index(text, s.text); j >= 0 && (at < 0 || j < at) {
				best, at = i, j
			}
		}
		if at < 0 {
			at = len(text)
		}
		for i := 0; i < at; i++ {
			out = append(out, uint32(text[i]))
		}
		if best >= 0 {
			out = append(out, fakeSpecials[best].id)
			text = text[at+len(fakeSpecials[best].text):]
		} else {
			text = ""
		}
	}
	return out, nil
}

func (fakeModel) Render(vars map[string]any) (string, error) {
	b, _ := json.Marshal(vars)
	var v struct {
		Messages []struct {
			Role             string          `json:"role"`
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
		Tools           []json.RawMessage `json:"tools"`
		EnableThinking  *bool             `json:"enable_thinking"`
		ReasoningEffort string            `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return "", err
	}
	var sb strings.Builder
	content := func(raw json.RawMessage) string {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
		var parts []map[string]string
		json.Unmarshal(raw, &parts)
		var o strings.Builder
		for _, p := range parts {
			if p["type"] == "image" {
				o.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
			} else {
				o.WriteString(p["text"])
			}
		}
		return o.String()
	}
	if len(v.Messages) == 0 {
		return "", fmt.Errorf("No messages provided.")
	}
	for i, m := range v.Messages {
		if m.Role == "system" && i != 0 {
			return "", fmt.Errorf("System message must be at the beginning.")
		}
	}
	if len(v.Tools) > 0 || v.ReasoningEffort != "" || v.Messages[0].Role == "system" {
		sb.WriteString("<|im_start|>system\n")
		if v.ReasoningEffort != "" {
			sb.WriteString("[effort:" + v.ReasoningEffort + "]\n")
		}
		for _, t := range v.Tools {
			sb.WriteString("[tool:" + string(t) + "]\n")
		}
		if v.Messages[0].Role == "system" {
			sb.WriteString(content(v.Messages[0].Content))
		}
		sb.WriteString("<|im_end|>\n")
	}
	for _, m := range v.Messages {
		switch m.Role {
		case "system":
		case "assistant":
			sb.WriteString("<|im_start|>assistant\n<think>\n" + m.ReasoningContent + "\n</think>\n\n" + content(m.Content))
			for _, tc := range m.ToolCalls {
				sb.WriteString("<tool_call>[" + tc.ID + " " + tc.Function.Name + " " + string(tc.Function.Arguments) + "]</tool_call>")
			}
			sb.WriteString("<|im_end|>\n")
		case "tool":
			sb.WriteString("<|im_start|>tool\n[" + m.ToolCallID + "]" + content(m.Content) + "<|im_end|>\n")
		default:
			sb.WriteString("<|im_start|>" + m.Role + "\n" + content(m.Content) + "<|im_end|>\n")
		}
	}
	sb.WriteString("<|im_start|>assistant\n")
	if v.EnableThinking != nil && !*v.EnableThinking {
		sb.WriteString("<think>\n\n</think>\n\n")
	} else {
		sb.WriteString("<think>\n")
	}
	return sb.String(), nil
}

// ---- the fake engine process ---------------------------------------------------------------------

const fakeAnswer = "The user wants the weather in Paris: café 日本語. I should call the tool.\n</think>\n\n" +
	"Let me check.\n\n" +
	"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n" +
	"<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n</parameter>\n</function>\n</tool_call><|im_end|>"

type fakeRun struct {
	id        string
	ids       []uint32
	maxNew    int
	cancelled chan struct{}
}

func fakeEngine(scenario string) {
	out := os.Getenv("MUX_FAKE_OUT")
	note := func(name, s string) {
		if out == "" {
			return
		}
		f, err := os.OpenFile(out+"/"+name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err == nil {
			f.WriteString(s)
			f.Close()
		}
	}
	var model Model = fakeModel{}
	if g := os.Getenv("MUX_FAKE_GGUF"); g != "" {
		tk, err := tmpl.Load(g)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fake engine:", err)
			os.Exit(4)
		}
		model = tk
	}
	conn, err := net.Dial("unix", os.Getenv("MUX_SOCK"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake engine:", err)
		os.Exit(2)
	}
	var wmu sync.Mutex
	send := func(v map[string]any) {
		b, _ := json.Marshal(v)
		wmu.Lock()
		conn.Write(append(b, '\n'))
		wmu.Unlock()
	}
	positions := 1 << 20
	if scenario == "smallpos" {
		positions = 10
	}
	maxNew := 8192
	if s := os.Getenv("MUX_FAKE_MAXNEW"); s != "" {
		maxNew, _ = strconv.Atoi(s)
	}
	slots := 1
	if s := os.Getenv("MUX_FAKE_SLOTS"); s != "" {
		slots, _ = strconv.Atoi(s)
	}
	if scenario == "spawn" {
		child := exec.Command("sleep", "600")
		child.Start()
		note("child.pid", strconv.Itoa(child.Process.Pid)+"\n")
	}
	note("pid", strconv.Itoa(os.Getpid())+"\n")
	send(map[string]any{"op": "register", "service": os.Getenv("MUX_SERVICE"), "kind": "engine", "positions": positions, "max_new": maxNew, "slots": slots, "vision": true})

	br := bufio.NewReaderSize(conn, 1<<20)
	var mu sync.Mutex
	runs := map[string]*fakeRun{}
	active := 0
	for {
		line, err := br.ReadBytes('\n')
		if err != nil {
			os.Exit(0)
		}
		var m struct {
			Op      string   `json:"op"`
			ID      string   `json:"id"`
			IDs     []uint32 `json:"ids"`
			MaxNew  int      `json:"max_new"`
			Stop    []uint32 `json:"stop"`
			Message string   `json:"message"`
			Img     string   `json:"img"`
			Images  []struct {
				Img string `json:"img"`
				At  int    `json:"at"`
			} `json:"images"`
		}
		json.Unmarshal(line, &m)
		switch m.Op {
		case "error":
			note("refused", m.Message)
			os.Exit(3)
		case "image":
			send(map[string]any{"op": "image", "img": m.Img, "tokens": 4})
		case "cancel":
			mu.Lock()
			if r := runs[m.ID]; r != nil {
				select {
				case <-r.cancelled:
				default:
					close(r.cancelled)
				}
			}
			mu.Unlock()
			note("cancel", m.ID+"\n")
		case "run":
			r := &fakeRun{id: m.ID, ids: m.IDs, maxNew: m.MaxNew, cancelled: make(chan struct{})}
			mu.Lock()
			runs[m.ID] = r
			active++
			overlap := active > slots
			mu.Unlock()
			text := ""
			for _, id := range m.IDs {
				text += string(model.Piece(id))
			}
			note("prompt", text+"\n====\n")
			rj, _ := json.Marshal(map[string]any{"ids": len(m.IDs), "max_new": m.MaxNew, "stop": m.Stop, "images": m.Images})
			note("run.json", string(rj)+"\n")
			go func() {
				// the slot is free before the last line goes out: mux may send the next run at once
				var once sync.Once
				release := func() { once.Do(func() { mu.Lock(); active--; delete(runs, m.ID); mu.Unlock() }) }
				defer release()
				sendRun := func(v map[string]any) {
					if op := v["op"]; op == "done" || op == "error" {
						release()
					}
					send(v)
				}
				if overlap {
					sendRun(map[string]any{"op": "error", "id": m.ID, "message": "overlapping runs on a one-slot engine"})
					return
				}
				fakeRunOne(scenario, model, r, sendRun, note)
			}()
		}
	}
}

func fakeRunOne(scenario string, model Model, r *fakeRun, send func(map[string]any), note func(string, string)) {
	np := len(r.ids)
	if scenario == "error" {
		send(map[string]any{"op": "error", "id": r.id, "message": "kernel crashed"})
		return
	}
	if scenario == "silent" {
		// the engine says nothing for a while: mux still sends a stats event every 500 ms
		send(map[string]any{"op": "prefill", "id": r.id, "done": 0, "total": np})
		time.Sleep(1700 * time.Millisecond)
		send(map[string]any{"op": "prefill", "id": r.id, "done": np, "total": np})
		send(map[string]any{"op": "stats", "id": r.id, "prefill_tps": 10.0, "decode_tps": 0.0})
		send(map[string]any{"op": "tokens", "id": r.id, "ids": mustEncode("hi</think>\n\nok<|im_end|>")})
		send(map[string]any{"op": "done", "id": r.id, "prompt": np, "generated": 10, "prefill_tps": 10.0, "decode_tps": 5.0, "idle_pct": 90.0, "stopped": true})
		return
	}
	if scenario == "timings" {
		// a slow prefill that says how it is going: four steps, a stats line with each
		for k := 0; k < 4; k++ {
			send(map[string]any{"op": "prefill", "id": r.id, "done": k * np / 4, "total": np})
			send(map[string]any{"op": "stats", "id": r.id, "prefill_tps": 111.5, "decode_tps": 0.0, "idle_pct": nil})
			time.Sleep(200 * time.Millisecond)
		}
		send(map[string]any{"op": "prefill", "id": r.id, "done": np, "total": np})
	}
	if scenario != "noprefill" && scenario != "timings" {
		send(map[string]any{"op": "prefill", "id": r.id, "done": 0, "total": np})
		time.Sleep(30 * time.Millisecond)
		send(map[string]any{"op": "prefill", "id": r.id, "done": np, "total": np})
	}
	var ids []uint32
	switch scenario {
	case "hang":
		// generates until cancelled
		for i := 0; ; i++ {
			select {
			case <-r.cancelled:
				note("cancelled", "yes\n")
				send(map[string]any{"op": "done", "id": r.id, "prompt": np, "generated": i, "prefill_tps": 100.0, "decode_tps": 10.0, "idle_pct": 1.0, "stopped": false})
				return
			case <-time.After(20 * time.Millisecond):
			}
			send(map[string]any{"op": "tokens", "id": r.id, "ids": []uint32{'x'}})
			if i%10 == 0 {
				send(map[string]any{"op": "stats", "id": r.id, "prefill_tps": 100.0, "decode_tps": 10.0})
			}
		}
	case "summary":
		ids, _ = model.Encode("Summary: weather in Paris, 3 days.<|im_end|>")
	case "timings":
		ids, _ = model.Encode("Hmm.</think>\n\nIt is 42.<|im_end|>")
	case "maxnew":
		for i := 0; i < r.maxNew; i++ {
			ids = append(ids, 'a')
		}
	case "slow":
		ids, _ = model.Encode("quick answer</think>\n\nok<|im_end|>")
	default:
		ids, _ = model.Encode(fakeAnswer)
	}
	stopped := scenario != "maxnew"
	// stream in chunks of 3 tokens, 15 ms apart: a stats line every 50 ms
	chunk := 3
	delay := 15 * time.Millisecond
	if scenario == "slow" || scenario == "timings" {
		delay = 150 * time.Millisecond
	}
	if scenario == "maxnew" || scenario == "nostats" || scenario == "noprefill" || scenario == "noidle" {
		delay = time.Millisecond
	}
	start := time.Now()
	for i := 0; i < len(ids); i += chunk {
		select {
		case <-r.cancelled:
			send(map[string]any{"op": "done", "id": r.id, "prompt": np, "generated": i, "prefill_tps": 100.0, "decode_tps": 10.0, "idle_pct": 1.0, "stopped": false})
			return
		default:
		}
		j := i + chunk
		if j > len(ids) {
			j = len(ids)
		}
		send(map[string]any{"op": "tokens", "id": r.id, "ids": ids[i:j]})
		if scenario != "nostats" && ((i/chunk)%3 == 0 || scenario == "timings") {
			var idle any // null at first: the engine has no idle % yet
			if i > 0 {
				idle = 3.5
			}
			send(map[string]any{"op": "stats", "id": r.id, "prefill_tps": 1234.5, "decode_tps": float64(i+1)/time.Since(start).Seconds() + 1, "idle_pct": idle})
		}
		time.Sleep(delay)
	}
	done := map[string]any{"op": "done", "id": r.id, "prompt": np, "generated": len(ids), "prefill_tps": 1234.5, "decode_tps": 67.8, "idle_pct": 4.2, "stopped": stopped}
	if scenario == "noidle" {
		delete(done, "idle_pct")
	}
	send(done)
}
