package mux

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"sync"
	"time"
)

// fakeSvc is the connection side of a fake shim, embed or job process.
type fakeSvc struct {
	conn net.Conn
	br   *bufio.Reader
	wmu  sync.Mutex
	out  string
}

func newFakeSvc() *fakeSvc {
	conn, err := net.Dial("unix", os.Getenv("MUX_SOCK"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake service:", err)
		os.Exit(2)
	}
	return &fakeSvc{conn: conn, br: bufio.NewReaderSize(conn, 1<<20), out: os.Getenv("MUX_FAKE_OUT")}
}

func (f *fakeSvc) send(v map[string]any) {
	b, _ := json.Marshal(v)
	f.wmu.Lock()
	f.conn.Write(append(b, '\n'))
	f.wmu.Unlock()
}

func (f *fakeSvc) note(name, s string) {
	if f.out == "" {
		return
	}
	if fh, err := os.OpenFile(f.out+"/"+name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
		fh.WriteString(s)
		fh.Close()
	}
}

func (f *fakeSvc) register(m map[string]any) {
	m["op"] = "register"
	m["service"] = os.Getenv("MUX_SERVICE")
	f.send(m)
}

// read returns the next line of mux as a map, and the raw line.
func (f *fakeSvc) read() (map[string]any, []byte) {
	line, err := f.br.ReadBytes('\n')
	if err != nil {
		os.Exit(0)
	}
	var m map[string]any
	json.Unmarshal(line, &m)
	return m, line
}

// ---- shim -----------------------------------------------------------------------------------------------------

// fakeShim: a provider over the network. Scenarios: shim, shimauto, shim_nottft, shim_noroute, shim_hang,
// shim_error, shim_summary (answers a compaction prompt).
func fakeShim(sc string) {
	f := newFakeSvc()
	auto := sc == "shimauto" || sc == "shim_noroute"
	f.register(map[string]any{"kind": "shim", "slots": 2, "vision": true, "auto": auto})
	var mu sync.Mutex
	cancelled := map[string]chan struct{}{}
	for {
		m, line := f.read()
		switch m["op"] {
		case "error":
			f.note("refused", fmt.Sprint(m["message"]))
			os.Exit(3)
		case "cancel":
			mu.Lock()
			if c := cancelled[fmt.Sprint(m["id"])]; c != nil {
				select {
				case <-c:
				default:
					close(c)
				}
			}
			mu.Unlock()
			f.note("cancel", fmt.Sprint(m["id"])+"\n")
		case "run":
			id := fmt.Sprint(m["id"])
			c := make(chan struct{})
			mu.Lock()
			cancelled[id] = c
			mu.Unlock()
			f.note("shim.runs", string(line))
			go fakeShimRun(f, sc, id, c)
		}
	}
}

func fakeShimRun(f *fakeSvc, sc, id string, cancel chan struct{}) {
	d := func(ch, text string) { f.send(map[string]any{"op": "delta", "id": id, "channel": ch, "text": text}) }
	pause := func(ms int) bool {
		select {
		case <-cancel:
			return false
		case <-time.After(time.Duration(ms) * time.Millisecond):
			return true
		}
	}
	done := func(in, out int) {
		f.send(map[string]any{"op": "done", "id": id, "input_tokens": in, "output_tokens": out})
	}
	switch sc {
	case "shim_error":
		f.send(map[string]any{"op": "error", "id": id, "message": "provider said 429"})
		return
	case "shim_nottft":
		d("content", "too early")
		done(1, 1)
		return
	case "shim_hang":
		f.send(map[string]any{"op": "ttft", "id": id, "ms": 5})
		for {
			d("content", "x")
			if !pause(20) {
				f.note("cancelled", "yes\n")
				done(1, 1)
				return
			}
		}
	case "shim_summary":
		f.send(map[string]any{"op": "ttft", "id": id, "ms": 7})
		d("reasoning", "thinking about the summary")
		d("content", "  The user wants weather data for Paris; ")
		d("content", "the plan is to call get_weather.\n")
		done(40, 12)
		return
	}
	if sc == "shimauto" || sc == "shim_noroute" {
		if sc == "shimauto" {
			f.send(map[string]any{"op": "route", "id": id, "trying": []string{"vertex/gemini-3-pro"}, "failed": []string{}})
			pause(60)
			f.send(map[string]any{"op": "route", "id": id, "trying": []string{"openrouter/qwen"}, "failed": []string{"vertex/gemini-3-pro"}})
			pause(60)
		}
	}
	pause(30)
	f.send(map[string]any{"op": "ttft", "id": id, "ms": 123.5})
	d("reasoning", "Let me ")
	d("reasoning", "think. café")
	pause(10)
	d("content", "Hello")
	d("content", " world.\n")
	f.send(map[string]any{"op": "tool_call", "id": id, "call_id": "call_shim1", "name": "get_weather", "arguments": `{"city":"Paris","days":3}`})
	f.send(map[string]any{"op": "tool_call", "id": id, "call_id": "call_shim2", "name": "apply_patch", "arguments": map[string]any{"input": "*** Begin Patch\n*** End Patch"}})
	if sc == "shim" || sc == "shimauto" || sc == "shim_noroute" {
		done(11, 22)
	}
}

// ---- embed ---------------------------------------------------------------------------------------------------

// fakeEmbed: an embedding server. Scenarios: embed, embed_bad (one vector short), embed_dims (wrong width),
// embed_err.
func fakeEmbed(sc string) {
	f := newFakeSvc()
	f.register(map[string]any{"kind": "embed", "dims": 4, "max_batch": 3, "device": "fake embed gpu"})
	for {
		m, line := f.read()
		switch m["op"] {
		case "error":
			f.note("refused", fmt.Sprint(m["message"]))
			os.Exit(3)
		case "run":
			f.note("embed.runs", string(line))
			id := fmt.Sprint(m["id"])
			texts, _ := m["texts"].([]any)
			if sc == "embed_err" {
				f.send(map[string]any{"op": "error", "id": id, "message": "model overloaded"})
				continue
			}
			var vecs [][]float64
			for _, t := range texts {
				s := fmt.Sprint(t)
				sum := 0
				for _, c := range []byte(s) {
					sum += int(c)
				}
				v := []float64{float64(len(s)), float64(sum % 100), float64(s[0]), float64(s[len(s)-1])}
				if sc == "embed_dims" {
					v = v[:3]
				}
				vecs = append(vecs, v)
			}
			if sc == "embed_bad" && len(vecs) > 1 {
				vecs = vecs[:len(vecs)-1]
			}
			f.send(map[string]any{"op": "vectors", "id": id, "vectors": vecs})
		}
	}
}

// ---- job -------------------------------------------------------------------------------------------------------

// fakeJob: finite work. Scenarios: job (0, 25, 50, 75 %, then finished), job_stall (one progress, then
// silence), job_silent (never any progress), job_fail (finished with ok false), job_badpct, job_exit (exits
// without finished), job_linger (finished, but the process stays).
func fakeJob(sc string) {
	f := newFakeSvc()
	f.register(map[string]any{"kind": "job", "title": "backfill embeddings"})
	go func() { // read (and ignore) what mux sends
		for {
			m, _ := f.read()
			if m["op"] == "error" {
				f.note("refused", fmt.Sprint(m["message"]))
				os.Exit(3)
			}
		}
	}()
	progress := func(pct float64, detail string) {
		f.send(map[string]any{"op": "progress", "pct": pct, "detail": detail})
	}
	switch sc {
	case "job_silent":
		select {}
	case "job_stall":
		progress(10, "started")
		select {}
	case "job_badpct":
		progress(140, "too much")
		select {}
	case "job_exit":
		progress(5, "about to die")
		os.Exit(1)
	case "job_garbage":
		f.conn.Write([]byte("this is not json\n"))
		select {}
	}
	for i, pct := range []float64{0, 25, 50, 75} {
		progress(pct, fmt.Sprintf("batch %d of 4", i+1))
		time.Sleep(150 * time.Millisecond)
	}
	if sc == "job_fail" {
		f.send(map[string]any{"op": "finished", "ok": false, "message": "disk full"})
		os.Exit(0)
	}
	f.send(map[string]any{"op": "finished", "ok": true, "message": "4 batches, 1200 vectors"})
	if sc == "job_linger" {
		select {}
	}
	os.Exit(0)
}
