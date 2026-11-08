package mux

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

type epSpec struct {
	name     string
	scenario string
	ctx      int
	restart  string
	slots    int
	maxNew   int
	kind     string // engine (default), shim, embed, job
	postStop []string
	efforts  []string
	extraEnv map[string]string
	gguf     string
	display  string // service.yaml display:
	cfgGGUF  string // the gguf: in service.yaml, when the engine process must not be given it
	compat   bool   // this service is the compat service, fronted by a compat listener
}

type tenv struct {
	t      *testing.T
	srv    *Server
	base   string
	compat string // the compat listener's base URL, if any
	dir    string
	out    string
}

// shortDir is a short temp dir (a Unix socket path is limited to about 100 bytes).
func shortDir(t *testing.T) string {
	t.Helper()
	d, err := os.MkdirTemp("", "mux")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(d) })
	return d
}

func startMux(t *testing.T, specs ...epSpec) *tenv {
	return startMuxOpts(t, Options{}, specs...)
}

func startMuxOpts(t *testing.T, opts Options, specs ...epSpec) *tenv {
	t.Helper()
	dir := shortDir(t)
	out := filepath.Join(dir, "out")
	os.MkdirAll(out, 0o755)
	cfg := &Config{Listen: "127.0.0.1:0", Socket: filepath.Join(dir, "s.sock")}
	for _, sp := range specs {
		env := map[string]string{"MUX_FAKE_ENGINE": sp.scenario, "MUX_FAKE_OUT": out}
		if sp.slots > 0 {
			env["MUX_FAKE_SLOTS"] = strconv.Itoa(sp.slots)
		}
		if sp.maxNew > 0 {
			env["MUX_FAKE_MAXNEW"] = strconv.Itoa(sp.maxNew)
		}
		if sp.gguf != "" {
			env["MUX_FAKE_GGUF"] = sp.gguf
		}
		for k, v := range sp.extraEnv {
			env[k] = v
		}
		ctxN := sp.ctx
		if ctxN == 0 {
			ctxN = 100000
		}
		restart := sp.restart
		if restart == "" {
			restart = "no"
		}
		efforts := sp.efforts
		if efforts == nil {
			efforts = []string{"none", "low", "medium", "high", "xhigh"}
		}
		gguf := sp.gguf
		if gguf == "" {
			gguf = "fake.gguf"
		}
		kind := sp.kind
		if kind == "" {
			kind = KindEngine
		}
		env["MUX_FAKE_KIND"] = kind
		if kind == KindEmbed || kind == KindJob {
			ctxN, efforts, gguf = 0, nil, ""
		}
		if kind == KindShim {
			gguf = ""
		}
		if sp.cfgGGUF != "" {
			gguf = sp.cfgGGUF
		}
		cfg.Services = append(cfg.Services, &ServiceConfig{
			Name: sp.name, Display: sp.display, Dir: dir, Up: true, Kind: kind, GGUF: gguf, Ctx: ctxN, Efforts: efforts, Vision: kind == KindEngine || kind == KindShim,
			Device: "test device", Command: []string{os.Args[0]}, Cwd: dir, Env: env, PostStop: sp.postStop, Restart: restart,
		})
		if sp.compat {
			cfg.CompatListen, cfg.CompatService = "127.0.0.1:0", sp.name
		}
	}
	if opts.Config == nil {
		opts.Config = cfg
	}
	if opts.LogDir == "" {
		opts.LogDir = filepath.Join(dir, "logs")
	}
	if opts.Supervisor == "" {
		opts.Supervisor = "plain"
	}
	if opts.LoadModel == nil {
		opts.LoadModel = func(string) (Model, error) { return fakeModel{}, nil }
	}
	if opts.JobProgressEvery == 0 {
		opts.JobProgressEvery = 400 * time.Millisecond
	}
	if opts.JobFinishGrace == 0 {
		opts.JobFinishGrace = 3 * time.Second
	}
	if opts.MinBackoff == 0 {
		opts.MinBackoff = 200 * time.Millisecond
	}
	opts.KillGrace = 3 * time.Second
	opts.Logf = t.Logf
	srv, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Shutdown)
	te := &tenv{t: t, srv: srv, base: "http://" + srv.Addr(), dir: dir, out: out}
	if a := srv.CompatAddr(); a != "" {
		te.compat = "http://" + a
	}
	return te
}

func (e *tenv) waitState(name, state string, d time.Duration) ServiceInfo {
	e.t.Helper()
	deadline := time.Now().Add(d)
	var in ServiceInfo
	for time.Now().Before(deadline) {
		in = e.srv.Service(name).Info()
		if in.State == state {
			return in
		}
		time.Sleep(20 * time.Millisecond)
	}
	e.t.Fatalf("endpoint %s: want state %s within %s, have %s (reason %q)", name, state, d, in.State, in.Reason)
	return in
}

func (e *tenv) readOut(name string) string {
	b, _ := os.ReadFile(filepath.Join(e.out, name))
	return string(b)
}

type sseEvent struct {
	ID   string
	Name string
	Data json.RawMessage
	At   time.Time
}

func (s sseEvent) obj() map[string]any {
	var m map[string]any
	json.Unmarshal(s.Data, &m)
	return m
}

type sseResult struct {
	events     []sseEvent
	keepalives int
}

func (r sseResult) named(name string) []sseEvent {
	var out []sseEvent
	for _, e := range r.events {
		if e.Name == name {
			out = append(out, e)
		}
	}
	return out
}

// readSSE reads an event stream to its end.
func readSSE(t *testing.T, body io.Reader) sseResult {
	t.Helper()
	var res sseResult
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var cur sseEvent
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if cur.Name != "" || cur.Data != nil {
				res.events = append(res.events, cur)
			}
			cur = sseEvent{}
		case strings.HasPrefix(line, ":"):
			res.keepalives++
		case strings.HasPrefix(line, "id: "):
			cur.ID = line[len("id: "):]
		case strings.HasPrefix(line, "event: "):
			cur.Name = line[len("event: "):]
			cur.At = time.Now()
		case strings.HasPrefix(line, "data: "):
			cur.Data = json.RawMessage(line[len("data: "):])
			if cur.At.IsZero() {
				cur.At = time.Now()
			}
		}
	}
	return res
}

func (e *tenv) post(path string, body any) *http.Response {
	e.t.Helper()
	return e.postCtx(context.Background(), path, body)
}

func (e *tenv) postCtx(ctx context.Context, path string, body any) *http.Response {
	e.t.Helper()
	var b []byte
	switch v := body.(type) {
	case string:
		b = []byte(v)
	default:
		b, _ = json.Marshal(v)
	}
	req, _ := http.NewRequestWithContext(ctx, "POST", e.base+path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	return resp
}

func (e *tenv) getJSON(path string, v any) {
	e.t.Helper()
	resp, err := http.Get(e.base + path)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		e.t.Fatal(err)
	}
}

func readBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return string(b)
}

// pidAlive: the process exists (and is not a zombie we have not reaped).
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if err := syscall.Kill(pid, 0); err != nil {
		return false
	}
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	return i > 0 && !strings.HasPrefix(strings.TrimSpace(s[i+1:]), "Z")
}

func waitFor(t *testing.T, d time.Duration, what string, f func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if f() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", d, what)
}

func syscallKill(pid int) { syscall.Kill(pid, syscall.SIGKILL) }

// rogueRegister connects to the engine socket from this process and registers; it returns what mux answered.
func rogueRegister(t *testing.T, sock, endpoint string) string {
	t.Helper()
	c, err := net.Dial("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.Write([]byte(`{"op":"register","endpoint":"` + endpoint + `","positions":1000000,"max_new":8192,"slots":1,"vision":false}` + "\n"))
	c.SetReadDeadline(time.Now().Add(5 * time.Second))
	line, _ := bufio.NewReader(c).ReadString('\n')
	return line
}
