package mux

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigSentry(t *testing.T) {
	c, err := LoadConfig("../hosts/sentry.yaml")
	if err != nil {
		t.Skipf("hosts/sentry.yaml: %v", err)
	}
	if c.Listen != "0.0.0.0:18100" || c.Socket != "/run/user/1000/ai-mux.sock" || len(c.Endpoints) != 1 {
		t.Fatalf("config: %+v", c)
	}
	e := c.Endpoints[0]
	if e.Name != "qwen3.8-27b" || !e.Up || e.Ctx != 262144 || e.Restart != "on-failure" || e.Vision ||
		e.Cwd != "/data/models/q27" || e.Env["JULIA_DEPOT_PATH"] != "/data/julia-depot" ||
		len(e.Efforts) != 5 || e.Command[0] != "/usr/bin/julia" || e.Command[1] != "-t" || e.Command[2] != "24" ||
		len(e.PostStop) != 3 || e.PostStop[0] != "sh" || !strings.HasSuffix(e.GGUF, "Qwen3.8-27B.Q5_K_M.gguf") {
		t.Fatalf("endpoint: %+v", e)
	}
}

const sampleYAML = `# sentry
listen: 127.0.0.1:18100   # clients
socket: /tmp/x.sock
endpoints:
  a:
    up: true    # desired
    gguf: /g.gguf
    ctx: 1000
    command: [sleep, "1"]
    env: {N: 24, S: x}
  b:
    gguf: /g.gguf
    ctx: 1000
    command: [sleep, "1"]
    restart: no
`

func TestConfigParseAndSetUp(t *testing.T) {
	c, err := ParseConfig([]byte(sampleYAML))
	if err != nil {
		t.Fatal(err)
	}
	if c.Endpoints[0].Env["N"] != "24" || c.Endpoints[1].Up || c.Endpoints[1].Restart != "no" || len(c.Endpoints[0].Efforts) != 5 {
		t.Fatalf("%+v %+v", c.Endpoints[0], c.Endpoints[1])
	}
	p := filepath.Join(t.TempDir(), "ai-mux.yaml")
	os.WriteFile(p, []byte(sampleYAML), 0o644)
	if err := SetUpInFile(p, "a", false); err != nil {
		t.Fatal(err)
	}
	if err := SetUpInFile(p, "b", true); err != nil { // no `up:` yet: it is added
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	got := string(b)
	for _, want := range []string{"# sentry", "# clients", "# desired", "up: false", "up: true"} {
		if !strings.Contains(got, want) {
			t.Errorf("rewritten file lost %q:\n%s", want, got)
		}
	}
	c2, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Endpoints[0].Up || !c2.Endpoints[1].Up {
		t.Fatalf("up not rewritten:\n%s", got)
	}
	if err := SetUpInFile(p, "nope", true); err == nil {
		t.Error("an unknown endpoint was accepted")
	}
	for _, bad := range []string{
		"socket: /x\nendpoints: {a: {gguf: g, ctx: 1, command: [x], restart: sometimes}}",
		"socket: /x\nendpoints: {a: {ctx: 1, command: [x]}}",
		"socket: /x\nendpoints: {a: {gguf: g, command: [x]}}",
		"socket: /x\nendpoints: {a: {gguf: g, ctx: 1}}",
	} {
		if _, err := ParseConfig([]byte(bad)); err == nil {
			t.Errorf("accepted: %s", bad)
		}
	}
}
