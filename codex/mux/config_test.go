package mux

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigSentry(t *testing.T) {
	b, err := os.ReadFile("../hosts/sentry.yaml")
	if err != nil {
		t.Skipf("hosts/sentry.yaml: %v", err)
	}
	h, err := ParseHost(b, "../hosts")
	if err != nil {
		t.Fatal(err)
	}
	if h.Listen != "0.0.0.0:18100" || h.Socket != "/run/user/1000/ai-mux.sock" || h.ServicesDir != "/data/ai-mux/services" ||
		len(h.Services) != 1 || h.Services[0].Name != "qwen3.8-27b" || !h.Services[0].Up {
		t.Fatalf("host: %+v", h)
	}
	// the host file points at a checkout on sentry; load the same file against this repo's services/
	here := strings.Replace(string(b), "/data/ai-mux/services", "../services", 1)
	p := filepath.Join(t.TempDir(), "ai-mux.yaml")
	os.WriteFile(p, []byte(strings.Replace(here, "../services", mustAbs(t, "../services"), 1)), 0o644)
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	e := c.Services[0]
	if e.Name != "qwen3.8-27b" || e.Kind != KindEngine || !e.Up || e.Ctx != 262144 || e.Restart != "on-failure" || e.Vision ||
		e.Cwd != "/data/models/q27" || e.Env["JULIA_DEPOT_PATH"] != "/data/julia-depot" ||
		len(e.Efforts) != 5 || e.Command[0] != "/usr/bin/julia" || e.Command[1] != "-t" || e.Command[2] != "24" ||
		e.Command[len(e.Command)-2] != "--console" || e.Command[len(e.Command)-1] != "/data/models/q27/console.sock" ||
		len(e.PostStop) != 3 || e.PostStop[0] != "sh" || !strings.HasSuffix(e.GGUF, "Qwen3.8-27B.Q5_K_M.gguf") ||
		!strings.HasSuffix(e.Dir, "services/qwen3.8-27b") {
		t.Fatalf("service: %+v", e)
	}
}

func mustAbs(t *testing.T, p string) string {
	t.Helper()
	a, err := filepath.Abs(p)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

const sampleHost = `# sentry
listen: 127.0.0.1:18100   # clients
socket: /tmp/x.sock
services_dir: SVCDIR
services:
  a: {up: true}    # desired
  b: {up: false}
  c:
`

func writeService(t *testing.T, dir, name, yaml string) {
	t.Helper()
	os.MkdirAll(filepath.Join(dir, name), 0o755)
	if err := os.WriteFile(filepath.Join(dir, name, "service.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestConfigParseAndSetUp(t *testing.T) {
	dir := t.TempDir()
	writeService(t, dir, "a", "kind: engine\ngguf: /g.gguf\nctx: 1000\ncommand: [sleep, \"1\"]\nenv: {N: 24, S: x}\n")
	writeService(t, dir, "b", "kind: shim\nctx: 5000\ncommand: [./run.sh]\ncwd: sub\nrestart: no\n")
	writeService(t, dir, "c", "kind: job\ncommand: [train]\n")
	p := filepath.Join(dir, "ai-mux.yaml")
	os.WriteFile(p, []byte(strings.Replace(sampleHost, "SVCDIR", "'.'", 1)), 0o644) // relative to the file
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	a, b, jc := c.Services[0], c.Services[1], c.Services[2]
	if a.Env["N"] != "24" || !a.Up || b.Up || jc.Up || len(a.Efforts) != 5 || b.Kind != KindShim || len(b.Efforts) != 5 {
		t.Fatalf("%+v %+v %+v", a, b, jc)
	}
	if a.Cwd != filepath.Join(dir, "a") || b.Cwd != filepath.Join(dir, "b", "sub") {
		t.Errorf("cwd defaults to the service folder: %q %q", a.Cwd, b.Cwd)
	}
	if a.Restart != "on-failure" || b.Restart != "no" || jc.Restart != "no" || jc.Kind != KindJob {
		t.Errorf("restart defaults: %q %q %q", a.Restart, b.Restart, jc.Restart)
	}
	if err := SetUpInFile(p, "a", false); err != nil {
		t.Fatal(err)
	}
	if err := SetUpInFile(p, "c", true); err != nil { // `c:` has no value yet: it gets one
		t.Fatal(err)
	}
	if err := SetUpInFile(p, "b", true); err != nil {
		t.Fatal(err)
	}
	out, _ := os.ReadFile(p)
	got := string(out)
	for _, want := range []string{"# sentry", "# clients", "# desired", "a: {up: false}", "b: {up: true}", "c: {up: true}"} {
		if !strings.Contains(got, want) {
			t.Errorf("rewritten file lost %q:\n%s", want, got)
		}
	}
	c2, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Services[0].Up || !c2.Services[1].Up || !c2.Services[2].Up {
		t.Fatalf("up not rewritten:\n%s", got)
	}
	if err := SetUpInFile(p, "nope", true); err == nil {
		t.Error("an unknown service was accepted")
	}
}

func TestConfigRejects(t *testing.T) {
	dir := t.TempDir()
	host := "socket: /x\nservices_dir: " + dir + "\nservices: {s: {up: true}}\n"
	for name, svc := range map[string]string{
		"no kind":        "command: [x]",
		"bad kind":       "kind: spaceship\ncommand: [x]",
		"engine no gguf": "kind: engine\nctx: 1\ncommand: [x]",
		"engine no ctx":  "kind: engine\ngguf: g\ncommand: [x]",
		"shim no ctx":    "kind: shim\ncommand: [x]",
		"no command":     "kind: embed",
		"bad restart":    "kind: job\ncommand: [x]\nrestart: sometimes",
	} {
		writeService(t, dir, "s", svc)
		if _, err := loadConfig([]byte(host), ""); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := loadConfig([]byte("socket: /x\nservices: {s: {up: true}}\n"), ""); err == nil {
		t.Error("a host file without services_dir accepted")
	}
	if _, err := loadConfig([]byte("socket: /x\nservices_dir: "+dir+"\nservices: {missing: {up: true}}\n"), ""); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Errorf("a service without a folder: %v", err)
	}
	if _, err := loadConfig([]byte("socket: /x\nservices_dir: "+dir+"\nservices: [a]\n"), ""); err == nil {
		t.Error("services as a list accepted")
	}
}
