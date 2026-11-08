package mux

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

func cgroupOf(pid int) string {
	b, _ := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cgroup")
	return string(b)
}

// TestSystemdScopes runs an endpoint the way the systemd user unit does: in a transient scope bound to
// the unit ai-mux runs as. A stand-in unit plays ai-mux.service, so the real one is not touched.
func TestSystemdScopes(t *testing.T) {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		t.Skip("no systemd-run")
	}
	probe := "aimux-probe-" + strconv.Itoa(os.Getpid())
	if err := exec.Command("systemd-run", "--user", "--scope", "--collect", "--unit="+probe, "true").Run(); err != nil {
		t.Skipf("no systemd user session to start scopes in: %v", err)
	}
	unit := "aimux-test-" + strconv.Itoa(os.Getpid()) + ".service"
	if out, err := exec.Command("systemd-run", "--user", "--unit="+unit, "--collect", "sleep", "600").CombinedOutput(); err != nil {
		t.Skipf("cannot start a stand-in unit: %v: %s", err, out)
	}
	t.Cleanup(func() { exec.Command("systemctl", "--user", "stop", unit).Run() })

	name := "t" + strconv.Itoa(os.Getpid())
	scope := "ai-mux-" + name + ".scope"
	e := startMuxOpts(t, Options{Supervisor: "systemd", Unit: unit}, epSpec{name: name, scenario: "spawn"})
	in := e.waitState(name, StateUp, 15*time.Second)
	waitFor(t, 3*time.Second, "child pid", func() bool { return e.readOut("child.pid") != "" })
	child, _ := strconv.Atoi(strings.TrimSpace(e.readOut("child.pid")))

	// invariant 3: the engine registered because it is in the scope; so is everything it started
	if !strings.Contains(cgroupOf(in.PID), "/"+scope) || !strings.Contains(cgroupOf(child), "/"+scope) {
		t.Fatalf("engine cgroup %q, child cgroup %q", cgroupOf(in.PID), cgroupOf(child))
	}
	if !e.srv.Supervised() {
		t.Error("not supervised through systemd")
	}
	if got := rogueRegister(t, e.dir+"/s.sock", name); !strings.Contains(got, "register refused") {
		t.Errorf("a process outside the scope registered: %q", got)
	}
	if out, _ := exec.Command("systemctl", "--user", "show", "-p", "BindsTo", "-p", "After", scope).Output(); !strings.Contains(string(out), unit) {
		t.Errorf("scope is not bound to %s: %s", unit, out)
	}

	// down: the scope is stopped, which kills the engine and its child
	resp := e.post("/v1/services/"+name+"/down", "")
	resp.Body.Close()
	e.waitState(name, StateDown, 20*time.Second)
	if pidAlive(in.PID) || pidAlive(child) {
		t.Errorf("down left processes: engine %v child %v", pidAlive(in.PID), pidAlive(child))
	}
	if out, _ := exec.Command("systemctl", "--user", "is-active", scope).Output(); strings.TrimSpace(string(out)) == "active" {
		t.Error("scope still active after down")
	}
	resp = e.post("/v1/services/"+name+"/up", "")
	resp.Body.Close()
	in = e.waitState(name, StateUp, 15*time.Second)
	waitFor(t, 3*time.Second, "child pid", func() bool { return len(strings.Fields(e.readOut("child.pid"))) == 2 })
	child, _ = strconv.Atoi(strings.Fields(e.readOut("child.pid"))[1])

	// invariant 2: when the unit ai-mux runs as goes away, systemd stops every endpoint scope
	exec.Command("systemctl", "--user", "stop", unit).Run()
	waitFor(t, 20*time.Second, "systemd to stop the scope", func() bool { return !pidAlive(in.PID) && !pidAlive(child) })
	e.waitState(name, StateFailed, 10*time.Second)
}
