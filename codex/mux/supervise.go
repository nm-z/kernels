package mux

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// proc is a running service process. With systemd it lives in the transient scope
// mux-<name>.scope (BindsTo=mux.service); without, it leads its own process group.
type proc struct {
	cmd    *exec.Cmd
	pid    int
	scope  string // "mux-<name>.scope", or "" without systemd
	log    *os.File
	exited chan struct{}
	err    error // cmd.Wait's result, valid after exited is closed

	mu     sync.Mutex
	logCl  bool
	regAt  time.Time
	exitAt time.Time
}

var unitUnsafe = regexp.MustCompile(`[^A-Za-z0-9:_.-]`)

func scopeName(service string) string {
	return "mux-" + unitUnsafe.ReplaceAllString(service, "_") + ".scope"
}

// useSystemd decides how services are started: "systemd" always, "plain" never, "auto" (the
// default) when mux itself runs as mux.service.
func useSystemd(mode, unit string) bool {
	switch mode {
	case "systemd":
		return true
	case "plain":
		return false
	}
	if _, err := exec.LookPath("systemd-run"); err != nil {
		return false
	}
	b, err := os.ReadFile("/proc/self/cgroup")
	return err == nil && strings.Contains(string(b), unit)
}

func (s *Server) logPath(name string) string {
	return filepath.Join(s.logDir, unitUnsafe.ReplaceAllString(name, "_")+".log")
}

func (p *proc) mark(format string, args ...any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.log != nil && !p.logCl {
		fmt.Fprintf(p.log, "mux: %s %s\n", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
	}
}

func (p *proc) closeLog() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.log != nil && !p.logCl {
		p.log.Close()
		p.logCl = true
	}
}

func (p *proc) setRegistered() {
	p.mu.Lock()
	if p.regAt.IsZero() {
		p.regAt = time.Now()
	}
	p.mu.Unlock()
}

// uptime is how long the process was registered before it exited (0 if it never registered).
func (p *proc) uptime() time.Duration {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.regAt.IsZero() || p.exitAt.IsZero() {
		return 0
	}
	return p.exitAt.Sub(p.regAt)
}

func (p *proc) exitDescription() string { return describeExit(p.err) }

// startProc launches the service's command.
func (s *Server) startProc(e *Service) (*proc, error) {
	if err := os.MkdirAll(s.logDir, 0o755); err != nil {
		return nil, err
	}
	lf, err := os.OpenFile(e.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	cfg := e.cfg
	env := append(os.Environ(), "MUX_SOCK="+s.cfg.Socket, "MUX_SERVICE="+e.name, "MUX_SERVICE_DIR="+cfg.Dir)
	setenv := []string{"--setenv=MUX_SOCK=" + s.cfg.Socket, "--setenv=MUX_SERVICE=" + e.name, "--setenv=MUX_SERVICE_DIR=" + cfg.Dir}
	keys := make([]string, 0, len(cfg.Env))
	for k := range cfg.Env {
		keys = append(keys, k)
	}
	sortStrings(keys)
	for _, k := range keys {
		env = append(env, k+"="+cfg.Env[k])
		setenv = append(setenv, "--setenv="+k+"="+cfg.Env[k])
	}
	p := &proc{log: lf, exited: make(chan struct{})}
	var cmd *exec.Cmd
	if s.systemd {
		p.scope = scopeName(e.name)
		// a scope left by a crashed mux must not block the unit name
		exec.Command("systemctl", "--user", "stop", p.scope).Run()
		args := []string{"--user", "--scope", "--unit=" + strings.TrimSuffix(p.scope, ".scope"), "--collect",
			"-p", "BindsTo=" + s.unit, "-p", "After=" + s.unit, "-p", "TimeoutStopSec=10"}
		args = append(args, setenv...)
		if cfg.Cwd != "" {
			args = append(args, "--working-directory="+cfg.Cwd)
		}
		args = append(args, "--")
		args = append(args, cfg.Command...)
		cmd = exec.Command("systemd-run", args...)
	} else {
		cmd = exec.Command(cfg.Command[0], cfg.Command[1:]...)
	}
	cmd.Dir = cfg.Cwd
	cmd.Env = env
	cmd.Stdin = nil
	cmd.Stdout, cmd.Stderr = lf, lf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	p.cmd = cmd
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(lf, "mux: %s start failed: %v\n", time.Now().Format(time.RFC3339), err)
		lf.Close()
		return nil, err
	}
	p.pid = cmd.Process.Pid
	how := "plain process group"
	if p.scope != "" {
		how = "scope " + p.scope
	}
	p.mark("start pid=%d (%s) cwd=%s cmd=%q", p.pid, how, cfg.Cwd, cfg.Command)
	go func() {
		p.err = cmd.Wait()
		p.mu.Lock()
		p.exitAt = time.Now()
		p.mu.Unlock()
		p.mark("exit pid=%d: %s", p.pid, describeExit(p.err))
		close(p.exited)
	}()
	return p, nil
}

// stop kills the process and everything it started: the scope with systemd, else the process group
// (SIGTERM, then SIGKILL after a grace period). It returns when the process is gone.
func (p *proc) stop(s *Server) {
	p.mark("stop requested")
	if p.scope != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		exec.CommandContext(ctx, "systemctl", "--user", "stop", p.scope).Run()
		cancel()
		select {
		case <-p.exited:
			return
		case <-time.After(5 * time.Second):
		}
	} else {
		syscall.Kill(-p.pid, syscall.SIGTERM)
		select {
		case <-p.exited:
			return
		case <-time.After(s.killGrace):
		}
	}
	syscall.Kill(-p.pid, syscall.SIGKILL)
	syscall.Kill(p.pid, syscall.SIGKILL)
	<-p.exited
}

// reap kills what is left in the process group of an exited process.
func (p *proc) reap() {
	<-p.exited
	syscall.Kill(-p.pid, syscall.SIGKILL)
}

// runPostStop runs the service's post_stop command after its process is gone.
func (s *Server) runPostStop(e *Service, p *proc) {
	if len(e.cfg.PostStop) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, e.cfg.PostStop[0], e.cfg.PostStop[1:]...)
	cmd.Dir = e.cfg.Cwd
	cmd.Stdout, cmd.Stderr = p.log, p.log
	p.mark("post_stop %q", e.cfg.PostStop)
	err := cmd.Run()
	p.mark("post_stop: %s", describeExit(err))
	if err != nil {
		s.logf("service %s: post_stop: %v", e.name, err)
	}
}

// ---- who may register ---------------------------------------------------------------------------

// peerPID is the process on the other end of an engine socket.
func peerCred(c syscallConn) (pid, uid int, err error) {
	raw, err := c.SyscallConn()
	if err != nil {
		return 0, 0, err
	}
	var cred *syscall.Ucred
	var serr error
	if err := raw.Control(func(fd uintptr) {
		cred, serr = syscall.GetsockoptUcred(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	}); err != nil {
		return 0, 0, err
	}
	if serr != nil {
		return 0, 0, serr
	}
	return int(cred.Pid), int(cred.Uid), nil
}

type syscallConn interface {
	SyscallConn() (syscall.RawConn, error)
}

// inService reports whether pid is one of the processes mux started for p: in its scope, or (no
// systemd) its descendant.
func (p *proc) contains(pid int) bool {
	if p.scope != "" {
		b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cgroup")
		return err == nil && strings.Contains(string(b), "/"+p.scope)
	}
	for i := 0; i < 64 && pid > 1; i++ {
		if pid == p.pid {
			return true
		}
		pid = parentPID(pid)
	}
	return false
}

func parentPID(pid int) int {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return 0
	}
	// pid (comm) state ppid …; comm may contain spaces and parentheses
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return 0
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 2 {
		return 0
	}
	n, _ := strconv.Atoi(f[1])
	return n
}
