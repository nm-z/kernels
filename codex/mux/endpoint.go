package mux

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// Endpoint states (PROTOCOL.md).
const (
	StateDown     = "down"
	StateStarting = "starting"
	StateUp       = "up"
	StateBusy     = "busy"
	StateFailed   = "failed"
)

const (
	minBackoff    = 10 * time.Second
	maxBackoff    = 5 * time.Minute
	healthyUptime = 60 * time.Second // an engine that stayed up this long starts its backoff over
)

// LastRun is the measured numbers of the last finished run.
type LastRun struct {
	PrefillTPS float64   `json:"prefill_tps"`
	DecodeTPS  float64   `json:"decode_tps"`
	IdlePct    float64   `json:"idle_pct"`
	At         time.Time `json:"at"`
}

// Endpoint is one served model: its configuration, its process and its registered engine.
type Endpoint struct {
	s       *Server
	cfg     *EndpointConfig
	name    string
	logPath string

	mu       sync.Mutex
	state    string
	since    time.Time
	reason   string
	desired  bool
	stopping bool // a stop was requested; no new requests
	p        *proc
	conn     *engineConn
	model    Model
	last     *LastRun
	active   int
	queued   int

	wake    chan struct{} // desired state changed
	failReq chan string   // an engine broke a rule: kill it and mark failed
	done    chan struct{} // the supervision loop ended
}

func newEndpoint(s *Server, cfg *EndpointConfig) *Endpoint {
	return &Endpoint{
		s: s, cfg: cfg, name: cfg.Name,
		logPath: s.logPath(cfg.Name),
		state:   StateDown, since: time.Now(), desired: cfg.Up,
		wake:    make(chan struct{}, 1),
		failReq: make(chan string, 1),
		done:    make(chan struct{}),
	}
}

// EndpointInfo is an endpoint as GET /v1/endpoints shows it.
type EndpointInfo struct {
	Name      string   `json:"name"`
	State     string   `json:"state"`
	Reason    string   `json:"reason,omitempty"`
	Device    string   `json:"device"`
	PID       int      `json:"pid"`
	Ctx       int      `json:"ctx"`
	Slots     int      `json:"slots"`
	Since     string   `json:"since"`
	Last      *LastRun `json:"last"`
	Log       string   `json:"log"`
	Up        bool     `json:"up"`
	Active    int      `json:"active"`
	Queued    int      `json:"queued"`
	MaxNew    int      `json:"max_new,omitempty"`
	Positions int      `json:"positions,omitempty"`
	Vision    bool     `json:"vision"`
	Efforts   []string `json:"efforts"`
}

func (e *Endpoint) Info() EndpointInfo {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.infoLocked()
}

func (e *Endpoint) infoLocked() EndpointInfo {
	in := EndpointInfo{
		Name: e.name, State: e.state, Reason: e.reason, Device: e.cfg.Device, Ctx: e.cfg.Ctx,
		Since: e.since.Format(time.RFC3339), Last: e.last, Log: e.logPath, Up: e.desired,
		Active: e.active, Queued: e.queued, Vision: e.cfg.Vision, Efforts: e.cfg.Efforts,
	}
	if e.p != nil {
		in.PID = e.p.pid
	}
	if e.conn != nil {
		in.Slots, in.MaxNew, in.Positions = e.conn.slots, e.conn.maxNew, e.conn.positions
	}
	return in
}

func (e *Endpoint) setStateLocked(st, reason string) {
	if e.state == st && e.reason == reason {
		return
	}
	if e.state != st {
		e.since = time.Now()
	}
	e.state, e.reason = st, reason
	f := map[string]any{"endpoint": e.name, "state": st, "pid": 0, "since": e.since.Format(time.RFC3339)}
	if e.p != nil {
		f["pid"] = e.p.pid
	}
	if reason != "" {
		f["reason"] = reason
	}
	e.s.hub.Publish("endpoint.state", f)
}

func (e *Endpoint) setState(st, reason string) {
	e.mu.Lock()
	e.setStateLocked(st, reason)
	e.mu.Unlock()
}

// SetDesired changes the desired state (not the file); the supervision loop does the rest.
func (e *Endpoint) SetDesired(up bool) {
	e.mu.Lock()
	e.desired = up
	e.mu.Unlock()
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

func (e *Endpoint) wantUp() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.desired
}

// fail asks the supervision loop to kill the endpoint's process and mark it failed.
func (e *Endpoint) fail(reason string) {
	e.mu.Lock()
	if e.conn != nil {
		e.conn.close()
	}
	e.mu.Unlock()
	select {
	case e.failReq <- reason:
	default:
	}
}

// attach makes a registered engine the endpoint's (it becomes up). The caller has sent `registered`.
func (e *Endpoint) attach(c *engineConn) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.p == nil || e.stopping || !e.desired {
		return fmt.Errorf("endpoint %s is not starting", e.name)
	}
	if e.conn != nil {
		return fmt.Errorf("endpoint %s is already registered", e.name)
	}
	e.conn = c
	e.reason = ""
	e.setStateLocked(e.upStateLocked(), "")
	return nil
}

func (e *Endpoint) upStateLocked() string {
	if e.active > 0 {
		return StateBusy
	}
	return StateUp
}

// detach drops the registration of c (its socket closed or it broke a rule).
func (e *Endpoint) detach(c *engineConn) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn != c {
		return
	}
	e.conn = nil
	if e.p != nil && (e.state == StateUp || e.state == StateBusy) {
		e.setStateLocked(StateStarting, "")
	}
}

// currentConn is the registered engine, or nil with the state to report.
func (e *Endpoint) currentConn() (*engineConn, string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn == nil || e.stopping {
		return nil, e.state
	}
	return e.conn, e.state
}

func (e *Endpoint) runStarted() {
	e.mu.Lock()
	e.active++
	if e.conn != nil && e.state == StateUp {
		e.setStateLocked(StateBusy, "")
	}
	e.mu.Unlock()
}

func (e *Endpoint) runEnded(last *LastRun) {
	e.mu.Lock()
	e.active--
	if last != nil {
		e.last = last
	}
	if e.active == 0 && e.conn != nil && e.state == StateBusy {
		e.setStateLocked(StateUp, "")
	}
	e.mu.Unlock()
}

// loop is the endpoint's supervision: start it when it should be up, restart it per policy, stop it
// when it should be down. It owns the process.
func (e *Endpoint) loop() {
	// Pdeathsig fires when the thread that started the child exits: keep this goroutine on one thread.
	runtime.LockOSThread()
	defer close(e.done)
	backoff := e.s.minBack
	for {
		select {
		case <-e.s.quit:
			return
		default:
		}
		if !e.wantUp() {
			e.mu.Lock()
			e.stopping = false
			e.setStateLocked(StateDown, "")
			e.mu.Unlock()
			select {
			case <-e.wake:
			case <-e.s.quit:
				return
			}
			continue
		}
		reason, uptime := e.runOnce()
		if e.s.quitting() {
			return
		}
		if reason == "" { // stopped on request
			continue
		}
		e.mu.Lock()
		e.stopping = false
		e.setStateLocked(StateFailed, reason)
		e.mu.Unlock()
		e.s.logf("endpoint %s failed: %s", e.name, reason)
		if uptime >= healthyUptime {
			backoff = e.s.minBack
		}
		if e.cfg.Restart == "no" {
			select {
			case <-e.wake:
				backoff = e.s.minBack
			case <-e.s.quit:
				return
			}
			continue
		}
		t := time.NewTimer(backoff)
		select {
		case <-t.C:
			backoff = minDur(backoff*2, maxBackoff)
		case <-e.wake:
			t.Stop()
			backoff = e.s.minBack // an explicit up retries now
		case <-e.s.quit:
			t.Stop()
			return
		}
	}
}

func minDur(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// runOnce starts the process and supervises it until it is gone. It returns "" when the process was
// stopped on request (down, shutdown), otherwise why it failed, and how long it stayed registered.
func (e *Endpoint) runOnce() (reason string, uptime time.Duration) {
	model, err := e.s.models.get(e.cfg.GGUF)
	if err != nil {
		return fmt.Sprintf("tokenizer and chat template: %v", err), 0
	}
	e.mu.Lock()
	e.model = model
	e.mu.Unlock()
	p, err := e.s.startProc(e)
	if err != nil {
		return fmt.Sprintf("start: %v", err), 0
	}
	e.mu.Lock()
	e.p = p
	e.stopping = false
	e.setStateLocked(StateStarting, "")
	e.mu.Unlock()
	e.s.logf("endpoint %s: started pid %d", e.name, p.pid)

	stopWhy := ""      // "" the process is running, else why we stopped it
	var failWhy string // set when we stopped it because it broke a rule
	exited := p.exited
	for exited != nil {
		select {
		case <-exited:
			exited = nil
		case <-e.wake:
			if !e.wantUp() && stopWhy == "" {
				stopWhy = "down"
				e.markStopping()
				go p.stop(e.s)
			}
		case r := <-e.failReq:
			if stopWhy == "" {
				stopWhy, failWhy = "fail", r
				e.markStopping()
				go p.stop(e.s)
			}
		case <-e.s.quit:
			if stopWhy == "" {
				stopWhy = "quit"
				e.markStopping()
				go p.stop(e.s)
			}
		}
	}
	// the process is gone: nothing it started may stay
	p.reap()
	e.mu.Lock()
	if e.conn != nil {
		e.conn.close()
	}
	e.p = nil
	e.mu.Unlock()
	e.s.runPostStop(e, p)
	p.closeLog()
	e.mu.Lock()
	if e.conn != nil {
		e.conn = nil
	}
	e.mu.Unlock()
	uptime = p.uptime()
	switch stopWhy {
	case "down", "quit":
		return "", uptime
	case "fail":
		return failWhy, uptime
	}
	if e.wantUp() {
		return "exited: " + p.exitDescription(), uptime
	}
	return "", uptime
}

func (e *Endpoint) markStopping() {
	e.mu.Lock()
	e.stopping = true
	e.mu.Unlock()
}

// helpers for processes -------------------------------------------------------------------------

func describeExit(err error) string {
	if err == nil {
		return "exit code 0"
	}
	if ee, ok := err.(interface{ Sys() any }); ok {
		if ws, ok := ee.Sys().(syscall.WaitStatus); ok {
			if ws.Signaled() {
				return "killed by signal " + ws.Signal().String()
			}
			return fmt.Sprintf("exit code %d", ws.ExitStatus())
		}
	}
	return err.Error()
}
