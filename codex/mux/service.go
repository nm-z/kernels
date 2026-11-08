package mux

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// Service states (PROTOCOL.md).
const (
	StateDown     = "down"
	StateStarting = "starting"
	StateUp       = "up"
	StateBusy     = "busy"
	StateFailed   = "failed"
	StateDone     = "done" // a job that finished
)

const (
	minBackoff    = 10 * time.Second
	maxBackoff    = 5 * time.Minute
	healthyUptime = 60 * time.Second // a service that stayed up this long starts its backoff over
)

// LastRun is what the last finished run measured, as far as its kind reports it.
type LastRun struct {
	PrefillTPS   *float64  `json:"prefill_tps,omitempty"` // engine
	DecodeTPS    *float64  `json:"decode_tps,omitempty"`
	IdlePct      *float64  `json:"idle_pct,omitempty"`
	TTFTMS       *float64  `json:"ttft_ms,omitempty"` // shim
	Route        []string  `json:"route,omitempty"`
	InputTokens  int       `json:"input_tokens,omitempty"`
	OutputTokens int       `json:"output_tokens,omitempty"`
	Texts        int       `json:"texts,omitempty"` // embed
	At           time.Time `json:"at"`
}

// Progress is a job's last report.
type Progress struct {
	Title  string    `json:"title,omitempty"`
	Pct    float64   `json:"pct"`
	Detail string    `json:"detail,omitempty"`
	At     time.Time `json:"at"`
}

type jobResult struct {
	ok      bool
	message string
}

// Service is one thing mux serves: its configuration, its process and its registered connection.
type Service struct {
	s       *Server
	cfg     *ServiceConfig
	name    string
	kind    string
	logPath string

	mu           sync.Mutex
	state        string
	since        time.Time
	reason       string
	desired      bool
	stopping     bool // a stop was requested; no new requests
	p            *proc
	conn         *svcConn
	model        Model // an engine's tokenizer and chat template
	last         *LastRun
	progress     *Progress
	lastProgress time.Time // a job's last progress (or its registration)
	jobRes       *jobResult
	device       string // an embed service reports its own
	active       int
	queued       int

	wake       chan struct{} // desired state changed
	failReq    chan string   // a service broke a rule: kill it and mark failed
	finishedCh chan struct{} // a job sent `finished`
	done       chan struct{} // the supervision loop ended
}

func newService(s *Server, cfg *ServiceConfig) *Service {
	return &Service{
		s: s, cfg: cfg, name: cfg.Name, kind: cfg.Kind,
		logPath: s.logPath(cfg.Name),
		state:   StateDown, since: time.Now(), desired: cfg.Up,
		wake:       make(chan struct{}, 1),
		failReq:    make(chan string, 1),
		finishedCh: make(chan struct{}, 1),
		done:       make(chan struct{}),
	}
}

// ServiceInfo is a service as GET /v1/services shows it.
type ServiceInfo struct {
	Name      string    `json:"name"`
	Kind      string    `json:"kind"`
	State     string    `json:"state"`
	Reason    string    `json:"reason,omitempty"`
	Device    string    `json:"device"`
	PID       int       `json:"pid"`
	Ctx       int       `json:"ctx"`
	Slots     int       `json:"slots"`
	Since     string    `json:"since"`
	Last      *LastRun  `json:"last"`
	Progress  *Progress `json:"progress,omitempty"`
	Log       string    `json:"log"`
	Up        bool      `json:"up"`
	Active    int       `json:"active"`
	Queued    int       `json:"queued"`
	MaxNew    int       `json:"max_new,omitempty"`
	Positions int       `json:"positions,omitempty"`
	Dims      int       `json:"dims,omitempty"`
	Auto      bool      `json:"auto,omitempty"`
	Vision    bool      `json:"vision"`
	Efforts   []string  `json:"efforts,omitempty"`
}

func (e *Service) Info() ServiceInfo {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.infoLocked()
}

func (e *Service) infoLocked() ServiceInfo {
	in := ServiceInfo{
		Name: e.name, Kind: e.kind, State: e.state, Reason: e.reason, Device: e.cfg.Device, Ctx: e.cfg.Ctx,
		Since: e.since.Format(time.RFC3339), Last: e.last, Progress: e.progress, Log: e.logPath, Up: e.desired,
		Active: e.active, Queued: e.queued, Vision: e.cfg.Vision, Efforts: e.cfg.Efforts,
	}
	if e.device != "" && in.Device == "" {
		in.Device = e.device
	}
	if e.p != nil {
		in.PID = e.p.pid
	}
	if c := e.conn; c != nil {
		in.Slots, in.MaxNew, in.Positions, in.Dims, in.Auto = c.slots, c.maxNew, c.positions, c.dims, c.auto
	}
	return in
}

func (e *Service) setStateLocked(st, reason string) {
	if e.state == st && e.reason == reason {
		return
	}
	if e.state != st {
		e.since = time.Now()
	}
	e.state, e.reason = st, reason
	f := map[string]any{"service": e.name, "kind": e.kind, "state": st, "pid": 0, "since": e.since.Format(time.RFC3339)}
	if e.p != nil {
		f["pid"] = e.p.pid
	}
	if reason != "" {
		f["reason"] = reason
	}
	e.s.hub.Publish("service.state", f)
}

func (e *Service) setState(st, reason string) {
	e.mu.Lock()
	e.setStateLocked(st, reason)
	e.mu.Unlock()
}

// SetDesired changes the desired state (not the file); the supervision loop does the rest.
func (e *Service) SetDesired(up bool) {
	e.mu.Lock()
	e.desired = up
	e.mu.Unlock()
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

func (e *Service) wantUp() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.desired
}

// fail asks the supervision loop to kill the service's process and mark it failed.
func (e *Service) fail(reason string) {
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

// attach makes a registered connection the service's (it becomes up). The caller has sent `registered`.
func (e *Service) attach(c *svcConn) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.p == nil || e.stopping || !e.desired {
		return fmt.Errorf("service %s is not starting", e.name)
	}
	if e.conn != nil {
		return fmt.Errorf("service %s is already registered", e.name)
	}
	e.conn = c
	e.reason = ""
	if c.device != "" {
		e.device = c.device
	}
	if e.kind == KindJob {
		e.jobRes = nil
		e.lastProgress = time.Now()
		e.progress = &Progress{Title: c.title, At: e.lastProgress}
	}
	e.setStateLocked(e.upStateLocked(), "")
	return nil
}

func (e *Service) upStateLocked() string {
	if e.active > 0 {
		return StateBusy
	}
	return StateUp
}

// detach drops the registration of c (its socket closed or it broke a rule).
func (e *Service) detach(c *svcConn) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn != c {
		return
	}
	e.conn = nil
	if e.p != nil && (e.state == StateUp || e.state == StateBusy) && e.jobRes == nil {
		e.setStateLocked(StateStarting, "")
	}
}

// currentConn is the registered connection, or nil with the state to report.
func (e *Service) currentConn() (*svcConn, string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn == nil || e.stopping {
		return nil, e.state
	}
	return e.conn, e.state
}

func (e *Service) runStarted() {
	e.mu.Lock()
	e.active++
	if e.conn != nil && e.state == StateUp {
		e.setStateLocked(StateBusy, "")
	}
	e.mu.Unlock()
}

func (e *Service) runEnded(last *LastRun) {
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

// ---- jobs ------------------------------------------------------------------------------------------------

// jobProgress takes a job's progress line.
func (e *Service) jobProgress(c *svcConn, m *msg) {
	if m.Pct == nil || *m.Pct < 0 || *m.Pct > 100 {
		e.fail("rule 5: job progress without a pct between 0 and 100")
		return
	}
	now := time.Now()
	e.mu.Lock()
	if e.conn != c {
		e.mu.Unlock()
		return
	}
	e.lastProgress = now
	e.progress = &Progress{Title: c.title, Pct: *m.Pct, Detail: m.Detail, At: now}
	e.mu.Unlock()
	e.s.hub.Publish("job.progress", map[string]any{"service": e.name, "title": c.title, "pct": *m.Pct, "detail": m.Detail})
}

// jobFinished takes a job's last line; the process should now exit.
func (e *Service) jobFinished(c *svcConn, m *msg) {
	ok := m.OK != nil && *m.OK
	e.mu.Lock()
	if e.conn != c || e.jobRes != nil {
		e.mu.Unlock()
		return
	}
	e.jobRes = &jobResult{ok: ok, message: m.Message}
	if ok && e.progress != nil {
		e.progress.Pct, e.progress.At = 100, time.Now()
	}
	e.mu.Unlock()
	e.s.hub.Publish("job.finished", map[string]any{"service": e.name, "title": c.title, "ok": ok, "message": m.Message})
	select {
	case e.finishedCh <- struct{}{}:
	default:
	}
}

// jobStalled reports a running job that has sent no progress for too long (rule 5).
func (e *Service) jobStalled() (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.kind != KindJob || e.conn == nil || e.jobRes != nil {
		return "", false
	}
	limit := e.s.jobStall()
	if idle := time.Since(e.lastProgress); idle > limit {
		return fmt.Sprintf("rule 5: job sent no progress for %s (it must send progress at least every %s)", idle.Round(100*time.Millisecond), e.s.jobEvery), true
	}
	return "", false
}

// ---- supervision -----------------------------------------------------------------------------------------

// loop is the service's supervision: start it when it should be up, restart it per policy, stop it
// when it should be down. It owns the process.
func (e *Service) loop() {
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
			e.progress = nil
			e.setStateLocked(StateDown, "")
			e.mu.Unlock()
			select {
			case <-e.wake:
			case <-e.s.quit:
				return
			}
			continue
		}
		res := e.runOnce()
		if e.s.quitting() {
			return
		}
		if res.stopped { // down or shutdown
			continue
		}
		if res.done {
			e.mu.Lock()
			e.stopping = false
			e.setStateLocked(StateDone, res.message)
			e.mu.Unlock()
			e.s.logf("service %s: finished", e.name)
			select { // an `up` that came while it ran is spent
			case <-e.wake:
			default:
			}
			select {
			case <-e.wake: // up again runs it again; down puts it down
			case <-e.s.quit:
				return
			}
			continue
		}
		e.mu.Lock()
		e.stopping = false
		e.setStateLocked(StateFailed, res.reason)
		e.mu.Unlock()
		e.s.logf("service %s failed: %s", e.name, res.reason)
		if res.uptime >= healthyUptime {
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

type runResult struct {
	stopped bool // stopped on request (down, shutdown)
	done    bool // a job that finished ok
	message string
	reason  string // why it failed
	uptime  time.Duration
}

// runOnce starts the process and supervises it until it is gone.
func (e *Service) runOnce() (res runResult) {
	if e.kind == KindEngine {
		model, err := e.s.models.get(e.cfg.GGUF)
		if err != nil {
			return runResult{reason: fmt.Sprintf("tokenizer and chat template: %v", err)}
		}
		e.mu.Lock()
		e.model = model
		e.mu.Unlock()
	}
	p, err := e.s.startProc(e)
	if err != nil {
		return runResult{reason: fmt.Sprintf("start: %v", err)}
	}
	e.mu.Lock()
	e.p = p
	e.stopping = false
	e.jobRes = nil
	e.progress = nil
	select {
	case <-e.finishedCh:
	default:
	}
	e.setStateLocked(StateStarting, "")
	e.mu.Unlock()
	e.s.logf("service %s: started pid %d", e.name, p.pid)

	stopWhy := "" // "" the process is running, else why we stopped it
	var failWhy string
	exited := p.exited
	var grace <-chan time.Time
	var watchdog <-chan time.Time
	if e.kind == KindJob {
		wd := time.NewTicker(e.s.jobStall() / 4)
		defer wd.Stop()
		watchdog = wd.C
	}
	stop := func(why string) {
		stopWhy = why
		e.markStopping()
		go p.stop(e.s)
	}
	for exited != nil {
		select {
		case <-exited:
			exited = nil
		case <-e.wake:
			if !e.wantUp() && stopWhy == "" {
				stop("down")
			}
		case r := <-e.failReq:
			if stopWhy == "" {
				failWhy = r
				stop("fail")
			}
		case <-e.finishedCh:
			grace = time.After(e.s.finishGrace) // the job said it is finished: its process should exit
		case <-grace:
			grace = nil
			if stopWhy == "" {
				stop("finished")
			}
		case <-watchdog:
			if why, bad := e.jobStalled(); bad && stopWhy == "" {
				failWhy = why
				stop("fail")
			}
		case <-e.s.quit:
			if stopWhy == "" {
				stop("quit")
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
	jobRes := e.jobRes
	e.mu.Unlock()
	e.s.runPostStop(e, p)
	p.closeLog()
	e.mu.Lock()
	e.conn = nil
	e.mu.Unlock()
	res.uptime = p.uptime()
	switch {
	case stopWhy == "down" || stopWhy == "quit":
		res.stopped = true
	case stopWhy == "fail":
		res.reason = failWhy
	case e.kind == KindJob && jobRes != nil && jobRes.ok:
		res.done, res.message = true, jobRes.message
	case e.kind == KindJob && jobRes != nil:
		res.reason = "job failed: " + jobRes.message
	case !e.wantUp():
		res.stopped = true
	case e.kind == KindJob:
		res.reason = "exited without finishing: " + p.exitDescription()
	default:
		res.reason = "exited: " + p.exitDescription()
	}
	return res
}

func (e *Service) markStopping() {
	e.mu.Lock()
	e.stopping = true
	e.mu.Unlock()
}

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
