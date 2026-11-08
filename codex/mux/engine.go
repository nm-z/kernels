package mux

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"sync"
	"time"
)

// runID is a run's id. ai-mux makes them (strings); an engine may echo them as numbers.
type runID string

func (r *runID) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*r = runID(s)
		return nil
	}
	*r = runID(string(b))
	return nil
}

// engMsg is any line an engine sends. Fields whose absence is a protocol violation are pointers.
type engMsg struct {
	Op string `json:"op"`

	// register
	Endpoint  string `json:"endpoint"`
	Positions int    `json:"positions"`
	MaxNew    int    `json:"max_new"`
	Slots     int    `json:"slots"`
	Vision    bool   `json:"vision"`

	// run lines
	ID         runID    `json:"id"`
	Done       *int     `json:"done"`  // prefill
	Total      *int     `json:"total"` // prefill
	IDs        []uint32 `json:"ids"`   // tokens
	PrefillTPS *float64 `json:"prefill_tps"`
	DecodeTPS  *float64 `json:"decode_tps"`
	IdlePct    *float64 `json:"idle_pct"`
	Prompt     *int     `json:"prompt"`
	Generated  *int     `json:"generated"`
	Stopped    bool     `json:"stopped"`
	Message    string   `json:"message"`

	// image
	Img    string `json:"img"`
	Tokens int    `json:"tokens"`

	lost bool // synthetic: the engine's socket closed under the run
}

// run is one request's run on an engine: the lines the engine sends for it, queued for the handler.
type run struct {
	id  runID
	mu  sync.Mutex
	q   []*engMsg
	sig chan struct{}
}

func newRun(id runID) *run { return &run{id: id, sig: make(chan struct{}, 1)} }

func (r *run) push(m *engMsg) {
	r.mu.Lock()
	r.q = append(r.q, m)
	r.mu.Unlock()
	select {
	case r.sig <- struct{}{}:
	default:
	}
}

func (r *run) take() []*engMsg {
	r.mu.Lock()
	q := r.q
	r.q = nil
	r.mu.Unlock()
	return q
}

// engineConn is a registered engine's socket.
type engineConn struct {
	ep        *Endpoint
	nc        net.Conn
	positions int
	maxNew    int
	slots     int
	vision    bool

	sem      chan struct{} // one token per request running at once
	dead     chan struct{}
	deadOnce sync.Once

	wmu sync.Mutex

	mu      sync.Mutex
	runs    map[runID]*run
	imgWait map[string]chan int
	nextRun uint64
}

func newEngineConn(ep *Endpoint, nc net.Conn, reg *engMsg) *engineConn {
	slots := reg.Slots
	if slots < 1 {
		slots = 1
	}
	return &engineConn{
		ep: ep, nc: nc, positions: reg.Positions, maxNew: reg.MaxNew, slots: slots, vision: reg.Vision,
		sem: make(chan struct{}, slots), dead: make(chan struct{}),
		runs: map[runID]*run{}, imgWait: map[string]chan int{},
	}
}

func (c *engineConn) close() {
	c.deadOnce.Do(func() {
		close(c.dead)
		c.nc.Close()
	})
}

// send writes one JSON line.
func (c *engineConn) send(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	c.wmu.Lock()
	defer c.wmu.Unlock()
	c.nc.SetWriteDeadline(time.Now().Add(30 * time.Second))
	_, err = c.nc.Write(b)
	return err
}

func (c *engineConn) newRunID() runID {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextRun++
	return runID("r" + strconv.FormatUint(c.nextRun, 10))
}

func (c *engineConn) addRun(r *run) {
	c.mu.Lock()
	c.runs[r.id] = r
	c.mu.Unlock()
}

func (c *engineConn) removeRun(id runID) {
	c.mu.Lock()
	delete(c.runs, id)
	c.mu.Unlock()
}

// readLoop dispatches the engine's lines until its socket closes.
func (c *engineConn) readLoop(br *bufio.Reader) error {
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			var m engMsg
			if jerr := json.Unmarshal(line, &m); jerr != nil {
				return fmt.Errorf("malformed line from the engine: %v", jerr)
			}
			c.dispatch(&m)
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func (c *engineConn) dispatch(m *engMsg) {
	switch m.Op {
	case "prefill", "tokens", "stats", "done", "error":
		c.mu.Lock()
		r := c.runs[m.ID]
		c.mu.Unlock()
		if r != nil {
			r.push(m)
		}
	case "image":
		c.mu.Lock()
		ch := c.imgWait[m.Img]
		c.mu.Unlock()
		if ch != nil {
			select {
			case ch <- m.Tokens:
			default:
			}
		}
	}
	// other ops (a second register, anything newer) are ignored
}

// loseRuns tells every run on this socket that the engine is gone.
func (c *engineConn) loseRuns() {
	c.mu.Lock()
	rs := make([]*run, 0, len(c.runs))
	for _, r := range c.runs {
		rs = append(rs, r)
	}
	c.mu.Unlock()
	for _, r := range rs {
		r.push(&engMsg{Op: "error", lost: true, Message: "the engine's connection closed during the run"})
	}
}

// ---- the socket server -----------------------------------------------------------------------------

func (s *Server) serveEngines() {
	for {
		nc, err := s.engLn.Accept()
		if err != nil {
			if s.quitting() {
				return
			}
			s.logf("engine socket: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		go s.handleEngine(nc)
	}
}

func refuse(nc net.Conn, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	b, _ := json.Marshal(map[string]any{"op": "error", "message": "register refused: " + msg})
	nc.SetWriteDeadline(time.Now().Add(2 * time.Second))
	nc.Write(append(b, '\n'))
	nc.Close()
}

func (s *Server) handleEngine(nc net.Conn) {
	defer nc.Close()
	uc, ok := nc.(*net.UnixConn)
	if !ok {
		return
	}
	pid, uid, err := peerCred(uc)
	if err != nil {
		refuse(nc, "cannot read the peer credentials: %v", err)
		return
	}
	br := bufio.NewReaderSize(nc, 1<<20)
	nc.SetReadDeadline(time.Now().Add(10 * time.Second))
	line, err := br.ReadBytes('\n')
	if err != nil {
		refuse(nc, "no register line: %v", err)
		return
	}
	var reg engMsg
	if err := json.Unmarshal(line, &reg); err != nil || reg.Op != "register" {
		refuse(nc, "the first line must be a register")
		return
	}
	ep := s.eps[reg.Endpoint]
	if ep == nil {
		refuse(nc, "unknown endpoint %q", reg.Endpoint)
		return
	}
	if uid != os.Getuid() {
		refuse(nc, "peer uid %d is not ai-mux's", uid)
		return
	}
	// the supervision loop records the process right after starting it; a very quick engine may be first
	var p *proc
	for i := 0; i < 40; i++ {
		ep.mu.Lock()
		p = ep.p
		ep.mu.Unlock()
		if p != nil {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if p == nil || !p.contains(pid) {
		s.logf("register for %s refused: pid %d is not in the process ai-mux started", reg.Endpoint, pid)
		refuse(nc, "peer pid %d is not a process ai-mux started for endpoint %s", pid, reg.Endpoint)
		return
	}
	if reg.Positions < ep.cfg.Ctx {
		refuse(nc, "positions %d is less than the endpoint's ctx %d", reg.Positions, ep.cfg.Ctx)
		return
	}
	if ep.cfg.Vision && !reg.Vision {
		refuse(nc, "the endpoint is configured with vision but the engine has none")
		return
	}
	c := newEngineConn(ep, nc, &reg)
	if err := c.send(map[string]any{"op": "registered", "ctx": ep.cfg.Ctx}); err != nil {
		return
	}
	if err := ep.attach(c); err != nil {
		refuse(nc, "%v", err)
		return
	}
	p.setRegistered()
	nc.SetReadDeadline(time.Time{})
	s.logf("endpoint %s registered (pid %d, positions %d, max_new %d, slots %d, vision %v)", ep.name, pid, reg.Positions, reg.MaxNew, c.slots, reg.Vision)
	rerr := c.readLoop(br)
	c.close()
	c.loseRuns()
	ep.detach(c)
	if rerr != nil && !s.quitting() {
		s.logf("endpoint %s: %v", ep.name, rerr)
		if ep.isCurrentProc(p) {
			ep.fail(rerr.Error())
		}
	}
}

func (e *Endpoint) isCurrentProc(p *proc) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.p == p
}
