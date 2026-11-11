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

// runID is a run's id. mux makes them (strings); a service may echo them as numbers.
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

// msg is any line a service sends, of any kind. Fields whose absence is a protocol violation are pointers.
type msg struct {
	Op string `json:"op"`

	// register
	Service   string `json:"service"`
	Kind      string `json:"kind"`
	Positions int    `json:"positions"` // engine
	MaxNew    int    `json:"max_new"`   // engine
	Slots     int    `json:"slots"`     // engine, shim
	Vision    bool   `json:"vision"`    // engine, shim
	Auto      bool   `json:"auto"`      // shim
	Dims      int    `json:"dims"`      // embed
	MaxBatch  int    `json:"max_batch"` // embed
	Device    string `json:"device"`    // embed
	Title     string `json:"title"`     // job

	// run lines
	ID         runID    `json:"id"`
	Done       *int     `json:"done"`  // engine prefill
	Total      *int     `json:"total"` // engine prefill
	IDs        []uint32 `json:"ids"`   // engine tokens
	PrefillTPS *float64 `json:"prefill_tps"`
	DecodeTPS  *float64 `json:"decode_tps"`
	IdlePct    *float64 `json:"idle_pct"`
	Prompt     *int     `json:"prompt"`
	Generated  *int     `json:"generated"`
	Stopped    bool     `json:"stopped"`
	Message    string   `json:"message"`
	Img        string   `json:"img"`    // engine image reply
	Tokens     int      `json:"tokens"` // engine image reply

	// shim
	Trying            []string        `json:"trying"`
	Failed            []string        `json:"failed"`
	Why               []string        `json:"why"`    // one reason a failed route, in order
	Model             string          `json:"model"`  // the model answering this run, as its provider names it
	Models            []string        `json:"models"` // every model the shim's provider offers this account
	MS                *float64        `json:"ms"`
	Channel           string          `json:"channel"`
	Text              string          `json:"text"`
	CallID            string          `json:"call_id"`
	Name              string          `json:"name"`
	Arguments         json.RawMessage `json:"arguments"`
	InputTokens       *int            `json:"input_tokens"`
	OutputTokens      *int            `json:"output_tokens"`
	TotalTokens       *int            `json:"total_tokens"`
	Remaining         any             `json:"remaining"`
	ResetTime         any             `json:"reset_time"`
	TokensServedToday any             `json:"tokens_served_today"`

	// embed
	Vectors [][]float32 `json:"vectors"`

	// job
	Pct    *float64 `json:"pct"`
	Detail string   `json:"detail"`
	OK     *bool    `json:"ok"`

	lost bool // synthetic: the socket closed under the run
}

// run is one request's run on a service: the lines the service sends for it, queued for the handler.
type run struct {
	id  runID
	mu  sync.Mutex
	q   []*msg
	sig chan struct{}
}

func newRun(id runID) *run { return &run{id: id, sig: make(chan struct{}, 1)} }

func (r *run) push(m *msg) {
	r.mu.Lock()
	r.q = append(r.q, m)
	r.mu.Unlock()
	select {
	case r.sig <- struct{}{}:
	default:
	}
}

func (r *run) take() []*msg {
	r.mu.Lock()
	q := r.q
	r.q = nil
	r.mu.Unlock()
	return q
}

// svcConn is a registered service's socket.
type svcConn struct {
	svc  *Service
	nc   net.Conn
	kind string

	positions int // engine
	maxNew    int // engine
	slots     int // engine, shim; embed 1
	vision    bool
	auto      bool   // shim
	dims      int    // embed
	maxBatch  int    // embed
	device    string // embed
	title     string // job
	capacity  CapacityFacts

	sem      chan struct{} // one token per request running at once
	dead     chan struct{}
	deadOnce sync.Once

	wmu sync.Mutex

	mu      sync.Mutex
	runs    map[runID]*run
	imgWait map[string]chan int
	nextRun uint64
}

func newSvcConn(svc *Service, nc net.Conn, reg *msg) *svcConn {
	c := &svcConn{
		svc: svc, nc: nc, kind: svc.kind, positions: reg.Positions, maxNew: reg.MaxNew, slots: reg.Slots, vision: reg.Vision,
		auto: reg.Auto, dims: reg.Dims, maxBatch: reg.MaxBatch, device: reg.Device, title: reg.Title,
		capacity: capacityFactsFromMessage(reg),
		dead:     make(chan struct{}), runs: map[runID]*run{}, imgWait: map[string]chan int{},
	}
	switch c.kind {
	case KindEmbed:
		c.slots = 1 // the protocol has no slots for an embed service: one batch at a time
		if c.maxBatch < 1 {
			c.maxBatch = 64
		}
	case KindJob:
		c.slots = 0
	default:
		if c.slots < 1 {
			c.slots = 1
		}
	}
	c.sem = make(chan struct{}, maxInt(c.slots, 1))
	return c
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (c *svcConn) close() {
	c.deadOnce.Do(func() {
		close(c.dead)
		c.nc.Close()
	})
}

// send writes one JSON line.
func (c *svcConn) send(v any) error {
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

func (c *svcConn) newRunID() runID {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextRun++
	return runID("r" + strconv.FormatUint(c.nextRun, 10))
}

func (c *svcConn) addRun(r *run) {
	c.mu.Lock()
	c.runs[r.id] = r
	c.mu.Unlock()
}

func (c *svcConn) removeRun(id runID) {
	c.mu.Lock()
	delete(c.runs, id)
	c.mu.Unlock()
}

// protocolError is a service breaking the protocol (as opposed to its socket just closing, which the process
// exiting explains better).
type protocolError struct{ msg string }

func (e protocolError) Error() string { return e.msg }

// readLoop dispatches the service's lines until its socket closes.
func (c *svcConn) readLoop(br *bufio.Reader) error {
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			var m msg
			if jerr := json.Unmarshal(line, &m); jerr != nil {
				return protocolError{fmt.Sprintf("malformed line from the service: %v", jerr)}
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

func (c *svcConn) dispatch(m *msg) {
	switch m.Op {
	case "prefill", "tokens", "stats", "done", "error", "route", "model", "ttft", "delta", "tool_call", "vectors":
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
	case "progress":
		if c.kind == KindJob {
			c.svc.jobProgress(c, m)
		}
	case "finished":
		if c.kind == KindJob {
			c.svc.jobFinished(c, m)
		}
	case "models":
		if c.kind == KindShim {
			c.svc.reportModels(m.Models)
		}
	case "capacity":
		if c.kind == KindShim {
			c.svc.reportCapacity(c, m)
		}
	}
	// other ops (a second register, anything newer) are ignored
}

// loseRuns tells every run on this socket that the service is gone.
func (c *svcConn) loseRuns() {
	c.mu.Lock()
	rs := make([]*run, 0, len(c.runs))
	for _, r := range c.runs {
		rs = append(rs, r)
	}
	c.mu.Unlock()
	for _, r := range rs {
		r.push(&msg{Op: "error", lost: true, Message: "the service's connection closed during the run"})
	}
}

// ---- the socket server -----------------------------------------------------------------------------

func (s *Server) serveSocket() {
	for {
		nc, err := s.sockLn.Accept()
		if err != nil {
			if s.quitting() {
				return
			}
			s.logf("service socket: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		go s.handleConn(nc)
	}
}

func refuse(nc net.Conn, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	b, _ := json.Marshal(map[string]any{"op": "error", "message": "register refused: " + msg})
	nc.SetWriteDeadline(time.Now().Add(2 * time.Second))
	nc.Write(append(b, '\n'))
	nc.Close()
}

func (s *Server) handleConn(nc net.Conn) {
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
	var reg msg
	if err := json.Unmarshal(line, &reg); err != nil || reg.Op != "register" {
		refuse(nc, "the first line must be a register")
		return
	}
	svc := s.svcs[reg.Service]
	if svc == nil {
		refuse(nc, "unknown service %q", reg.Service)
		return
	}
	if uid != os.Getuid() {
		refuse(nc, "peer uid %d is not mux's", uid)
		return
	}
	// the supervision loop records the process right after starting it; a very quick service may be first
	var p *proc
	for i := 0; i < 40; i++ {
		svc.mu.Lock()
		p = svc.p
		svc.mu.Unlock()
		if p != nil {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if p == nil || !p.contains(pid) {
		s.logf("register for %s refused: pid %d is not in the process mux started", reg.Service, pid)
		refuse(nc, "peer pid %d is not a process mux started for service %s", pid, reg.Service)
		return
	}
	if reg.Kind != svc.kind {
		refuse(nc, "kind %q is not the kind of service %s (service.yaml says %q)", reg.Kind, svc.name, svc.kind)
		return
	}
	switch svc.kind {
	case KindEngine:
		if reg.Positions < svc.cfg.Ctx {
			refuse(nc, "positions %d is less than the service's ctx %d", reg.Positions, svc.cfg.Ctx)
			return
		}
		if svc.cfg.Vision && !reg.Vision {
			refuse(nc, "the service is configured with vision but the engine has none")
			return
		}
	case KindShim:
		if svc.cfg.Vision && !reg.Vision {
			refuse(nc, "the service is configured with vision but the shim has none")
			return
		}
	case KindEmbed:
		if reg.Dims < 1 {
			refuse(nc, "an embed service must register its dims")
			return
		}
	}
	c := newSvcConn(svc, nc, &reg)
	ack := map[string]any{"op": "registered"}
	if svc.kind == KindEngine || svc.kind == KindShim {
		ack["ctx"] = svc.cfg.Ctx
	}
	if err := c.send(ack); err != nil {
		return
	}
	if err := svc.attach(c); err != nil {
		refuse(nc, "%v", err)
		return
	}
	p.setRegistered()
	nc.SetReadDeadline(time.Time{})
	s.logf("service %s (%s) registered (pid %d, positions %d, max_new %d, slots %d, vision %v, dims %d)", svc.name, svc.kind, pid, reg.Positions, reg.MaxNew, c.slots, reg.Vision, reg.Dims)
	rerr := c.readLoop(br)
	c.close()
	c.loseRuns()
	svc.detach(c)
	if rerr != nil && !s.quitting() {
		s.logf("service %s: %v", svc.name, rerr)
		var pe protocolError
		if errors.As(rerr, &pe) && svc.isCurrentProc(p) {
			svc.fail(rerr.Error())
		}
	}
}

func (e *Service) isCurrentProc(p *proc) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.p == p
}
