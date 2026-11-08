// Package mux is mux: the one process on a host that serves AI. Everything AI on the host (a model
// engine, a shim over another provider, an embedding server, a job) is a service: one process mux
// starts and stops itself, in a systemd scope, and that registers with it over a Unix socket. Clients
// talk to mux over HTTP. See ../PROTOCOL.md.
package mux

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Options configure a Server.
type Options struct {
	Config     *Config
	LogDir     string                           // service logs and events.db; default $XDG_STATE_HOME/mux or ~/.local/state/mux
	EventsDB   string                           // default LogDir/events.db
	Supervisor string                           // "auto" (default), "systemd" or "plain"; also MUX_SUPERVISOR
	Unit       string                           // the unit scopes are bound to (default mux.service; also MUX_UNIT)
	LoadModel  func(string) (Model, error)      // default LoadTokenizer
	Logf       func(format string, args ...any) // default log.Printf
	KillGrace  time.Duration                    // plain mode: SIGTERM to SIGKILL (default 10 s)
	MinBackoff time.Duration                    // first restart delay (default 10 s; it doubles to 5 min)

	JobProgressEvery time.Duration // a job must send progress at least this often (default 10 s)
	JobFinishGrace   time.Duration // how long a job's process may linger after `finished` (default 30 s)
}

// Server is mux.
type Server struct {
	cfg         *Config
	hub         *Hub
	svcs        map[string]*Service
	order       []*Service
	models      modelCache
	logDir      string
	systemd     bool
	killGrace   time.Duration
	logfn       func(string, ...any)
	unit        string // the systemd unit service scopes are bound to
	minBack     time.Duration
	jobEvery    time.Duration
	finishGrace time.Duration

	sockLn    net.Listener
	httpLn    net.Listener
	httpSrv   *http.Server
	compatLn  net.Listener // the llama.cpp face (compat_listen), if configured
	compatSrv *http.Server

	quit     chan struct{}
	quitOnce sync.Once
	wg       sync.WaitGroup
}

func New(o Options) (*Server, error) {
	if o.Config == nil {
		return nil, errors.New("no config")
	}
	s := &Server{cfg: o.Config, svcs: map[string]*Service{}, quit: make(chan struct{}), killGrace: o.KillGrace}
	if s.killGrace == 0 {
		s.killGrace = 10 * time.Second
	}
	s.logfn = o.Logf
	if s.logfn == nil {
		s.logfn = log.Printf
	}
	load := o.LoadModel
	if load == nil {
		load = LoadTokenizer
	}
	s.models.load = load
	s.logDir = o.LogDir
	if s.logDir == "" {
		base := os.Getenv("XDG_STATE_HOME")
		if base == "" {
			home, _ := os.UserHomeDir()
			base = filepath.Join(home, ".local", "state")
		}
		s.logDir = filepath.Join(base, "mux")
	}
	if err := os.MkdirAll(s.logDir, 0o755); err != nil {
		return nil, err
	}
	db := o.EventsDB
	if db == "" {
		db = filepath.Join(s.logDir, "events.db")
	}
	hub, err := NewRecordingHub(db)
	if err != nil {
		return nil, fmt.Errorf("events.db: %w", err)
	}
	s.hub = hub
	mode := o.Supervisor
	if mode == "" {
		mode = os.Getenv("MUX_SUPERVISOR")
	}
	s.minBack = o.MinBackoff
	if s.minBack == 0 {
		s.minBack = minBackoff
	}
	s.jobEvery = o.JobProgressEvery
	if s.jobEvery == 0 {
		s.jobEvery = 10 * time.Second
	}
	s.finishGrace = o.JobFinishGrace
	if s.finishGrace == 0 {
		s.finishGrace = 30 * time.Second
	}
	s.unit = o.Unit
	if s.unit == "" {
		s.unit = os.Getenv("MUX_UNIT")
	}
	if s.unit == "" {
		s.unit = "mux.service"
	}
	s.systemd = useSystemd(mode, s.unit)
	for _, c := range s.cfg.Services {
		e := newService(s, c)
		s.svcs[c.Name] = e
		s.order = append(s.order, e)
	}
	return s, nil
}

func (s *Server) logf(format string, args ...any) { s.logfn(format, args...) }

func (s *Server) quitting() bool {
	select {
	case <-s.quit:
		return true
	default:
		return false
	}
}

// jobStall is how long a job may go without progress before it is failed: its interval plus a quarter.
func (s *Server) jobStall() time.Duration { return s.jobEvery + s.jobEvery/4 }

// Supervised reports whether services are started in systemd scopes.
func (s *Server) Supervised() bool { return s.systemd }

// Service returns one service by name.
func (s *Server) Service(name string) *Service { return s.svcs[name] }

// Hub is the event hub (GET /v1/events, events.db).
func (s *Server) Hub() *Hub { return s.hub }

// Addr is the HTTP address mux listens on.
func (s *Server) Addr() string { return s.httpLn.Addr().String() }

// CompatAddr is the address of the llama.cpp face (compat_listen), or "" if there is none.
func (s *Server) CompatAddr() string {
	if s.compatLn == nil {
		return ""
	}
	return s.compatLn.Addr().String()
}

// Start opens the service socket and the HTTP listener, and starts every service that should be up.
func (s *Server) Start() error {
	sock := s.cfg.Socket
	if err := os.MkdirAll(filepath.Dir(sock), 0o755); err != nil {
		return err
	}
	if c, err := net.DialTimeout("unix", sock, time.Second); err == nil {
		c.Close()
		return fmt.Errorf("%s: another mux is already listening", sock)
	}
	os.Remove(sock)
	ln, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	os.Chmod(sock, 0o600)
	s.sockLn = ln
	hl, err := net.Listen("tcp", s.cfg.Listen)
	if err != nil {
		ln.Close()
		return err
	}
	s.httpLn = hl
	s.httpSrv = &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	if s.cfg.CompatListen != "" {
		cl, err := net.Listen("tcp", s.cfg.CompatListen)
		if err != nil {
			hl.Close()
			ln.Close()
			return fmt.Errorf("compat_listen: %w", err)
		}
		s.compatLn = cl
		s.compatSrv = &http.Server{Handler: s.compatHandler(), ReadHeaderTimeout: 10 * time.Second}
		go func() {
			if err := s.compatSrv.Serve(cl); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logf("compat http: %v", err)
			}
		}()
	}
	go s.serveSocket()
	go func() {
		if err := s.httpSrv.Serve(hl); err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.logf("http: %v", err)
		}
	}()
	for _, e := range s.order {
		s.wg.Add(1)
		go func(e *Service) {
			defer s.wg.Done()
			e.loop()
		}(e)
	}
	s.hub.Publish("mux.start", map[string]any{"listen": hl.Addr().String(), "services": len(s.order)})
	s.logf("mux: listening on %s, services on %s (%s), %d services", hl.Addr(), sock, map[bool]string{true: "systemd scopes", false: "plain process groups"}[s.systemd], len(s.order))
	if s.compatLn != nil {
		s.logf("mux: llama.cpp face for %s on %s", s.cfg.CompatService, s.compatLn.Addr())
	}
	return nil
}

// Shutdown stops every service (processes and everything they started), then the listeners.
func (s *Server) Shutdown() {
	s.quitOnce.Do(func() { close(s.quit) })
	done := make(chan struct{})
	go func() { s.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		s.logf("shutdown: services did not stop within 2 minutes")
	}
	for _, srv := range []*http.Server{s.httpSrv, s.compatSrv} {
		if srv != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			srv.Shutdown(ctx)
			cancel()
			srv.Close()
		}
	}
	if s.sockLn != nil {
		s.sockLn.Close()
		os.Remove(s.cfg.Socket)
	}
	s.hub.Publish("mux.stop", nil)
	s.hub.Close()
}

// SetServiceUp changes a service's desired state and rewrites `up:` in the host file.
func (s *Server) SetServiceUp(name string, up bool) error {
	e := s.svcs[name]
	if e == nil {
		return fmt.Errorf("no service %q", name)
	}
	if s.cfg.Path != "" {
		if err := SetUpInFile(s.cfg.Path, name, up); err != nil {
			return err
		}
	}
	e.SetDesired(up)
	return nil
}
