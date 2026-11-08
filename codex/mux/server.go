// Package mux is ai-mux: the one process on a host that serves AI. It starts and stops every
// endpoint (an engine process in a systemd scope), takes the engine's registration on a Unix socket,
// and serves the Responses API to clients over HTTP. See ../PROTOCOL.md.
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
	LogDir     string                           // endpoint logs; default $XDG_STATE_HOME/ai-mux or ~/.local/state/ai-mux
	Supervisor string                           // "auto" (default), "systemd" or "plain"; also AI_MUX_SUPERVISOR
	LoadModel  func(string) (Model, error)      // default LoadTokenizer
	Logf       func(format string, args ...any) // default log.Printf
	KillGrace  time.Duration                    // plain mode: SIGTERM to SIGKILL (default 10 s)
	Unit       string                           // the unit scopes are bound to (default ai-mux.service; also AI_MUX_UNIT)
	MinBackoff time.Duration                    // first restart delay (default 10 s; it doubles to 5 min)
}

// Server is ai-mux.
type Server struct {
	cfg       *Config
	hub       *Hub
	eps       map[string]*Endpoint
	order     []*Endpoint
	models    modelCache
	logDir    string
	systemd   bool
	killGrace time.Duration
	logfn     func(string, ...any)
	unit      string // the systemd unit endpoint scopes are bound to
	minBack   time.Duration

	engLn   net.Listener
	httpLn  net.Listener
	httpSrv *http.Server

	quit     chan struct{}
	quitOnce sync.Once
	wg       sync.WaitGroup
}

func New(o Options) (*Server, error) {
	if o.Config == nil {
		return nil, errors.New("no config")
	}
	s := &Server{cfg: o.Config, hub: NewHub(), eps: map[string]*Endpoint{}, quit: make(chan struct{}), killGrace: o.KillGrace}
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
		s.logDir = filepath.Join(base, "ai-mux")
	}
	mode := o.Supervisor
	if mode == "" {
		mode = os.Getenv("AI_MUX_SUPERVISOR")
	}
	s.minBack = o.MinBackoff
	if s.minBack == 0 {
		s.minBack = minBackoff
	}
	s.unit = o.Unit
	if s.unit == "" {
		s.unit = os.Getenv("AI_MUX_UNIT")
	}
	if s.unit == "" {
		s.unit = "ai-mux.service"
	}
	s.systemd = useSystemd(mode, s.unit)
	for _, c := range s.cfg.Endpoints {
		e := newEndpoint(s, c)
		s.eps[c.Name] = e
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

// Supervised reports whether endpoints are started in systemd scopes.
func (s *Server) Supervised() bool { return s.systemd }

// Endpoint returns one endpoint by name.
func (s *Server) Endpoint(name string) *Endpoint { return s.eps[name] }

// Hub is the event hub (GET /v1/events).
func (s *Server) Hub() *Hub { return s.hub }

// Addr is the HTTP address ai-mux listens on.
func (s *Server) Addr() string { return s.httpLn.Addr().String() }

// Start opens the engine socket and the HTTP listener, and starts every endpoint that should be up.
func (s *Server) Start() error {
	sock := s.cfg.Socket
	if err := os.MkdirAll(filepath.Dir(sock), 0o755); err != nil {
		return err
	}
	if c, err := net.DialTimeout("unix", sock, time.Second); err == nil {
		c.Close()
		return fmt.Errorf("%s: another ai-mux is already listening", sock)
	}
	os.Remove(sock)
	ln, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	os.Chmod(sock, 0o600)
	s.engLn = ln
	hl, err := net.Listen("tcp", s.cfg.Listen)
	if err != nil {
		ln.Close()
		return err
	}
	s.httpLn = hl
	s.httpSrv = &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go s.serveEngines()
	go func() {
		if err := s.httpSrv.Serve(hl); err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.logf("http: %v", err)
		}
	}()
	for _, e := range s.order {
		s.wg.Add(1)
		go func(e *Endpoint) {
			defer s.wg.Done()
			e.loop()
		}(e)
	}
	s.logf("ai-mux: listening on %s, engines on %s (%s), %d endpoints", hl.Addr(), sock, map[bool]string{true: "systemd scopes", false: "plain process groups"}[s.systemd], len(s.order))
	return nil
}

// Shutdown stops every endpoint (processes and everything they started), then the listeners.
func (s *Server) Shutdown() {
	s.quitOnce.Do(func() { close(s.quit) })
	done := make(chan struct{})
	go func() { s.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		s.logf("shutdown: endpoints did not stop within 2 minutes")
	}
	if s.httpSrv != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		s.httpSrv.Shutdown(ctx)
		cancel()
		s.httpSrv.Close()
	}
	if s.engLn != nil {
		s.engLn.Close()
		os.Remove(s.cfg.Socket)
	}
}

// SetEndpointUp changes an endpoint's desired state and rewrites `up:` in the compose file.
func (s *Server) SetEndpointUp(name string, up bool) error {
	e := s.eps[name]
	if e == nil {
		return fmt.Errorf("no endpoint %q", name)
	}
	if s.cfg.Path != "" {
		if err := SetUpInFile(s.cfg.Path, name, up); err != nil {
			return err
		}
	}
	e.SetDesired(up)
	return nil
}
