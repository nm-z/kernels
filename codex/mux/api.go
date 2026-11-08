package mux

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Handler is the client HTTP API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/responses", s.handleResponses)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/endpoints", s.handleEndpoints)
	mux.HandleFunc("/v1/endpoints/", s.handleEndpointAction)
	mux.HandleFunc("/v1/events", s.handleEvents)
	return mux
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	writeAPIError(w, &apiError{Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed", Message: "use " + allow})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	data := make([]map[string]any, 0, len(s.order))
	for _, e := range s.order {
		in := e.Info()
		data = append(data, map[string]any{
			"id": in.Name, "object": "model", "ctx": in.Ctx, "efforts": in.Efforts, "vision": in.Vision, "state": in.State,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) endpointInfos() []EndpointInfo {
	out := make([]EndpointInfo, 0, len(s.order))
	for _, e := range s.order {
		out = append(out, e.Info())
	}
	return out
}

func (s *Server) handleEndpoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": s.endpointInfos()})
}

// POST /v1/endpoints/NAME/up and /down
func (s *Server) handleEndpointAction(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/v1/endpoints/")
	name, action, ok := strings.Cut(rest, "/")
	if !ok || (action != "up" && action != "down") {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "use POST /v1/endpoints/NAME/up or /down"})
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	e := s.eps[name]
	if e == nil {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found", Message: fmt.Sprintf("no endpoint %q", name)})
		return
	}
	if err := s.SetEndpointUp(name, action == "up"); err != nil {
		writeAPIError(w, &apiError{Status: http.StatusInternalServerError, Type: "server_error", Code: "config_write_failed", Message: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, e.Info())
}

// GET /v1/events: everything every endpoint does, to any number of observers.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	fl, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, &apiError{Status: 500, Type: "server_error", Code: "no_flush", Message: "streaming is not supported here"})
		return
	}
	sub := s.hub.Subscribe()
	defer s.hub.Unsubscribe(sub)
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	rc := http.NewResponseController(w)
	write := func(typ string, data []byte) bool {
		rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", typ, data); err != nil {
			return false
		}
		fl.Flush()
		return true
	}
	snap, _ := json.Marshal(map[string]any{"type": "snapshot", "time": time.Now().Format(time.RFC3339Nano), "endpoints": s.endpointInfos()})
	if !write("snapshot", snap) {
		return
	}
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()
	for {
		select {
		case ev := <-sub.C:
			if !write(ev.Type, ev.Data) {
				return
			}
		case <-tick.C:
			rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
			if _, err := fmt.Fprint(w, ": keepalive\n\n"); err != nil {
				return
			}
			fl.Flush()
		case <-r.Context().Done():
			return
		case <-s.quit:
			return
		}
	}
}
