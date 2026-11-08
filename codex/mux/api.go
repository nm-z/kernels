package mux

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Handler is the client HTTP API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/responses", s.handleResponses)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/props", s.handleProps)
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/services", s.handleServices)
	mux.HandleFunc("/v1/services/", s.handleServiceAction)
	mux.HandleFunc("/v1/capacity", s.handleCapacity)
	mux.HandleFunc("/v1/events", s.handleEvents)
	return mux
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	writeAPIError(w, &apiError{Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed", Message: "use " + allow})
}

// handleModels lists what a client can ask for by name: engines, shims and embedding services.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	data := make([]map[string]any, 0, len(s.order))
	for _, e := range s.order {
		in := e.Info()
		if in.Kind == KindJob {
			continue
		}
		m := map[string]any{"id": in.Name, "object": "model", "kind": in.Kind, "vision": in.Vision, "state": in.State}
		if in.Kind == KindEmbed {
			m["dims"] = in.Dims
		} else {
			m["ctx"] = in.Ctx
			m["efforts"] = in.Efforts
		}
		data = append(data, m)
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) serviceInfos() []ServiceInfo {
	out := make([]ServiceInfo, 0, len(s.order))
	for _, e := range s.order {
		out = append(out, e.Info())
	}
	return out
}

func (s *Server) handleServices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": s.serviceInfos()})
}

// GET /v1/capacity returns mux state and provider-reported facts for every shim.
func (s *Server) handleCapacity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, "GET")
		return
	}
	sources := make([]map[string]any, 0, len(s.order))
	for _, e := range s.order {
		if e.kind != KindShim {
			continue
		}
		state, facts, since, slots := e.CapacitySnapshot()
		usage, err := s.hub.capacityUsageToday(e.name, time.Now())
		if err != nil {
			s.logf("capacity: read provider usage for %s: %v", e.name, err)
			usage = capacityUsage{}
		}
		tokensToday := facts.TokensServedToday
		tokenBasis := "unreported"
		if hasCapacityValue(tokensToday) {
			tokenBasis = "provider_reported"
		} else if usage.Replies > 0 && usage.TokenReports == usage.Replies {
			tokensToday = usage.TotalTokens
			tokenBasis = "provider_reported"
		} else if usage.Replies > 0 {
			tokenBasis = "incomplete"
		} else {
			tokenBasis = "no_replies"
		}
		var reportedTokens, tokenReports any = "", ""
		if usage.TokenReports > 0 {
			reportedTokens, tokenReports = usage.TotalTokens, usage.TokenReports
		}
		var parallelLimit any = ""
		parallelBasis := ""
		if slots > 0 {
			parallelLimit, parallelBasis = slots, "mux shim slots"
		}
		sources = append(sources, map[string]any{
			"source": e.name, "kind": e.kind, "adapter": "mux shim",
			"active_sessions": "", "session_cap": "", "configured_session_cap": "",
			"session_cap_basis": "", "parallel_generation_limit": parallelLimit,
			"parallel_generation_basis": parallelBasis, "model_states": "",
			"state": state, "state_at": since.Unix(), "state_basis": "mux service state",
			"remaining": facts.Remaining, "reset_time": facts.ResetTime,
			"characters_sent_today": "", "characters_received_today": "",
			"tokens_served_today": tokensToday, "tokens_served_basis": tokenBasis,
			"reported_tokens_today": reportedTokens, "token_reports_today": tokenReports,
			"replies_today": usage.Replies, "errors_today": usage.Errors, "discovery": "",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"sources": sources})
}

func hasCapacityValue(value any) bool {
	if value == nil {
		return false
	}
	if text, ok := value.(string); ok {
		return text != ""
	}
	return true
}

// POST /v1/services/NAME/up and /down
func (s *Server) handleServiceAction(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/v1/services/")
	name, action, ok := strings.Cut(rest, "/")
	if !ok || (action != "up" && action != "down") {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "use POST /v1/services/NAME/up or /down"})
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	e := s.svcs[name]
	if e == nil {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "service_not_found", Message: fmt.Sprintf("no service %q", name)})
		return
	}
	if err := s.SetServiceUp(name, action == "up"); err != nil {
		writeAPIError(w, &apiError{Status: http.StatusInternalServerError, Type: "server_error", Code: "config_write_failed", Message: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, e.Info())
}

// GET /v1/events[?since=TS]: without since, the current state and then what happens; with it, the
// recorded events since TS (an RFC 3339 time or Unix seconds) and then what happens. Last-Event-ID
// resumes after an event's id. Any number of observers.
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
	var since string
	var after uint64
	replay := false
	if v := r.URL.Query().Get("since"); v != "" {
		ts, err := parseSince(v)
		if err != nil {
			writeAPIError(w, badRequest("invalid_since", "since", "%v", err))
			return
		}
		since, replay = ts, true
	}
	if v := r.Header.Get("Last-Event-ID"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			after, replay = n, true
		}
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	rc := http.NewResponseController(w)
	started := false
	// the headers go out before the first event, and as soon as the observer is subscribed: a client that
	// has its headers is sure to see what happens next
	start := func() {
		if !started {
			started = true
			w.WriteHeader(http.StatusOK)
			fl.Flush()
		}
	}
	write := func(ev HubEvent) bool {
		start()
		rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
		var err error
		if ev.Seq > 0 {
			_, err = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", ev.Seq, ev.Type, ev.Data)
		} else {
			_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Type, ev.Data)
		}
		if err != nil {
			return false
		}
		fl.Flush()
		return true
	}
	var sentUpTo uint64
	if replay {
		// the recorded events up to now, then (once subscribed) the ones that came in between, then live
		sentUpTo = s.hub.Seq()
		evs, err := s.hub.Replay(since, after, sentUpTo)
		if err != nil {
			s.logf("events replay: %v", err)
		}
		for _, ev := range evs {
			if !write(ev) {
				return
			}
		}
	}
	sub := s.hub.Subscribe()
	defer s.hub.Unsubscribe(sub)
	if replay {
		evs, err := s.hub.Replay("", sentUpTo, 0)
		if err != nil {
			s.logf("events replay: %v", err)
		}
		for _, ev := range evs {
			if !write(ev) {
				return
			}
			sentUpTo = ev.Seq
		}
	} else {
		start()
		snap := map[string]any{"type": "snapshot", "time": time.Now().UTC().Format(tsLayout), "services": s.serviceInfos()}
		if !write(HubEvent{Type: "snapshot", Data: mustJSON(snap)}) {
			return
		}
	}
	start()
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()
	for {
		select {
		case ev := <-sub.C:
			if ev.Seq <= sentUpTo {
				continue
			}
			if !write(ev) {
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
