package mux

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const maxBodyBytes = 256 << 20

// handleResponses is POST /v1/responses for engine and shim services.
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		writeAPIError(w, badRequest("invalid_request", "", "cannot read the request body: %v", err))
		return
	}
	var probe struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &probe) != nil {
		writeAPIError(w, badRequest("invalid_json", "", "the body is not a JSON object"))
		return
	}
	svc, want := s.svcs[probe.Model], ""
	if svc == nil {
		svc, want = s.resolveRoute(probe.Model)
	}
	if svc == nil || svc.kind == KindJob {
		writeAPIError(w, s.modelNotFound(probe.Model))
		return
	}
	if svc.kind == KindEmbed {
		writeAPIError(w, badRequest("not_a_chat_model", "model", "%s is an embedding model: use POST /v1/embeddings", svc.name))
		return
	}
	pr, aerr := prepare(body, svc.cfg)
	if aerr != nil {
		writeAPIError(w, aerr)
		return
	}
	pr.want = want
	s.dispatch(w, r, svc, pr)
}

// dispatch runs a prepared request on its service: an engine or a shim, answered as the request asked
// (Responses events, or chat completion chunks).
func (s *Server) dispatch(w http.ResponseWriter, r *http.Request, svc *Service, pr *prepared) {
	conn, state := svc.currentConn()
	if conn == nil {
		writeAPIError(w, &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "service_unavailable",
			Message: fmt.Sprintf("model %s is %s", svc.name, state)})
		return
	}
	if svc.kind == KindShim {
		s.respondShim(w, r, svc, conn, pr)
		return
	}
	s.respondEngine(w, r, svc, conn, pr)
}

// openStream starts the SSE response (or none for stream:false) and the response being built.
func openStream(w http.ResponseWriter, pr *prepared) sink {
	if !pr.stream {
		return nil
	}
	var sk sink
	if pr.chat {
		sk = newChatSink(w, pr.model)
	} else {
		sk = newSSESink(w)
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	if fl, ok := w.(http.Flusher); ok {
		fl.Flush()
	}
	return sk
}

// failResponse ends a response that failed: the HTTP error for stream:false, response.failed otherwise.
func failResponse(w http.ResponseWriter, pr *prepared, rb *respBuilder, e *apiError) {
	if pr.stream {
		rb.failedEvent(e.Code, e.Message)
	} else {
		writeAPIError(w, e)
	}
}

// settleSummary turns what a compaction run wrote into the compaction item, or says why it cannot.
func settleSummary(rb *respBuilder) *apiError {
	summary := strings.TrimSpace(rb.summary.String())
	if summary == "" {
		return &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "compaction_failed", Message: "the model wrote no summary"}
	}
	rb.compactionItem(summary)
	return nil
}

// completeResponse writes the end of a successful response.
func completeResponse(w http.ResponseWriter, pr *prepared, rb *respBuilder, usage map[string]any, incomplete bool, extra map[string]any) {
	if pr.stream {
		if incomplete {
			rb.incomplete(usage, "max_output_tokens")
		} else {
			rb.completed(usage)
		}
		return
	}
	if pr.chat {
		writeJSON(w, http.StatusOK, chatCompletionObj(rb, usage, incomplete))
		return
	}
	status := "completed"
	ex := map[string]any{"usage": usageOrNull(usage)}
	if incomplete {
		status = "incomplete"
		ex["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	for k, v := range extra {
		ex[k] = v
	}
	writeJSON(w, http.StatusOK, rb.responseObj(status, ex))
}

// usageOrNull is the usage object of a Responses reply: the provider's token counts when it
// reported any, else null. Codex reads `usage` as numbers, so an empty object breaks the stream.
func usageOrNull(usage map[string]any) any {
	if len(usage) == 0 {
		return nil
	}
	return usage
}
