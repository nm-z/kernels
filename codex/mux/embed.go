package mux

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// handleEmbeddings is POST /v1/embeddings (the OpenAI shape) on an embed service.
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		writeAPIError(w, badRequest("invalid_request", "", "cannot read the request body: %v", err))
		return
	}
	var req struct {
		Model          string          `json:"model"`
		Input          json.RawMessage `json:"input"`
		EncodingFormat string          `json:"encoding_format"`
		Dimensions     *int            `json:"dimensions"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeAPIError(w, badRequest("invalid_json", "", "the body is not a JSON object"))
		return
	}
	svc := s.svcs[req.Model]
	if svc == nil || svc.kind == KindJob {
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found", Param: "model",
			Message: fmt.Sprintf("model %q is not served here", req.Model)})
		return
	}
	if svc.kind != KindEmbed {
		writeAPIError(w, badRequest("not_an_embedding_model", "model", "%s is not an embedding model: use POST /v1/responses", svc.name))
		return
	}
	var texts []string
	var one string
	if isNull(req.Input) {
		writeAPIError(w, badRequest("missing_input", "input", "input is required"))
		return
	} else if json.Unmarshal(req.Input, &one) == nil {
		texts = []string{one}
	} else if json.Unmarshal(req.Input, &texts) != nil {
		writeAPIError(w, badRequest("unsupported_input", "input", "input must be a string or an array of strings (token arrays are not supported)"))
		return
	}
	if len(texts) == 0 {
		writeAPIError(w, badRequest("empty_input", "input", "input has no texts"))
		return
	}
	for _, t := range texts {
		if t == "" {
			writeAPIError(w, badRequest("empty_input", "input", "input must not contain an empty string"))
			return
		}
	}
	b64 := false
	switch req.EncodingFormat {
	case "", "float":
	case "base64":
		b64 = true
	default:
		writeAPIError(w, badRequest("invalid_encoding_format", "encoding_format", "encoding_format must be float or base64"))
		return
	}
	conn, state := svc.currentConn()
	if conn == nil {
		writeAPIError(w, &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "service_unavailable",
			Message: fmt.Sprintf("model %s is %s", svc.name, state)})
		return
	}
	if req.Dimensions != nil && *req.Dimensions != conn.dims {
		writeAPIError(w, badRequest("unsupported_parameter", "dimensions", "%s makes %d-dimensional vectors; dimensions cannot change that", svc.name, conn.dims))
		return
	}
	vecs, aerr, cancelled := s.embed(r.Context(), svc, conn, texts)
	if cancelled {
		return
	}
	if aerr != nil {
		writeAPIError(w, aerr)
		return
	}
	data := make([]map[string]any, len(vecs))
	for i, v := range vecs {
		var emb any = v
		if b64 {
			buf := make([]byte, 4*len(v))
			for j, f := range v {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(f))
			}
			emb = base64.StdEncoding.EncodeToString(buf)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": emb}
	}
	tokens := 0
	for _, t := range texts {
		tokens += (len(t) + 3) / 4
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list", "data": data, "model": svc.name,
		"usage": map[string]any{"prompt_tokens": tokens, "total_tokens": tokens},
	})
}

// embed runs the texts on the service, a batch at a time (max_batch), and returns their vectors in order.
func (s *Server) embed(ctx context.Context, svc *Service, conn *svcConn, texts []string) (vecs [][]float32, aerr *apiError, cancelled bool) {
	if !acquireSlot(ctx, svc, conn, nil) {
		if ctx.Err() != nil {
			return nil, nil, true
		}
		return nil, &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "service_unavailable", Message: fmt.Sprintf("model %s went down", svc.name)}, false
	}
	defer func() { <-conn.sem }()
	svc.runStarted()
	var last *LastRun
	defer func() { svc.runEnded(last) }()

	runLabel := conn.newRunID()
	started := time.Now()
	s.hub.Publish("run.start", map[string]any{"service": svc.name, "run": string(runLabel), "texts": len(texts)})
	end := func(status, reason string) {
		f := map[string]any{"service": svc.name, "run": string(runLabel), "status": status, "texts": len(texts), "ms": time.Since(started).Milliseconds()}
		if reason != "" {
			f["reason"] = reason
		}
		s.hub.Publish("run.end", f)
	}
	fail := func(code, format string, args ...any) ([][]float32, *apiError, bool) {
		m := fmt.Sprintf(format, args...)
		end("failed", m)
		return nil, &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: code, Message: fmt.Sprintf("model %s: %s", svc.name, m)}, false
	}
	for i := 0; i < len(texts); i += conn.maxBatch {
		batch := texts[i:minInt(i+conn.maxBatch, len(texts))]
		id := conn.newRunID()
		r := newRun(id)
		conn.addRun(r)
		err := conn.send(map[string]any{"op": "run", "id": id, "texts": batch})
		if err != nil {
			conn.removeRun(id)
			return fail("service_unavailable", "cannot reach the embedding service: %v", err)
		}
		got, why, gone := waitVectors(ctx, conn, r, len(batch), conn.dims)
		conn.removeRun(id)
		switch {
		case gone: // the client left: tell the service and stop
			conn.send(map[string]any{"op": "cancel", "id": id})
			end("cancelled", "")
			return nil, nil, true
		case why != "":
			if strings.HasPrefix(why, "rule 5") {
				s.logf("service %s: %s", svc.name, why)
				svc.fail(why)
				end("violation", why)
				return nil, &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "service_failed", Message: fmt.Sprintf("model %s was dropped: %s", svc.name, why)}, false
			}
			return fail("provider_error", "%s", why)
		}
		vecs = append(vecs, got...)
	}
	last = &LastRun{Texts: len(texts), At: time.Now()}
	end("done", "")
	return vecs, nil, false
}

// waitVectors waits for the `vectors` line of a batch: n vectors of dims numbers each (rule 5).
func waitVectors(ctx context.Context, conn *svcConn, r *run, n, dims int) (vecs [][]float32, why string, gone bool) {
	for {
		select {
		case <-r.sig:
			for _, m := range r.take() {
				switch m.Op {
				case "vectors":
					if len(m.Vectors) != n {
						return nil, fmt.Sprintf("rule 5: %d vectors for %d texts", len(m.Vectors), n), false
					}
					for i, v := range m.Vectors {
						if len(v) != dims {
							return nil, fmt.Sprintf("rule 5: vector %d has %d numbers, the service registered dims %d", i, len(v), dims), false
						}
					}
					return m.Vectors, "", false
				case "error":
					if m.Message == "" {
						m.Message = "the embedding service reported an error"
					}
					return nil, m.Message, false
				}
			}
		case <-ctx.Done():
			return nil, "", true
		case <-conn.dead:
			select {
			case <-r.sig:
				r.sig <- struct{}{}
			case <-time.After(time.Second):
				return nil, "the embedding service went away during the run", false
			}
		}
	}
}
