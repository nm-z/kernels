package mux

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	statsEvery   = 250 * time.Millisecond // at most this often a stats event goes to the client …
	statsMaxGap  = 500 * time.Millisecond // … and at least this often while the engine runs
	cancelWait   = 30 * time.Second       // an engine must finish a cancelled run within this
	imageWait    = 2 * time.Minute
	maxBodyBytes = 256 << 20
)

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
	ep := s.eps[probe.Model]
	if ep == nil {
		names := make([]string, 0, len(s.eps))
		for n := range s.eps {
			names = append(names, n)
		}
		sort.Strings(names)
		writeAPIError(w, &apiError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "model_not_found", Param: "model",
			Message: fmt.Sprintf("model %q is not served here (models: %s)", probe.Model, strings.Join(names, ", "))})
		return
	}
	pr, aerr := prepare(body, ep)
	if aerr != nil {
		writeAPIError(w, aerr)
		return
	}
	conn, state := ep.currentConn()
	if conn == nil {
		writeAPIError(w, &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "endpoint_unavailable",
			Message: fmt.Sprintf("model %s is %s", ep.name, state)})
		return
	}
	ep.mu.Lock()
	model := ep.model
	ep.mu.Unlock()
	if model == nil {
		writeAPIError(w, &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "endpoint_unavailable", Message: "model " + ep.name + " has no tokenizer yet"})
		return
	}
	prompt, err := model.Render(pr.templateVars())
	if err != nil {
		writeAPIError(w, badRequest("chat_template_error", "input", "the chat template rejected the input: %v", err))
		return
	}
	ids, err := model.Encode(prompt)
	if err != nil || len(ids) == 0 {
		writeAPIError(w, badRequest("tokenizer_error", "input", "cannot tokenize the prompt: %v", err))
		return
	}
	padID := -1
	if len(pr.images) > 0 {
		padID = model.Find("<|image_pad|>")
		n := 0
		for _, id := range ids {
			if int(id) == padID {
				n++
			}
		}
		if padID < 0 || n != len(pr.images) {
			writeAPIError(w, badRequest("chat_template_error", "input", "the chat template did not place one image token per image"))
			return
		}
	}
	if len(ids)+1 > ep.cfg.Ctx {
		writeAPIError(w, badRequest("context_length_exceeded", "input", "the prompt is %d tokens; the context window of %s is %d", len(ids), ep.name, ep.cfg.Ctx))
		return
	}
	imEnd := model.Find("<|im_end|>")
	if imEnd < 0 {
		writeAPIError(w, &apiError{Status: 500, Type: "server_error", Code: "no_stop_token", Message: "the vocabulary has no <|im_end|>"})
		return
	}
	stop := []uint32{uint32(imEnd)}
	if eos := model.EOS(); eos != 0xffffffff && eos != uint32(imEnd) {
		stop = append(stop, eos)
	}

	var sk sink
	if pr.stream {
		sk = newSSESink(w)
		h := w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
	}
	rb := newRespBuilder(ep.name, model, pr.toolInfo, sk, strings.HasSuffix(prompt, "<think>\n"))
	rb.start()

	res := s.execute(r.Context(), ep, conn, &execReq{
		pr: pr, rb: rb, ids: ids, padID: padID, stop: stop,
	})
	s.finishResponse(w, pr, rb, ep, res)
}

// finishResponse writes the end of the response (the final stats event and response.completed, or the
// failure), or the whole JSON for stream:false.
func (s *Server) finishResponse(w http.ResponseWriter, pr *prepared, rb *respBuilder, ep *Endpoint, res *execResult) {
	if res.cancelled {
		return // the client left; there is nobody to write to
	}
	if res.err != nil {
		if pr.stream {
			rb.failedEvent(res.err.Code, res.err.Message)
		} else {
			writeAPIError(w, res.err)
		}
		return
	}
	rb.finish()
	d := res.done
	gen := 0
	if d.Generated != nil {
		gen = *d.Generated
	}
	prompt := res.promptTokens
	if d.Prompt != nil && *d.Prompt > 0 {
		prompt = *d.Prompt
	}
	cached := 0
	if res.live.prefillTot > 0 && prompt > res.live.prefillTot {
		cached = prompt - res.live.prefillTot
	}
	usage := map[string]any{
		"input_tokens": prompt, "output_tokens": gen, "total_tokens": prompt + gen,
		"input_tokens_details":  map[string]any{"cached_tokens": cached},
		"output_tokens_details": map[string]any{"reasoning_tokens": rb.reasoningTokens},
	}
	// the final measured values, before response.completed
	live := res.live
	live.phase = "decode"
	live.prefillDone, live.prefillTot = prompt-cached, prompt-cached
	if d.PrefillTPS != nil {
		live.prefillTPS = *d.PrefillTPS
	}
	if d.DecodeTPS != nil {
		live.decodeTPS = *d.DecodeTPS
	}
	live.tokens = gen
	rb.stats(&live, d.IdlePct)

	incomplete := res.maxNew > 0 && gen >= res.maxNew && !d.Stopped
	if pr.stream {
		if incomplete {
			rb.incomplete(usage, "max_output_tokens")
		} else {
			rb.completed(usage)
		}
		return
	}
	status := "completed"
	extra := map[string]any{"usage": usage}
	if incomplete {
		status = "incomplete"
		extra["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	am := map[string]any{}
	if d.PrefillTPS != nil {
		am["prefill_tps"] = *d.PrefillTPS
	}
	if d.DecodeTPS != nil {
		am["decode_tps"] = *d.DecodeTPS
	}
	if d.IdlePct != nil {
		am["idle_pct"] = *d.IdlePct
	}
	extra["ai_mux"] = am
	writeJSON(w, http.StatusOK, rb.responseObj(status, extra))
}

// ---- running on the engine -------------------------------------------------------------------------------

type execReq struct {
	pr    *prepared
	rb    *respBuilder
	ids   []uint32
	padID int
	stop  []uint32
}

type execResult struct {
	err          *apiError // the request failed (the response is a failure)
	cancelled    bool      // the client left
	done         *engMsg
	promptTokens int
	maxNew       int
	live         liveStats
}

func (s *Server) execute(ctx context.Context, ep *Endpoint, conn *engineConn, q *execReq) (res *execResult) {
	rb := q.rb
	res = &execResult{}
	unavailable := func(msg string) *execResult {
		res.err = &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "endpoint_unavailable", Message: msg}
		return res
	}

	// wait for a slot
	ep.mu.Lock()
	ep.queued++
	ep.mu.Unlock()
	acquired := false
	ka := time.NewTicker(10 * time.Second)
wait:
	for {
		select {
		case conn.sem <- struct{}{}:
			acquired = true
			break wait
		case <-ctx.Done():
			break wait
		case <-conn.dead:
			break wait
		case <-ka.C:
			rb.keepalive()
		}
	}
	ka.Stop()
	ep.mu.Lock()
	ep.queued--
	ep.mu.Unlock()
	if !acquired {
		if ctx.Err() != nil {
			res.cancelled = true
			return res
		}
		return unavailable(fmt.Sprintf("model %s went down", ep.name))
	}
	defer func() { <-conn.sem }()

	ep.runStarted()
	var last *LastRun
	defer func() { ep.runEnded(last) }()

	ids := q.ids
	var images []map[string]any
	if len(q.pr.images) > 0 {
		var err *apiError
		ids, images, err = s.encodeImages(ctx, conn, q)
		if err != nil {
			if ctx.Err() != nil {
				res.cancelled = true
				return res
			}
			res.err = err
			return res
		}
	}
	np := len(ids)
	res.promptTokens = np
	maxNew := ep.cfg.Ctx - np
	if conn.maxNew > 0 && conn.maxNew < maxNew {
		maxNew = conn.maxNew
	}
	if maxNew < 1 {
		res.err = badRequest("context_length_exceeded", "input", "the prompt is %d tokens; the context window of %s is %d", np, ep.name, ep.cfg.Ctx)
		return res
	}
	res.maxNew = maxNew

	id := conn.newRunID()
	r := newRun(id)
	conn.addRun(r)
	defer conn.removeRun(id)
	if images == nil {
		images = []map[string]any{}
	}
	if err := conn.send(map[string]any{"op": "run", "id": id, "ids": ids, "max_new": maxNew, "stop": q.stop, "images": images}); err != nil {
		return unavailable(fmt.Sprintf("model %s: cannot reach the engine: %v", ep.name, err))
	}
	s.hub.Publish("run.start", map[string]any{"endpoint": ep.name, "run": string(id), "response": rb.id, "prompt": np, "max_new": maxNew})

	live := liveStats{phase: "prefill", prefillTot: np, ctx: ep.cfg.Ctx}
	rb.stats(&live, nil)
	lastStats := time.Now()
	dirty := false
	sawPrefill, sawStats := false, false
	cancelSent := false
	var cancelDeadline <-chan time.Time
	ctxDone := ctx.Done()
	tick := time.NewTicker(statsEvery)
	defer tick.Stop()
	var lastTokPub time.Time
	end := func(status, reason string, d *engMsg) {
		f := map[string]any{"endpoint": ep.name, "run": string(id), "response": rb.id, "status": status,
			"prompt": np, "generated": live.tokens}
		if reason != "" {
			f["reason"] = reason
		}
		if d != nil {
			if d.Generated != nil {
				f["generated"] = *d.Generated
			}
			if d.PrefillTPS != nil {
				f["prefill_tps"] = *d.PrefillTPS
			}
			if d.DecodeTPS != nil {
				f["decode_tps"] = *d.DecodeTPS
			}
			if d.IdlePct != nil {
				f["idle_pct"] = *d.IdlePct
			}
			f["stopped"] = d.Stopped
		}
		s.hub.Publish("run.end", f)
	}

	for {
		select {
		case <-r.sig:
			for _, m := range r.take() {
				switch m.Op {
				case "prefill":
					sawPrefill = true
					if m.Done != nil {
						live.prefillDone = *m.Done
					}
					if m.Total != nil {
						live.prefillTot = *m.Total
					}
					dirty = true
					s.hub.Publish("run.prefill", map[string]any{"endpoint": ep.name, "run": string(id), "done": live.prefillDone, "total": live.prefillTot})
				case "stats":
					sawStats = true
					if m.PrefillTPS != nil {
						live.prefillTPS = *m.PrefillTPS
					}
					if m.DecodeTPS != nil {
						live.decodeTPS = *m.DecodeTPS
					}
					dirty = true
					s.hub.Publish("run.stats", map[string]any{"endpoint": ep.name, "run": string(id), "prefill_tps": live.prefillTPS, "decode_tps": live.decodeTPS})
				case "tokens":
					live.phase = "decode"
					if live.prefillDone < live.prefillTot {
						live.prefillDone = live.prefillTot
					}
					for _, t := range m.IDs {
						rb.feed(t)
					}
					live.tokens += len(m.IDs)
					dirty = true
					if time.Since(lastTokPub) >= 100*time.Millisecond {
						lastTokPub = time.Now()
						s.hub.Publish("run.tokens", map[string]any{"endpoint": ep.name, "run": string(id), "tokens": live.tokens})
					}
				case "error":
					msg := m.Message
					if msg == "" {
						msg = "the engine reported an error"
					}
					if cancelSent && !m.lost {
						res.cancelled = true
						end("cancelled", msg, nil)
						return res
					}
					if m.lost && ctx.Err() != nil {
						res.cancelled = true
						end("cancelled", msg, nil)
						return res
					}
					res.err = &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "engine_error", Message: fmt.Sprintf("model %s: %s", ep.name, msg)}
					end("failed", msg, nil)
					return res
				case "done":
					res.done = m
					res.live = live
					if why := checkDone(m, sawPrefill, sawStats, cancelSent); why != "" {
						reason := fmt.Sprintf("engine broke rule 5 (it must stream): %s", why)
						s.logf("endpoint %s: %s", ep.name, reason)
						ep.fail(reason)
						end("violation", reason, m)
						if cancelSent {
							res.cancelled = true
						}
						if m.PrefillTPS == nil || m.DecodeTPS == nil || m.IdlePct == nil {
							zero := 0.0
							if m.PrefillTPS == nil {
								m.PrefillTPS = &zero
							}
							if m.DecodeTPS == nil {
								m.DecodeTPS = &zero
							}
						}
						return res
					}
					last = &LastRun{PrefillTPS: *m.PrefillTPS, DecodeTPS: *m.DecodeTPS, IdlePct: *m.IdlePct, At: time.Now()}
					if cancelSent {
						res.cancelled = true
						end("cancelled", "", m)
						return res
					}
					end("done", "", m)
					return res
				}
			}
		case <-tick.C:
			if rb.clientGone() && !cancelSent {
				ctxDone = nil
				cancelSent = true
				conn.send(map[string]any{"op": "cancel", "id": id})
				cancelDeadline = time.After(cancelWait)
			}
			if dirty || time.Since(lastStats) >= statsMaxGap-statsEvery/2 { // the tick is coarse: send before the gap can pass 500 ms
				rb.stats(&live, nil)
				lastStats, dirty = time.Now(), false
			}
			rb.keepalive()
		case <-ctxDone:
			ctxDone = nil
			cancelSent = true
			conn.send(map[string]any{"op": "cancel", "id": id})
			cancelDeadline = time.After(cancelWait)
		case <-cancelDeadline:
			reason := fmt.Sprintf("engine did not finish a cancelled run within %s", cancelWait)
			ep.fail(reason)
			res.cancelled = true
			end("violation", reason, nil)
			return res
		case <-conn.dead:
			// the socket's lines are all queued (the reader pushes an error last): drain them first
			select {
			case <-r.sig:
				r.sig <- struct{}{}
			case <-time.After(time.Second):
				res.err = &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "engine_error", Message: fmt.Sprintf("model %s went away during the run", ep.name)}
				end("failed", "engine connection closed", nil)
				return res
			}
		}
	}
}

// checkDone is rule 5 at the end of a run: the engine streamed prefill progress and decode stats and
// measured its speeds. A cancelled run may have ended before it had anything to report.
func checkDone(m *engMsg, sawPrefill, sawStats, cancelled bool) string {
	var miss []string
	if m.PrefillTPS == nil {
		miss = append(miss, "done has no prefill_tps")
	}
	if m.DecodeTPS == nil {
		miss = append(miss, "done has no decode_tps")
	}
	if m.IdlePct == nil {
		miss = append(miss, "done has no idle_pct")
	}
	gen := 0
	if m.Generated != nil {
		gen = *m.Generated
	}
	if !cancelled {
		if !sawPrefill {
			miss = append(miss, "no prefill line during the prompt")
		}
		if gen > 1 && !sawStats {
			miss = append(miss, "no stats line during generation")
		}
	}
	return strings.Join(miss, "; ")
}

// encodeImages has the engine encode every image and expands each <|image_pad|> in the prompt to the
// positions the image takes. It returns the new prompt and the run's images ([{img, at}]).
func (s *Server) encodeImages(ctx context.Context, conn *engineConn, q *execReq) ([]uint32, []map[string]any, *apiError) {
	if !conn.vision {
		return nil, nil, badRequest("images_not_supported", "input", "the engine has no vision encoder")
	}
	tokens := map[string]int{}
	for _, img := range q.pr.images {
		if _, ok := tokens[img.hash]; ok {
			continue
		}
		ch := make(chan int, 1)
		conn.mu.Lock()
		conn.imgWait[img.hash] = ch
		conn.mu.Unlock()
		err := conn.send(map[string]any{"op": "image", "img": img.hash, "bytes": base64.StdEncoding.EncodeToString(img.bytes)})
		var n int
		if err == nil {
			select {
			case n = <-ch:
			case <-ctx.Done():
				err = ctx.Err()
			case <-conn.dead:
				err = errors.New("the engine went away")
			case <-time.After(imageWait):
				err = errors.New("the engine did not encode the image in time")
			}
		}
		conn.mu.Lock()
		delete(conn.imgWait, img.hash)
		conn.mu.Unlock()
		if err != nil {
			return nil, nil, &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "engine_error", Message: "image: " + err.Error()}
		}
		if n < 1 {
			return nil, nil, &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "engine_error", Message: "the engine reported no positions for an image"}
		}
		tokens[img.hash] = n
	}
	var out []uint32
	var images []map[string]any
	k := 0
	for _, id := range q.ids {
		if int(id) == q.padID && k < len(q.pr.images) {
			img := q.pr.images[k]
			k++
			images = append(images, map[string]any{"img": img.hash, "at": len(out)})
			for i := 0; i < tokens[img.hash]; i++ {
				out = append(out, id)
			}
			continue
		}
		out = append(out, id)
	}
	return out, images, nil
}
