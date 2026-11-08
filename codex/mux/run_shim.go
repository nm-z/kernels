package mux

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"time"
)

// shimStatsEvent is response.ai_mux.stats for a shim.
type shimStatsEvent struct {
	Type  string `json:"type"`
	Phase string `json:"phase"` // "route" on every route change, "output" at the first token
	Route *struct {
		Trying []string `json:"trying"`
		Failed []string `json:"failed"`
	} `json:"route,omitempty"`
	TTFTMS *float64 `json:"ttft_ms,omitempty"`
	Ctx    int      `json:"ctx"`
}

func (b *respBuilder) shimStats(phase string, auto bool, trying, failed []string, ttft *float64, ctx int) {
	if b.sink == nil {
		return
	}
	e := shimStatsEvent{Type: "response.ai_mux.stats", Phase: phase, TTFTMS: ttft, Ctx: ctx}
	if auto {
		e.Route = &struct {
			Trying []string `json:"trying"`
			Failed []string `json:"failed"`
		}{Trying: nonNil(trying), Failed: nonNil(failed)}
	}
	if data, err := jsonMarshal(e); err == nil {
		b.sink.write(e.Type, data)
	}
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// respondShim answers a /v1/responses request on a shim service: the shim gets the conversation in chat
// form and streams text, reasoning and tool calls back.
func (s *Server) respondShim(w http.ResponseWriter, r *http.Request, svc *Service, conn *svcConn, pr *prepared) {
	msgs := pr.messages
	tools := pr.tools
	dropped := 0
	if pr.compact {
		var aerr *apiError
		msgs, dropped, aerr = fitCompactionEstimate(pr, svc.cfg.Ctx)
		if aerr != nil {
			writeAPIError(w, aerr)
			return
		}
		tools = nil
	}
	if len(pr.images) > 0 && !conn.vision {
		writeAPIError(w, badRequest("images_not_supported", "input", "the shim has no vision"))
		return
	}
	sk := openStream(w, pr)
	rb := newRespBuilder(svc.name, noModel{}, pr.toolInfo, sk, false)
	rb.compact = pr.compact
	rb.start()
	res := s.executeShim(r.Context(), svc, conn, &shimReq{pr: pr, rb: rb, msgs: msgs, tools: tools, dropped: dropped})
	s.finishShim(w, pr, rb, res)
}

type shimReq struct {
	pr      *prepared
	rb      *respBuilder
	msgs    []chatMsg
	tools   []any
	dropped int
}

type shimResult struct {
	err       *apiError
	cancelled bool
	done      *msg
	ttftMS    *float64
	trying    []string
	outBytes  int
	inTokens  int // estimated
}

func (s *Server) finishShim(w http.ResponseWriter, pr *prepared, rb *respBuilder, res *shimResult) {
	if res.cancelled {
		return
	}
	if res.err != nil {
		failResponse(w, pr, rb, res.err)
		return
	}
	rb.finishDirect()
	if pr.compact {
		if aerr := settleSummary(rb); aerr != nil {
			failResponse(w, pr, rb, aerr)
			return
		}
	}
	in, out := res.inTokens, (res.outBytes+3)/4
	if d := res.done; d != nil {
		if d.InputTokens != nil {
			in = *d.InputTokens
		}
		if d.OutputTokens != nil {
			out = *d.OutputTokens
		}
	}
	usage := map[string]any{
		"input_tokens": in, "output_tokens": out, "total_tokens": in + out,
		"input_tokens_details":  map[string]any{"cached_tokens": 0},
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
	}
	am := map[string]any{}
	if res.ttftMS != nil {
		am["ttft_ms"] = *res.ttftMS
	}
	if len(res.trying) > 0 {
		am["route"] = res.trying
	}
	completeResponse(w, pr, rb, usage, false, map[string]any{"ai_mux": am})
}

func (s *Server) executeShim(ctx context.Context, svc *Service, conn *svcConn, q *shimReq) (res *shimResult) {
	rb := q.rb
	res = &shimResult{inTokens: estimateTokens(q.msgs)}
	if !acquireSlot(ctx, svc, conn, rb) {
		if ctx.Err() != nil {
			res.cancelled = true
			return res
		}
		res.err = &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "service_unavailable", Message: fmt.Sprintf("model %s went down", svc.name)}
		return res
	}
	defer func() { <-conn.sem }()
	svc.runStarted()
	var last *LastRun
	defer func() { svc.runEnded(last) }()

	id := conn.newRunID()
	r := newRun(id)
	conn.addRun(r)
	defer conn.removeRun(id)
	images := []map[string]any{}
	for _, img := range q.pr.images {
		images = append(images, map[string]any{"img": img.hash, "mime": img.mime, "bytes": base64.StdEncoding.EncodeToString(img.bytes)})
	}
	tools := q.tools
	if tools == nil {
		tools = []any{}
	}
	var effort any
	if q.pr.effort != "" {
		effort = q.pr.effort
	}
	if err := conn.send(map[string]any{"op": "run", "id": id, "messages": q.msgs, "tools": tools, "effort": effort, "images": images}); err != nil {
		res.err = &apiError{Status: http.StatusServiceUnavailable, Type: "server_error", Code: "service_unavailable", Message: fmt.Sprintf("model %s: cannot reach the shim: %v", svc.name, err)}
		return res
	}
	start := map[string]any{"service": svc.name, "run": string(id), "response": rb.id, "messages": len(q.msgs)}
	if q.pr.compact {
		start["compaction"] = map[string]any{"dropped_turns": q.dropped}
	}
	s.hub.Publish("run.start", start)

	var trying, failed []string
	sawRoute, sawTTFT := false, false
	var lastPub time.Time
	var ttft *float64
	cancelSent := false
	var cancelDeadline <-chan time.Time
	ctxDone := ctx.Done()
	tick := time.NewTicker(statsEvery)
	defer tick.Stop()
	end := func(status, reason string, d *msg) {
		f := map[string]any{"service": svc.name, "run": string(id), "response": rb.id, "status": status}
		if reason != "" {
			f["reason"] = reason
		}
		if ttft != nil {
			f["ttft_ms"] = *ttft
		}
		if len(trying) > 0 {
			f["route"] = trying
		}
		if d != nil {
			if d.InputTokens != nil {
				f["input_tokens"] = *d.InputTokens
			}
			if d.OutputTokens != nil {
				f["output_tokens"] = *d.OutputTokens
			}
		}
		s.hub.Publish("run.end", f)
	}
	violation := func(why string) *shimResult {
		reason := "rule 5: " + why
		s.logf("service %s: %s", svc.name, reason)
		svc.fail(reason)
		end("violation", reason, nil)
		if cancelSent || ctx.Err() != nil {
			res.cancelled = true
		} else {
			res.err = &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "service_failed", Message: fmt.Sprintf("model %s was dropped: %s", svc.name, reason)}
		}
		return res
	}
	// before any output the shim must have said how fast it was (and which route it took, if it is auto)
	checkOutput := func(what string) string {
		if !sawTTFT {
			return "a shim must send ttft before its first " + what
		}
		if conn.auto && !sawRoute {
			return "an auto shim must send route before its first " + what
		}
		return ""
	}

	for {
		select {
		case <-r.sig:
			for _, m := range r.take() {
				switch m.Op {
				case "route":
					sawRoute = true
					trying, failed = m.Trying, m.Failed
					s.hub.Publish("run.route", map[string]any{"service": svc.name, "run": string(id), "trying": nonNil(trying), "failed": nonNil(failed)})
					rb.shimStats("route", conn.auto, trying, failed, nil, svc.cfg.Ctx)
				case "ttft":
					if sawTTFT {
						continue
					}
					sawTTFT = true
					v := 0.0
					if m.MS != nil {
						v = *m.MS
					}
					ttft = &v
					res.ttftMS = ttft
					s.hub.Publish("run.ttft", map[string]any{"service": svc.name, "run": string(id), "ms": v})
					rb.shimStats("output", conn.auto, trying, failed, ttft, svc.cfg.Ctx)
				case "delta":
					if why := checkOutput("delta"); why != "" {
						return violation(why)
					}
					switch m.Channel {
					case "reasoning":
						rb.closeMessage()
						rb.reasoning(m.Text)
					case "content":
						rb.closeReasoning()
						rb.text(m.Text)
					default:
						s.logf("service %s: delta on unknown channel %q ignored", svc.name, m.Channel)
						continue
					}
					res.outBytes += len(m.Text)
					if time.Since(lastPub) >= 100*time.Millisecond { // counts, not text, and not on every delta
						lastPub = time.Now()
						s.hub.Publish("run.tokens", map[string]any{"service": svc.name, "run": string(id), "bytes": res.outBytes})
					}
				case "tool_call":
					if why := checkOutput("tool_call"); why != "" {
						return violation(why)
					}
					args := toolArgsText(m.Arguments)
					rb.shimCall(m.CallID, m.Name, args)
					res.outBytes += len(args) + len(m.Name)
				case "error":
					msgText := m.Message
					if msgText == "" {
						msgText = "the shim reported an error"
					}
					if (cancelSent && !m.lost) || (m.lost && ctx.Err() != nil) {
						res.cancelled = true
						end("cancelled", msgText, nil)
						return res
					}
					res.err = &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "provider_error", Message: fmt.Sprintf("model %s: %s", svc.name, msgText)}
					end("failed", msgText, nil)
					return res
				case "done":
					res.done = m
					res.trying = trying
					if conn.auto && !sawRoute {
						// the answer is whole, but the shim broke the rule: drop it after this run
						reason := "rule 5: an auto shim must send route for every run"
						s.logf("service %s: %s", svc.name, reason)
						svc.fail(reason)
						end("violation", reason, m)
						if cancelSent {
							res.cancelled = true
						}
						return res
					}
					last = &LastRun{TTFTMS: ttft, Route: trying, At: time.Now()}
					if m.InputTokens != nil {
						last.InputTokens = *m.InputTokens
					}
					if m.OutputTokens != nil {
						last.OutputTokens = *m.OutputTokens
					}
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
			rb.keepalive()
		case <-ctxDone:
			ctxDone = nil
			cancelSent = true
			conn.send(map[string]any{"op": "cancel", "id": id})
			cancelDeadline = time.After(cancelWait)
		case <-cancelDeadline:
			reason := fmt.Sprintf("rule 5: the shim did not finish a cancelled run within %s", cancelWait)
			svc.fail(reason)
			res.cancelled = true
			end("violation", reason, nil)
			return res
		case <-conn.dead:
			select {
			case <-r.sig:
				r.sig <- struct{}{}
			case <-time.After(time.Second):
				res.err = &apiError{Status: http.StatusBadGateway, Type: "server_error", Code: "provider_error", Message: fmt.Sprintf("model %s went away during the run", svc.name)}
				end("failed", "service connection closed", nil)
				return res
			}
		}
	}
}
