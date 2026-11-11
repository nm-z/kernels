package mux

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ---- sinks ---------------------------------------------------------------------------------------------

// sink is where a response's events go: an SSE stream, or nowhere (stream:false).
type sink interface {
	write(typ string, data []byte)
	comment(text string)
	sinceWrite() time.Duration
	failed() bool
}

type sseSink struct {
	w    http.ResponseWriter
	fl   http.Flusher
	rc   *http.ResponseController
	last time.Time
	err  error
}

func newSSESink(w http.ResponseWriter) *sseSink {
	fl, _ := w.(http.Flusher)
	return &sseSink{w: w, fl: fl, rc: http.NewResponseController(w), last: time.Now()}
}

func (s *sseSink) put(format string, args ...any) {
	if s.err != nil {
		return
	}
	// a client that stops reading must not hold the engine: a write that blocks 30 s fails the stream
	s.rc.SetWriteDeadline(time.Now().Add(30 * time.Second))
	if _, err := fmt.Fprintf(s.w, format, args...); err != nil {
		s.err = err
		return
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	s.last = time.Now()
}

func (s *sseSink) write(typ string, data []byte) { s.put("event: %s\ndata: %s\n\n", typ, data) }
func (s *sseSink) comment(text string)           { s.put(": %s\n\n", text) }
func (s *sseSink) sinceWrite() time.Duration     { return time.Since(s.last) }
func (s *sseSink) failed() bool                  { return s.err != nil }

// ---- the response being built -----------------------------------------------------------------------------

const (
	modeThink = iota // reasoning, until </think>
	modeText         // the answer
	modeTool         // inside <tool_call>
)

// respBuilder turns the engine's token ids into a Responses API stream: it detokenizes, splits the
// reasoning from the answer, holds tool-call XML back from the text and parses it into calls.
type respBuilder struct {
	id            string
	served        string // the model named to the client by servedBy
	servedOptions string // the options named with it
	model         string
	created       int64
	m             Model
	tools         map[string]*toolInfo
	sink          sink
	seq           int

	thinkEnd, thinkStart int

	outIdx int
	items  []map[string]any // finished output items, in order

	mode int
	buf  string // decoded text not yet handled
	pend []byte // an incomplete UTF-8 tail
	any  bool   // any output yet

	reasoningTokens int

	chatTimings *chatTimings // the engine's final numbers, for a stream:false chat completion

	rTrim, aTrim trimmer
	rOpen        *openItem // the reasoning item being streamed
	aOpen        *openItem // the message item being streamed

	compact bool            // the answer is a summary for a compaction item, not output
	summary strings.Builder // what the model wrote as its summary
}

type openItem struct {
	id   string
	idx  int
	text strings.Builder
}

type trimmer struct {
	started bool
	ws      string
}

// push drops the whitespace that opens a part and holds the whitespace that ends what has come so far
// (it is only text if more text follows).
func (t *trimmer) push(s string) string {
	if !t.started {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" {
			return ""
		}
		t.started = true
	}
	body := strings.TrimRight(s, " \t\r\n")
	trail := s[len(body):]
	if body == "" {
		t.ws += trail
		return ""
	}
	out := t.ws + body
	t.ws = trail
	return out
}

func newRespBuilder(model string, m Model, tools map[string]*toolInfo, sk sink, inThink bool) *respBuilder {
	b := &respBuilder{
		id: newID("resp_"), model: model, created: time.Now().Unix(), m: m, tools: tools, sink: sk,
		thinkEnd: m.Find("</think>"), thinkStart: m.Find("<think>"),
	}
	if inThink {
		b.mode = modeThink
	} else {
		b.mode = modeText
	}
	return b
}

func (b *respBuilder) emit(typ string, p map[string]any) {
	if b.sink == nil {
		return
	}
	p["type"] = typ
	p["sequence_number"] = b.seq
	b.seq++
	data, err := json.Marshal(p)
	if err != nil {
		return
	}
	b.sink.write(typ, data)
}

func (b *respBuilder) keepalive() {
	if b.sink != nil && b.sink.sinceWrite() >= 10*time.Second {
		b.sink.comment("keepalive")
	}
}

func (b *respBuilder) clientGone() bool { return b.sink != nil && b.sink.failed() }

func (b *respBuilder) responseObj(status string, extra map[string]any) map[string]any {
	out := make([]any, len(b.items))
	for i, it := range b.items {
		out[i] = it
	}
	r := map[string]any{
		"id": b.id, "object": "response", "created_at": b.created, "status": status, "model": b.model,
		"output": out, "error": nil, "incomplete_details": nil,
	}
	for k, v := range extra {
		r[k] = v
	}
	return r
}

// servedBy tells the client which model answers when it is not the one it asked for (an auto
// shim's route): an in-progress event whose response carries the name as the `openai-model`
// header, the place a Responses client reads the serving model from. Once a name.
func (b *respBuilder) servedBy(model, options string) {
	if model == "" || (model == b.served && options == b.servedOptions) {
		return
	}
	b.served = model
	b.servedOptions = options
	headers := map[string]any{"openai-model": model}
	if options != "" {
		headers["x-mux-options"] = options
	}
	b.emit("response.in_progress", map[string]any{"response": b.responseObj("in_progress",
		map[string]any{"headers": headers})})
}

// answeredBy is the model named as serving, or the one asked for when none was named.
func (b *respBuilder) answeredBy() string {
	if b.served != "" {
		return b.served
	}
	return b.model
}

func (b *respBuilder) start() {
	b.emit("response.created", map[string]any{"response": b.responseObj("in_progress", nil)})
	b.emit("response.in_progress", map[string]any{"response": b.responseObj("in_progress", nil)})
}

// feed takes one generated token.
func (b *respBuilder) feed(id uint32) {
	if b.mode == modeThink {
		b.reasoningTokens++
	}
	if int(id) == b.thinkEnd && b.thinkEnd >= 0 && b.mode == modeThink {
		b.flushPend()
		b.drain(true)
		b.endThink()
		return
	}
	if int(id) == b.thinkStart && b.thinkStart >= 0 && b.mode == modeText && !b.any {
		b.mode = modeThink
		return
	}
	if b.m.Hidden(id) {
		return // control and unknown tokens (<|im_end|>) are not text; <tool_call> is
	}
	b.pend = append(b.pend, b.m.Piece(id)...)
	// an incomplete UTF-8 character at the end waits for its next bytes
	keep := len(b.pend)
	for back := 1; back <= 3 && back <= len(b.pend); back++ {
		c := b.pend[len(b.pend)-back]
		if c&0xc0 == 0x80 {
			continue
		}
		need := 1
		switch {
		case c >= 0xf0:
			need = 4
		case c >= 0xe0:
			need = 3
		case c >= 0xc0:
			need = 2
		}
		if need > back {
			keep = len(b.pend) - back
		}
		break
	}
	if keep > 0 {
		b.buf += string(b.pend[:keep])
		b.pend = append(b.pend[:0], b.pend[keep:]...)
		b.drain(false)
	}
}

func (b *respBuilder) flushPend() {
	if len(b.pend) > 0 {
		b.buf += strings.ToValidUTF8(string(b.pend), "�")
		b.pend = b.pend[:0]
	}
}

// partial is the length of the longest suffix of s that is a proper prefix of marker.
func partial(s, marker string) int {
	n := len(marker) - 1
	if n > len(s) {
		n = len(s)
	}
	for ; n > 0; n-- {
		if strings.HasSuffix(s, marker[:n]) {
			return n
		}
	}
	return 0
}

const (
	thinkClose = "</think>"
	callOpen   = "<tool_call>"
	callClose  = "</tool_call>"
)

// drain handles the decoded text in buf. final: no more text comes, so nothing is held back.
func (b *respBuilder) drain(final bool) {
	for {
		switch b.mode {
		case modeThink:
			if i := strings.Index(b.buf, thinkClose); i >= 0 {
				b.reasoning(b.buf[:i])
				b.buf = b.buf[i+len(thinkClose):]
				b.endThink()
				continue
			}
			hold := 0
			if !final {
				hold = partial(b.buf, thinkClose)
			}
			b.reasoning(b.buf[:len(b.buf)-hold])
			b.buf = b.buf[len(b.buf)-hold:]
			return
		case modeText:
			if i := strings.Index(b.buf, callOpen); i >= 0 {
				b.text(b.buf[:i])
				b.buf = b.buf[i+len(callOpen):]
				b.closeMessage()
				b.mode = modeTool
				continue
			}
			hold := 0
			if !final {
				hold = partial(b.buf, callOpen)
			}
			b.text(b.buf[:len(b.buf)-hold])
			b.buf = b.buf[len(b.buf)-hold:]
			return
		case modeTool:
			if i := strings.Index(b.buf, callClose); i >= 0 {
				body := b.buf[:i]
				b.buf = b.buf[i+len(callClose):]
				b.mode = modeText
				if !b.toolCall(body) {
					b.text(callOpen + body + callClose)
				}
				continue
			}
			return
		}
	}
}

func (b *respBuilder) endThink() {
	b.closeReasoning()
	b.mode = modeText
}

// ---- reasoning item ---------------------------------------------------------------------------------------

func (b *respBuilder) reasoning(s string) {
	if b.compact {
		return // a summary is the answer only
	}
	s = b.rTrim.push(s)
	if s == "" {
		return
	}
	b.any = true
	if b.rOpen == nil {
		b.rOpen = &openItem{id: newID("rs_"), idx: b.outIdx}
		b.outIdx++
		b.emit("response.output_item.added", map[string]any{"output_index": b.rOpen.idx, "item": map[string]any{
			"id": b.rOpen.id, "type": "reasoning", "summary": []any{}, "content": []any{}}})
	}
	b.rOpen.text.WriteString(s)
	b.emit("response.reasoning_text.delta", map[string]any{"item_id": b.rOpen.id, "output_index": b.rOpen.idx, "content_index": 0, "delta": s})
}

func (b *respBuilder) closeReasoning() {
	o := b.rOpen
	if o == nil {
		b.rTrim = trimmer{}
		return
	}
	b.rOpen = nil
	b.rTrim = trimmer{}
	text := o.text.String()
	item := map[string]any{"id": o.id, "type": "reasoning", "summary": []any{},
		"content": []any{map[string]any{"type": "reasoning_text", "text": text}}}
	b.emit("response.reasoning_text.done", map[string]any{"item_id": o.id, "output_index": o.idx, "content_index": 0, "text": text})
	b.emit("response.output_item.done", map[string]any{"output_index": o.idx, "item": item})
	b.items = append(b.items, item)
}

// ---- message item -------------------------------------------------------------------------------------------

func (b *respBuilder) text(s string) {
	s = b.aTrim.push(s)
	if s == "" {
		return
	}
	b.any = true
	if b.compact {
		b.summary.WriteString(s)
		return
	}
	if b.aOpen == nil {
		b.aOpen = &openItem{id: newID("msg_"), idx: b.outIdx}
		b.outIdx++
		b.emit("response.output_item.added", map[string]any{"output_index": b.aOpen.idx, "item": map[string]any{
			"id": b.aOpen.id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}})
		b.emit("response.content_part.added", map[string]any{"item_id": b.aOpen.id, "output_index": b.aOpen.idx, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}})
	}
	b.aOpen.text.WriteString(s)
	b.emit("response.output_text.delta", map[string]any{"item_id": b.aOpen.id, "output_index": b.aOpen.idx, "content_index": 0, "delta": s})
}

func (b *respBuilder) closeMessage() {
	o := b.aOpen
	b.aTrim = trimmer{}
	if o == nil {
		return
	}
	b.aOpen = nil
	text := o.text.String()
	part := map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
	item := map[string]any{"id": o.id, "type": "message", "status": "completed", "role": "assistant", "content": []any{part}}
	b.emit("response.output_text.done", map[string]any{"item_id": o.id, "output_index": o.idx, "content_index": 0, "text": text})
	b.emit("response.content_part.done", map[string]any{"item_id": o.id, "output_index": o.idx, "content_index": 0, "part": part})
	b.emit("response.output_item.done", map[string]any{"output_index": o.idx, "item": item})
	b.items = append(b.items, item)
}

// ---- tool calls --------------------------------------------------------------------------------------------

// toolCall turns the inside of a <tool_call> into a function_call or custom_tool_call item.
func (b *respBuilder) toolCall(body string) bool {
	name, params, ok := parseFunctionBlock(body)
	if !ok {
		return false
	}
	if b.compact {
		return true // a summary calls nothing
	}
	info := b.tools[name]
	var types map[string][]string
	if info != nil {
		types = info.params
	}
	input := ""
	if info != nil && info.custom {
		for _, p := range params {
			if p.name == "input" {
				input = p.value
			}
		}
		if input == "" && len(params) == 1 && params[0].name != "input" {
			input = params[0].value
		}
	}
	b.callItem(info, name, newID("call_"), string(argsObject(params, types)), input)
	return true
}

// shimCall is a shim's tool_call: arguments is JSON text.
func (b *respBuilder) shimCall(callID, name, arguments string) {
	if b.compact {
		return
	}
	if callID == "" {
		callID = newID("call_")
	}
	info := b.tools[name]
	input := ""
	if info != nil && info.custom {
		var m map[string]json.RawMessage
		if json.Unmarshal([]byte(arguments), &m) == nil {
			var s string
			if v, ok := m["input"]; ok && json.Unmarshal(v, &s) == nil {
				input = s
			} else if len(m) == 1 {
				for _, v := range m {
					json.Unmarshal(v, &s)
					input = s
				}
			}
		} else {
			input = arguments
		}
	}
	b.callItem(info, name, callID, arguments, input)
}

// callItem emits a finished call: a custom_tool_call (input) for a freeform tool, else a function_call.
func (b *respBuilder) callItem(info *toolInfo, name, callID, args, input string) {
	b.any = true
	b.closeReasoning() // a call after unfinished reasoning (no </think>) ends it
	b.closeMessage()
	idx := b.outIdx
	b.outIdx++
	if info != nil && info.custom {
		id := newID("ctc_")
		item := map[string]any{"id": id, "type": "custom_tool_call", "status": "completed", "call_id": callID, "name": info.name, "input": input}
		added := map[string]any{"id": id, "type": "custom_tool_call", "status": "in_progress", "call_id": callID, "name": info.name, "input": ""}
		if info.namespace != "" {
			item["namespace"], added["namespace"] = info.namespace, info.namespace
		}
		b.emit("response.output_item.added", map[string]any{"output_index": idx, "item": added})
		b.emit("response.custom_tool_call_input.delta", map[string]any{"item_id": id, "call_id": callID, "output_index": idx, "delta": input})
		b.emit("response.custom_tool_call_input.done", map[string]any{"item_id": id, "call_id": callID, "output_index": idx, "input": input})
		b.emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
		b.items = append(b.items, item)
		return
	}
	vis, ns := name, ""
	if info != nil {
		vis, ns = info.name, info.namespace
	}
	id := newID("fc_")
	item := map[string]any{"id": id, "type": "function_call", "status": "completed", "call_id": callID, "name": vis, "arguments": args}
	added := map[string]any{"id": id, "type": "function_call", "status": "in_progress", "call_id": callID, "name": vis, "arguments": ""}
	if ns != "" {
		item["namespace"], added["namespace"] = ns, ns
	}
	b.emit("response.output_item.added", map[string]any{"output_index": idx, "item": added})
	b.emit("response.function_call_arguments.delta", map[string]any{"item_id": id, "output_index": idx, "delta": args})
	b.emit("response.function_call_arguments.done", map[string]any{"item_id": id, "output_index": idx, "arguments": args})
	b.emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
	b.items = append(b.items, item)
}

// compaction emits the compaction item that answers a compaction_trigger.
func (b *respBuilder) compactionItem(summary string) {
	idx := b.outIdx
	b.outIdx++
	id := newID("cmp_")
	b.emit("response.output_item.added", map[string]any{"output_index": idx, "item": map[string]any{"id": id, "type": "compaction", "encrypted_content": ""}})
	item := map[string]any{"id": id, "type": "compaction", "encrypted_content": encodeCompaction(b.model, summary)}
	b.emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
	b.items = append(b.items, item)
}

// finish ends the text: whatever is held back is handled, open items are closed. A tool call cut off
// before </tool_call> is taken if it parses, else it is shown as text.
func (b *respBuilder) finish() {
	b.flushPend()
	if b.mode == modeTool {
		body := b.buf
		b.buf = ""
		b.mode = modeText
		if !b.toolCall(body) {
			b.text(callOpen + body)
		}
	}
	b.drain(true)
	b.closeReasoning()
	b.closeMessage()
}

// finishDirect ends a shim's output (nothing is held back; there is nothing to parse).
func (b *respBuilder) finishDirect() {
	b.closeReasoning()
	b.closeMessage()
}

// ---- stats and the end ------------------------------------------------------------------------------------

type statsEvent struct {
	Type    string `json:"type"`
	Phase   string `json:"phase"`
	Prefill struct {
		Done  int     `json:"done"`
		Total int     `json:"total"`
		TPS   float64 `json:"tps"`
	} `json:"prefill"`
	Decode struct {
		Tokens int     `json:"tokens"`
		TPS    float64 `json:"tps"`
	} `json:"decode"`
	IdlePct *float64 `json:"idle_pct"` // the run's idle % so far; null until the engine has one
	Ctx     int      `json:"ctx"`
}

// liveStats is what the engine has reported so far about the running request.
type liveStats struct {
	phase                   string
	prefillDone, prefillTot int
	prefillTPS              float64
	tokens                  int
	decodeTPS               float64
	idle                    *float64
	ctx                     int
}

func (b *respBuilder) stats(s *liveStats) {
	if b.sink == nil {
		return
	}
	var e statsEvent
	e.Type = "response.ai_mux.stats"
	e.Phase = s.phase
	e.Prefill.Done, e.Prefill.Total, e.Prefill.TPS = s.prefillDone, s.prefillTot, s.prefillTPS
	e.Decode.Tokens, e.Decode.TPS = s.tokens, s.decodeTPS
	e.IdlePct = s.idle
	e.Ctx = s.ctx
	data, err := json.Marshal(e)
	if err != nil {
		return
	}
	b.sink.write(e.Type, data)
}

func (b *respBuilder) completed(usage map[string]any) {
	b.emit("response.completed", map[string]any{"response": b.responseObj("completed", map[string]any{"usage": usageOrNull(usage)})})
}

func (b *respBuilder) incomplete(usage map[string]any, reason string) {
	b.emit("response.incomplete", map[string]any{"response": b.responseObj("incomplete", map[string]any{
		"usage": usageOrNull(usage), "incomplete_details": map[string]any{"reason": reason}})})
}

func (b *respBuilder) failedEvent(code, message string) {
	b.emit("response.failed", map[string]any{"response": b.responseObj("failed", map[string]any{
		"error": map[string]any{"code": code, "message": message}})})
}
