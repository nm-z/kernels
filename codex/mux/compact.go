package mux

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"
)

// Compaction (PROTOCOL.md): a request whose input ends in {"type":"compaction_trigger"} is answered with
// one compaction item holding a summary of the conversation; later requests expand that item back into
// the summary, as a user message at its place.

const summaryPrefix = "Summary of the earlier conversation: "

// compactInstruction is the last user message of a summarization prompt.
const compactInstruction = "You are about to run out of room in this conversation, so write a summary of it to continue from. " +
	"The summary replaces the conversation: it must carry everything needed to keep working. Include what the user asked for " +
	"and any constraints or preferences they gave; what has been done so far (files read or changed, commands run, results, " +
	"decisions and why); what is in progress and what remains, in order; and the names, paths, identifiers and values that " +
	"will be needed. Write plain prose and short lists, with no preamble. Do not call tools. Output only the summary."

type compactionPayload struct {
	V       int    `json:"v"`
	By      string `json:"by"`
	Model   string `json:"model"`
	Created string `json:"created"`
	Summary string `json:"summary"`
}

// encodeCompaction is the opaque encrypted_content of a compaction item: base64 of a small JSON.
func encodeCompaction(model, summary string) string {
	b, _ := json.Marshal(compactionPayload{V: 1, By: "mux", Model: model, Created: time.Now().UTC().Format(time.RFC3339), Summary: summary})
	return base64.StdEncoding.EncodeToString(b)
}

func decodeCompaction(enc string) (string, error) {
	b, err := base64.StdEncoding.DecodeString(enc)
	if err != nil {
		return "", errors.New("compaction encrypted_content is not base64")
	}
	var p compactionPayload
	if err := json.Unmarshal(b, &p); err != nil || p.By != "mux" || p.V != 1 {
		return "", errors.New("compaction encrypted_content is not a mux summary")
	}
	return p.Summary, nil
}

// compactReserve is the room left for the summary itself.
func compactReserve(ctx int) int {
	r := ctx / 8
	if r > 8192 {
		r = 8192
	}
	if r < 64 {
		r = 64
	}
	return r
}

// compactMessages is the conversation with its oldest turns dropped (keep: the first user turn to keep,
// an index into turns; 0 keeps everything) and the summarization instruction as the last user message.
type compactTurns struct {
	head   []chatMsg // the system message, if any
	body   []chatMsg
	starts []int // indices into body where a user turn begins
}

func splitTurns(msgs []chatMsg) compactTurns {
	var t compactTurns
	t.body = msgs
	if len(msgs) > 0 && msgs[0].Role == "system" {
		t.head, t.body = msgs[:1], msgs[1:]
	}
	for i, m := range t.body {
		if m.Role == "user" {
			t.starts = append(t.starts, i)
		}
	}
	return t
}

// maxKeep is the largest number of leading turns that may be dropped (the latest turn stays).
func (t compactTurns) maxDrop() int {
	if len(t.starts) <= 1 {
		return 0
	}
	return len(t.starts) - 1
}

// messages with the first drop user turns (and what follows them up to the next) removed.
func (t compactTurns) messages(drop int) []chatMsg {
	body := t.body
	if drop > 0 {
		body = t.body[t.starts[drop]:]
	}
	out := make([]chatMsg, 0, len(t.head)+len(body)+1)
	out = append(out, t.head...)
	out = append(out, body...)
	out = append(out, chatMsg{Role: "user", Content: compactInstruction})
	return out
}

// fitDrop finds the fewest oldest turns to drop so that fits(drop) holds; fits is monotonic in drop.
func (t compactTurns) fitDrop(fits func(msgs []chatMsg) bool) (int, bool) {
	hi := t.maxDrop()
	if fits(t.messages(0)) {
		return 0, true
	}
	if hi == 0 || !fits(t.messages(hi)) {
		return 0, false
	}
	i := sort.Search(hi+1, func(k int) bool { return k > 0 && fits(t.messages(k)) })
	return i, true
}

// fitCompaction renders the summarization prompt for an engine, dropping the oldest turns first until it
// fits the window with room left for the summary.
func fitCompaction(m Model, pr *prepared, ctx int) (prompt string, ids []uint32, dropped int, aerr *apiError) {
	turns := splitTurns(pr.messages)
	reserve := compactReserve(ctx)
	var rerr error
	render := func(msgs []chatMsg) (string, []uint32) {
		v := pr.varsFor(msgs, nil)
		if pr.effort == "" {
			v["enable_thinking"] = false // a summary needs no deliberation unless the client asked for it
		}
		p, err := m.Render(v)
		if err != nil {
			rerr = err
			return "", nil
		}
		ids, err := m.Encode(p)
		if err != nil {
			rerr = err
			return "", nil
		}
		return p, ids
	}
	fits := func(msgs []chatMsg) bool {
		_, ids := render(msgs)
		return rerr == nil && len(ids)+reserve <= ctx
	}
	k, ok := turns.fitDrop(fits)
	if rerr != nil {
		return "", nil, 0, badRequest("chat_template_error", "input", "the chat template rejected the input: %v", rerr)
	}
	if !ok {
		return "", nil, 0, badRequest("context_length_exceeded", "input",
			"the conversation cannot be summarized: even its latest turn and the summary (%d tokens reserved) do not fit the %d-token window", reserve, ctx)
	}
	prompt, ids = render(turns.messages(k))
	return prompt, ids, k, nil
}

// estimateTokens is a rough token count for a provider mux cannot tokenize for (about 4 bytes each).
func estimateTokens(v any) int {
	b, _ := json.Marshal(v)
	return (len(b) + 3) / 4
}

// fitCompactionEstimate is fitCompaction for a shim, by estimate.
func fitCompactionEstimate(pr *prepared, ctx int) (msgs []chatMsg, dropped int, aerr *apiError) {
	turns := splitTurns(pr.messages)
	reserve := compactReserve(ctx)
	k, ok := turns.fitDrop(func(ms []chatMsg) bool { return estimateTokens(ms)+reserve <= ctx })
	if !ok {
		return nil, 0, badRequest("context_length_exceeded", "input",
			"the conversation cannot be summarized: even its latest turn does not fit the %d-token window", ctx)
	}
	return turns.messages(k), k, nil
}

var _ = fmt.Sprintf
