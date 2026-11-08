package mux

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// eventStream reads GET /v1/events in the background.
type eventStream struct {
	ch     chan sseEvent
	cancel context.CancelFunc
}

func openEvents(t *testing.T, base, query string, hdr map[string]string) (*eventStream, int) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"/v1/events"+query, nil)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	es := &eventStream{ch: make(chan sseEvent, 4096), cancel: cancel}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		cancel()
		return es, resp.StatusCode
	}
	go func() {
		defer resp.Body.Close()
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 1<<20), 16<<20)
		var cur sseEvent
		for sc.Scan() {
			line := sc.Text()
			switch {
			case line == "":
				if cur.Name != "" {
					es.ch <- cur
				}
				cur = sseEvent{}
			case strings.HasPrefix(line, "id: "):
				cur.ID = line[4:]
			case strings.HasPrefix(line, "event: "):
				cur.Name = line[7:]
			case strings.HasPrefix(line, "data: "):
				cur.Data = json.RawMessage(line[6:])
			}
		}
	}()
	t.Cleanup(cancel)
	return es, 200
}

// until collects events until one satisfies stop (or the time runs out).
func (es *eventStream) until(t *testing.T, d time.Duration, stop func(sseEvent) bool) []sseEvent {
	t.Helper()
	var out []sseEvent
	deadline := time.After(d)
	for {
		select {
		case ev := <-es.ch:
			out = append(out, ev)
			if stop(ev) {
				return out
			}
		case <-deadline:
			t.Fatalf("timed out after %s with %d events", d, len(out))
		}
	}
}

func idOf(t *testing.T, ev sseEvent) uint64 {
	t.Helper()
	n, err := strconv.ParseUint(ev.ID, 10, 64)
	if err != nil {
		t.Fatalf("event %s has id %q", ev.Name, ev.ID)
	}
	return n
}

func oneRun(t *testing.T, e *tenv, model string) {
	t.Helper()
	resp := e.post("/v1/responses", `{"model":"`+model+`","stream":true,"input":"hi"}`)
	readSSE(t, resp.Body)
	resp.Body.Close()
}

func TestEventsDBRecordsAndReplays(t *testing.T) {
	e := startMux(t, epSpec{name: "m", scenario: "slow"})
	e.waitState("m", StateUp, 10*time.Second)
	before := time.Now().UTC()
	oneRun(t, e, "m")
	waitFor(t, 3*time.Second, "idle", func() bool { return e.srv.Service("m").Info().State == StateUp })

	// the file: SQLite, events(ts, service, run, type, json)
	e.srv.Hub().log.flush()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(e.dir, "logs", "events.db")+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	rows, err := db.Query(`SELECT ts, service, run, type, json FROM events ORDER BY rowid`)
	if err != nil {
		t.Fatal(err)
	}
	types := map[string]int{}
	var lastTS string
	n := 0
	for rows.Next() {
		var ts, service, run, typ, js string
		if err := rows.Scan(&ts, &service, &run, &typ, &js); err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if json.Unmarshal([]byte(js), &m) != nil || m["type"] != typ || m["time"] != ts {
			t.Fatalf("row %d: type %s ts %s json %s", n, typ, ts, js)
		}
		if _, err := time.Parse("2006-01-02T15:04:05.000000Z", ts); err != nil || ts < lastTS {
			t.Fatalf("ts %q after %q: %v", ts, lastTS, err)
		}
		lastTS = ts
		if strings.HasPrefix(typ, "run.") && (service != "m" || run == "") {
			t.Errorf("%s: service %q run %q", typ, service, run)
		}
		if typ == "service.state" && service != "m" {
			t.Errorf("service.state: service %q", service)
		}
		types[typ]++
		n++
	}
	rows.Close()
	for _, typ := range []string{"ai-mux.start", "service.state", "run.start", "run.prefill", "run.stats", "run.tokens", "run.end"} {
		if types[typ] == 0 {
			t.Errorf("no %s row (%v)", typ, types)
		}
	}

	// replay: everything since a time in the past, then live
	es, code := openEvents(t, e.base, "?since="+before.Add(-time.Hour).Format(time.RFC3339), nil)
	if code != 200 {
		t.Fatal(code)
	}
	old := es.until(t, 5*time.Second, func(ev sseEvent) bool { return ev.Name == "run.end" })
	seen := map[string]int{}
	var prev uint64
	for _, ev := range old {
		id := idOf(t, ev)
		if id <= prev {
			t.Errorf("ids not increasing: %d after %d", id, prev)
		}
		prev = id
		seen[ev.Name]++
	}
	if old[0].Name != "ai-mux.start" || seen["service.state"] < 2 || seen["run.start"] != 1 || seen["snapshot"] != 0 {
		t.Errorf("replayed %v (first %s)", seen, old[0].Name)
	}
	oneRun(t, e, "m") // live, on the same connection, after the replay with no gap and no duplicate
	live := es.until(t, 5*time.Second, func(ev sseEvent) bool { return ev.Name == "run.end" })
	if live[0].Name != "service.state" && live[0].Name != "run.start" {
		t.Errorf("first live event %s", live[0].Name)
	}
	for _, ev := range live {
		if id := idOf(t, ev); id <= prev {
			t.Errorf("duplicate or reordered live event %s: id %d after %d", ev.Name, id, prev)
		} else {
			prev = id
		}
	}
	es.cancel()

	// Unix seconds work as since; a time in the future replays nothing, only what comes
	es2, code := openEvents(t, e.base, "?since="+strconv.FormatFloat(float64(before.Add(time.Hour).UnixNano())/1e9, 'f', 3, 64), nil)
	if code != 200 {
		t.Fatal(code)
	}
	oneRun(t, e, "m")
	first := es2.until(t, 5*time.Second, func(ev sseEvent) bool { return true })
	if first[0].Name == "ai-mux.start" || idOf(t, first[0]) <= prev {
		t.Errorf("a future since replayed %s id %s", first[0].Name, first[0].ID)
	}
	es2.cancel()

	// Last-Event-ID resumes after an event
	es3, _ := openEvents(t, e.base, "", map[string]string{"Last-Event-ID": strconv.FormatUint(idOf(t, old[2]), 10)})
	again := es3.until(t, 5*time.Second, func(ev sseEvent) bool { return true })
	if idOf(t, again[0]) != idOf(t, old[3]) || again[0].Name != old[3].Name {
		t.Errorf("resumed at %s %s, want %s %s", again[0].Name, again[0].ID, old[3].Name, old[3].ID)
	}
	es3.cancel()

	// without since: the current state, then what happens (not recorded history)
	es4, _ := openEvents(t, e.base, "", nil)
	snap := es4.until(t, 5*time.Second, func(ev sseEvent) bool { return true })
	if snap[0].Name != "snapshot" {
		t.Errorf("first event %s", snap[0].Name)
	}
	es4.cancel()

	if _, code := openEvents(t, e.base, "?since=yesterday", nil); code != 400 {
		t.Errorf("a bad since: %d", code)
	}
}

func TestEventLogPersistsAndContinues(t *testing.T) {
	path := filepath.Join(shortDir(t), "events.db")
	h, err := NewRecordingHub(path)
	if err != nil {
		t.Fatal(err)
	}
	h.Publish("service.state", map[string]any{"service": "a", "state": "up"})
	h.Publish("run.start", map[string]any{"service": "a", "run": "r1"})
	time.Sleep(20 * time.Millisecond)
	cut := time.Now().UTC().Format(tsLayout)
	time.Sleep(20 * time.Millisecond)
	h.Publish("run.end", map[string]any{"service": "a", "run": "r1", "status": "done"})
	evs, err := h.Replay("", 0, 0)
	if err != nil || len(evs) != 3 || evs[0].Seq != 1 || evs[2].Type != "run.end" {
		t.Fatalf("replay: %v %v", evs, err)
	}
	if evs, _ = h.Replay(cut, 0, 0); len(evs) != 1 || evs[0].Type != "run.end" {
		t.Errorf("since: %v", evs)
	}
	if evs, _ = h.Replay("", 1, 2); len(evs) != 1 || evs[0].Seq != 2 {
		t.Errorf("after/upTo: %v", evs)
	}
	h.Close()
	h.Publish("late", nil) // after Close: dropped, not a panic

	h2, err := NewRecordingHub(path)
	if err != nil {
		t.Fatal(err)
	}
	defer h2.Close()
	if h2.Seq() != 3 {
		t.Fatalf("seq after reopen: %d", h2.Seq())
	}
	h2.Publish("service.state", map[string]any{"service": "a", "state": "down"})
	evs, _ = h2.Replay("", 0, 0)
	if len(evs) != 4 || evs[3].Seq != 4 || evs[3].Type != "service.state" {
		t.Errorf("after reopen: %v", evs)
	}
}
