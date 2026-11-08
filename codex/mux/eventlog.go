package mux

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite" // pure Go: no cgo
)

// tsLayout is the fixed-width UTC time events carry and events.db stores (it sorts as text).
const tsLayout = "2006-01-02T15:04:05.000000Z"

// parseSince reads GET /v1/events?since=: a time (RFC 3339) or Unix seconds.
func parseSince(s string) (string, error) {
	s = strings.TrimSpace(s)
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.UTC().Format(tsLayout), nil
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		sec := int64(f)
		return time.Unix(sec, int64((f-float64(sec))*1e9)).UTC().Format(tsLayout), nil
	}
	return "", fmt.Errorf("since %q is neither an RFC 3339 time nor Unix seconds", s)
}

// Hub fans every event out to any number of observers (GET /v1/events) and appends it to events.db. A
// slow observer loses events rather than slowing the services down.
type Hub struct {
	mu     sync.Mutex
	subs   map[*Sub]struct{}
	seq    uint64
	log    *eventLog
	closed bool
}

// Sub is one observer.
type Sub struct {
	C       chan HubEvent
	dropped int
}

// HubEvent is one event: its sequence number, its type (the SSE event name) and its JSON.
type HubEvent struct {
	Seq  uint64
	Type string
	Data []byte
}

// NewHub makes a hub that records nothing.
func NewHub() *Hub { return &Hub{subs: map[*Sub]struct{}{}} }

// NewRecordingHub makes a hub that appends every event to the SQLite file at path.
func NewRecordingHub(path string) (*Hub, error) {
	l, last, err := openEventLog(path)
	if err != nil {
		return nil, err
	}
	return &Hub{subs: map[*Sub]struct{}{}, log: l, seq: last}, nil
}

func (h *Hub) Subscribe() *Sub {
	s := &Sub{C: make(chan HubEvent, 512)}
	h.mu.Lock()
	h.subs[s] = struct{}{}
	h.mu.Unlock()
	return s
}

func (h *Hub) Unsubscribe(s *Sub) {
	h.mu.Lock()
	delete(h.subs, s)
	h.mu.Unlock()
}

// Publish sends {"type":typ,"time":now,...fields} to every observer and records it.
func (h *Hub) Publish(typ string, fields map[string]any) {
	now := time.Now().UTC().Format(tsLayout)
	m := make(map[string]any, len(fields)+2)
	for k, v := range fields {
		m[k] = v
	}
	m["type"] = typ
	m["time"] = now
	b, err := json.Marshal(m)
	if err != nil {
		return
	}
	svc, _ := fields["service"].(string)
	run, _ := fields["run"].(string)
	h.mu.Lock()
	defer h.mu.Unlock()
	h.seq++
	ev := HubEvent{Seq: h.seq, Type: typ, Data: b}
	if h.log != nil && !h.closed {
		h.log.add(logRow{seq: h.seq, ts: now, service: svc, run: run, typ: typ, json: b})
	}
	for s := range h.subs {
		select {
		case s.C <- ev:
		default:
			s.dropped++
		}
	}
}

// Seq is the sequence number of the last event published.
func (h *Hub) Seq() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.seq
}

// Replay returns the recorded events with a sequence number above after (and up to upTo, if not 0) whose
// time is at or after since (if not ""). It waits for what was published so far to be written.
func (h *Hub) Replay(since string, after, upTo uint64) ([]HubEvent, error) {
	if h.log == nil {
		return nil, nil
	}
	h.log.flush()
	return h.log.query(since, after, upTo)
}

// Close flushes and closes events.db.
func (h *Hub) Close() {
	h.mu.Lock()
	h.closed = true
	h.mu.Unlock()
	if h.log != nil {
		h.log.close()
	}
}

// ---- events.db -----------------------------------------------------------------------------------------------

type logRow struct {
	seq                   uint64
	ts, service, run, typ string
	json                  []byte
	flushed               chan struct{} // a flush marker, not an event
}

type eventLog struct {
	db        *sql.DB
	in        chan logRow
	done      chan struct{}
	quit      chan struct{}
	dropped   uint64
	mu        sync.Mutex
	closeOnce sync.Once
}

func openEventLog(path string) (*eventLog, uint64, error) {
	dsn := "file:" + path + "?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, 0, err
	}
	db.SetMaxOpenConns(4)
	for _, q := range []string{
		`CREATE TABLE IF NOT EXISTS events(ts TEXT NOT NULL, service TEXT NOT NULL DEFAULT '', run TEXT NOT NULL DEFAULT '', type TEXT NOT NULL, json TEXT NOT NULL)`,
		`CREATE INDEX IF NOT EXISTS events_ts ON events(ts)`,
		`CREATE INDEX IF NOT EXISTS events_service_ts ON events(service, ts)`,
	} {
		if _, err := db.Exec(q); err != nil {
			db.Close()
			return nil, 0, fmt.Errorf("%s: %w", path, err)
		}
	}
	var last sql.NullInt64
	if err := db.QueryRow(`SELECT max(rowid) FROM events`).Scan(&last); err != nil {
		db.Close()
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	l := &eventLog{db: db, in: make(chan logRow, 16384), done: make(chan struct{}), quit: make(chan struct{})}
	go l.writer()
	return l, uint64(last.Int64), nil
}

// add queues an event for the writer (never blocks: a full queue drops the event and counts it).
func (l *eventLog) add(r logRow) {
	select {
	case l.in <- r:
	default:
		l.mu.Lock()
		l.dropped++
		l.mu.Unlock()
	}
}

// flush returns when everything added before it is committed.
func (l *eventLog) flush() {
	ch := make(chan struct{})
	select {
	case l.in <- logRow{flushed: ch}:
	case <-l.done:
		return
	}
	select {
	case <-ch:
	case <-l.done:
	}
}

func (l *eventLog) close() {
	l.closeOnce.Do(func() {
		l.flush()
		close(l.quit)
		<-l.done
		l.db.Close()
	})
}

// writer commits queued events in batches (at most every 50 ms or 256 events).
func (l *eventLog) writer() {
	defer close(l.done)
	var batch []logRow
	var marks []chan struct{}
	commit := func() {
		if len(batch) > 0 {
			l.write(batch)
		}
		batch = batch[:0]
		for _, m := range marks {
			close(m)
		}
		marks = marks[:0]
	}
	t := time.NewTimer(time.Hour)
	t.Stop()
	for {
		select {
		case <-l.quit:
			for {
				select {
				case r := <-l.in:
					if r.flushed != nil {
						marks = append(marks, r.flushed)
					} else {
						batch = append(batch, r)
					}
				default:
					commit()
					return
				}
			}
		case r := <-l.in:
			if r.flushed != nil {
				marks = append(marks, r.flushed)
				commit()
				continue
			}
			if len(batch) == 0 {
				t.Reset(50 * time.Millisecond)
			}
			batch = append(batch, r)
			if len(batch) >= 256 {
				commit()
			}
		case <-t.C:
			commit()
		}
	}
}

func (l *eventLog) write(rows []logRow) {
	tx, err := l.db.Begin()
	if err != nil {
		return
	}
	st, err := tx.Prepare(`INSERT INTO events(rowid, ts, service, run, type, json) VALUES(?,?,?,?,?,?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	for _, r := range rows {
		st.Exec(r.seq, r.ts, r.service, r.run, r.typ, string(r.json))
	}
	st.Close()
	tx.Commit()
}

func (l *eventLog) query(since string, after, upTo uint64) ([]HubEvent, error) {
	q := `SELECT rowid, type, json FROM events WHERE rowid > ?`
	args := []any{after}
	if upTo > 0 {
		q += ` AND rowid <= ?`
		args = append(args, upTo)
	}
	if since != "" {
		q += ` AND ts >= ?`
		args = append(args, since)
	}
	rows, err := l.db.Query(q+` ORDER BY rowid`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []HubEvent
	for rows.Next() {
		var e HubEvent
		var js string
		if err := rows.Scan(&e.Seq, &e.Type, &js); err != nil {
			return nil, err
		}
		e.Data = []byte(js)
		out = append(out, e)
	}
	return out, rows.Err()
}
