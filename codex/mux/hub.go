package mux

import (
	"encoding/json"
	"sync"
	"time"
)

// Hub fans events out to any number of observers (GET /v1/events). A slow observer loses events
// rather than slowing the endpoints down.
type Hub struct {
	mu   sync.Mutex
	subs map[*Sub]struct{}
}

// Sub is one observer.
type Sub struct {
	C       chan HubEvent
	dropped int
}

// HubEvent is one event: its type (the SSE event name) and its JSON.
type HubEvent struct {
	Type string
	Data []byte
}

func NewHub() *Hub { return &Hub{subs: map[*Sub]struct{}{}} }

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

// Publish sends {"type":typ,"time":now,...fields} to every observer.
func (h *Hub) Publish(typ string, fields map[string]any) {
	m := make(map[string]any, len(fields)+2)
	for k, v := range fields {
		m[k] = v
	}
	m["type"] = typ
	m["time"] = time.Now().Format(time.RFC3339Nano)
	b, err := json.Marshal(m)
	if err != nil {
		return
	}
	ev := HubEvent{Type: typ, Data: b}
	h.mu.Lock()
	defer h.mu.Unlock()
	for s := range h.subs {
		select {
		case s.C <- ev:
		default:
			s.dropped++
		}
	}
}
