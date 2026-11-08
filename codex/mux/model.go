package mux

import (
	"sync"

	"github.com/nm-z/kernels/codex/tmpl"
)

// Model is what ai-mux needs from a GGUF: the chat template and the vocabulary. *tmpl.Tokenizer is
// the real one; tests use a small fake.
type Model interface {
	Render(vars map[string]any) (string, error)
	Encode(text string) ([]uint32, error)
	Piece(id uint32) []byte
	Hidden(id uint32) bool
	Find(text string) int
	EOS() uint32
}

var _ Model = (*tmpl.Tokenizer)(nil)

// LoadTokenizer is the default model loader: the GGUF's tokenizer and chat template through cgo.
func LoadTokenizer(gguf string) (Model, error) { return tmpl.Load(gguf) }

type modelCache struct {
	load func(string) (Model, error)
	mu   sync.Mutex
	m    map[string]*modelEntry
}

type modelEntry struct {
	once sync.Once
	m    Model
	err  error
}

func (c *modelCache) get(gguf string) (Model, error) {
	c.mu.Lock()
	if c.m == nil {
		c.m = map[string]*modelEntry{}
	}
	e := c.m[gguf]
	if e == nil {
		e = &modelEntry{}
		c.m[gguf] = e
	}
	c.mu.Unlock()
	e.once.Do(func() { e.m, e.err = c.load(gguf) })
	if e.err != nil {
		// a failed load is retried next time
		c.mu.Lock()
		if c.m[gguf] == e {
			delete(c.m, gguf)
		}
		c.mu.Unlock()
	}
	return e.m, e.err
}
