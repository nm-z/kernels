// Package tmpl is ai-mux's model side: the GGUF tokenizer (byte-level BPE, qwen35 pre-tokenizer) and
// the GGUF chat template (a Jinja interpreter), both in C (../c) linked through cgo.
package tmpl

/*
#cgo CFLAGS: -std=gnu99 -O2 -I${SRCDIR}/../c
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tok.h"
#include "jinja.h"
#include "json.h"

// aimux_render parses vars (a JSON object) and renders the template with them. The result is
// malloc'ed (free it); on failure it returns NULL and fills err.
static char *aimux_render(const jinja *t, const char *vars, size_t n, size_t *outn, char *err, size_t errcap) {
	arena *a = arena_new();
	const char *je = NULL;
	jv *v = json_parse(a, vars, n, &je);
	if (!v || v->k != J_DICT) {
		snprintf(err, errcap, "template variables: %s", je ? je : "not a JSON object");
		arena_free(a);
		return NULL;
	}
	char *out = jinja_render(t, a, v, outn, err, errcap);
	arena_free(a);
	return out;
}
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"unsafe"
)

// Tokenizer is a loaded GGUF vocabulary and chat template. It is safe for concurrent use.
type Tokenizer struct {
	v        *C.tok_vocab
	tpl      *C.jinja
	template string
}

// Load reads the tokenizer and chat template from a GGUF file.
func Load(path string) (*Tokenizer, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	v := C.tok_load(cp)
	if v == nil {
		return nil, fmt.Errorf("%s: cannot load the tokenizer (see stderr)", path)
	}
	t := &Tokenizer{v: v}
	ts := C.tok_template(v)
	if ts == nil {
		return nil, fmt.Errorf("%s: no chat template", path)
	}
	t.template = C.GoString(ts)
	src := C.CString(t.template)
	defer C.free(unsafe.Pointer(src))
	var errbuf [512]C.char
	tpl := C.jinja_compile(src, C.size_t(len(t.template)), &errbuf[0], C.size_t(len(errbuf)))
	if tpl == nil {
		return nil, fmt.Errorf("%s: chat template: %s", path, C.GoString(&errbuf[0]))
	}
	t.tpl = tpl
	return t, nil
}

// Template is the GGUF's tokenizer.chat_template.
func (t *Tokenizer) Template() string { return t.template }

// Count is the vocabulary size.
func (t *Tokenizer) Count() int { return int(C.tok_count(t.v)) }

// Render renders the chat template with vars (marshalled to JSON and read back by the C side).
func (t *Tokenizer) Render(vars map[string]any) (string, error) {
	js, err := json.Marshal(vars)
	if err != nil {
		return "", err
	}
	var n C.size_t
	var errbuf [1024]C.char
	cs := C.CString(string(js))
	defer C.free(unsafe.Pointer(cs))
	out := C.aimux_render(t.tpl, cs, C.size_t(len(js)), &n, &errbuf[0], C.size_t(len(errbuf)))
	if out == nil {
		return "", errors.New(C.GoString(&errbuf[0]))
	}
	defer C.free(unsafe.Pointer(out))
	return C.GoStringN(out, C.int(n)), nil
}

// Encode tokenizes text; control tokens written as text (<|im_start|>) become their ids.
func (t *Tokenizer) Encode(text string) ([]uint32, error) {
	if text == "" {
		return nil, nil
	}
	buf := make([]uint32, len(text)+16)
	n := C.tok_encode(t.v, (*C.char)(unsafe.Pointer(unsafe.StringData(text))), C.size_t(len(text)), 1,
		(*C.uint32_t)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if n < 0 {
		return nil, errors.New("tokenizer output overflow")
	}
	return buf[:n], nil
}

// Piece is the bytes a token adds to the output (special tokens as their text).
func (t *Tokenizer) Piece(id uint32) []byte {
	var buf [256]byte
	n := C.tok_piece(t.v, C.uint32_t(id), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if n <= 0 {
		return nil
	}
	return append([]byte(nil), buf[:n]...)
}

// Hidden reports control and unknown tokens, which a response does not show.
func (t *Tokenizer) Hidden(id uint32) bool { return C.tok_is_hidden(t.v, C.uint32_t(id)) != 0 }

// Find returns the id of the token with exactly this text, or -1.
func (t *Tokenizer) Find(text string) int {
	cs := C.CString(text)
	defer C.free(unsafe.Pointer(cs))
	return int(C.tok_find(t.v, cs))
}

// EOS is tokenizer.ggml.eos_token_id (0xffffffff when absent).
func (t *Tokenizer) EOS() uint32 { return uint32(C.tok_eos(t.v)) }
