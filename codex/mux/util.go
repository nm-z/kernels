package mux

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
)

// newID returns prefix + 24 random hex characters (resp_…, msg_…, rs_…, fc_…, call_…).
func newID(prefix string) string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return prefix + hex.EncodeToString(b[:])
}

// apiError is an OpenAI-style error: {"error":{"message","type","param","code"}}.
type apiError struct {
	Status  int
	Type    string
	Code    string
	Param   string
	Message string
}

func (e *apiError) Error() string { return e.Message }

func badRequest(code, param, format string, args ...any) *apiError {
	return &apiError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: code, Param: param, Message: fmt.Sprintf(format, args...)}
}

func (e *apiError) body() []byte {
	inner := map[string]any{"message": e.Message, "type": e.Type, "code": e.Code}
	if e.Param != "" {
		inner["param"] = e.Param
	} else {
		inner["param"] = nil
	}
	b, _ := json.Marshal(map[string]any{"error": inner})
	return b
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(b)
	w.Write([]byte("\n"))
}

func writeAPIError(w http.ResponseWriter, e *apiError) {
	w.Header().Set("Content-Type", "application/json")
	if e.Status == http.StatusServiceUnavailable {
		w.Header().Set("Retry-After", "5")
	}
	w.WriteHeader(e.Status)
	w.Write(e.body())
	w.Write([]byte("\n"))
}

// expandHome turns a leading ~/ into the home directory.
func expandHome(p, home string) string {
	if p == "~" {
		return home
	}
	if strings.HasPrefix(p, "~/") {
		return home + p[1:]
	}
	return p
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sortStrings(s []string) { sort.Strings(s) }

func hashBytes(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("null")
	}
	return b
}

func jsonMarshal(v any) ([]byte, error) { return json.Marshal(v) }

// toolArgsText is a tool_call's arguments as JSON text: a shim may send the text itself (a JSON string)
// or the object.
func toolArgsText(raw json.RawMessage) string {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || bytes.Equal(t, []byte("null")) {
		return "{}"
	}
	if t[0] == '"' {
		var s string
		if json.Unmarshal(t, &s) == nil {
			if strings.TrimSpace(s) == "" {
				return "{}"
			}
			return s
		}
	}
	var buf bytes.Buffer
	if json.Compact(&buf, t) == nil {
		return buf.String()
	}
	return string(t)
}
