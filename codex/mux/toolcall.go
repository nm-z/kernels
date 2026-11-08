package mux

import (
	"bytes"
	"encoding/json"
	"strings"
)

// param is one <parameter=NAME>VALUE</parameter> of a Qwen XML tool call, in the order written.
type param struct{ name, value string }

// parseFunctionBlock reads the inside of <tool_call>…</tool_call>:
//
//	<function=NAME>
//	<parameter=P>
//	VALUE
//	</parameter>
//	</function>
//
// It is lenient about a missing </parameter> or </function> at the end (a run cut off by max_new).
func parseFunctionBlock(raw string) (name string, params []param, ok bool) {
	s := strings.TrimSpace(raw)
	if !strings.HasPrefix(s, "<function=") {
		return "", nil, false
	}
	s = s[len("<function="):]
	j := strings.IndexByte(s, '>')
	if j < 0 {
		return "", nil, false
	}
	name = strings.TrimSpace(s[:j])
	s = s[j+1:]
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" || strings.HasPrefix(s, "</function>") {
			break
		}
		if !strings.HasPrefix(s, "<parameter=") {
			return "", nil, false
		}
		s = s[len("<parameter="):]
		j = strings.IndexByte(s, '>')
		if j < 0 {
			return "", nil, false
		}
		pname := strings.TrimSpace(s[:j])
		s = s[j+1:]
		var val string
		if end := strings.Index(s, "</parameter>"); end >= 0 {
			val, s = s[:end], s[end+len("</parameter>"):]
		} else if k := strings.Index(s, "</function>"); k >= 0 {
			val, s = s[:k], s[k:]
		} else {
			val, s = s, ""
		}
		// the template writes <parameter=P>\nVALUE\n</parameter>
		val = strings.TrimPrefix(val, "\n")
		val = strings.TrimSuffix(val, "\n")
		params = append(params, param{pname, val})
	}
	return name, params, name != ""
}

// argValue is a parameter value as JSON. With the tool's schema: a string parameter stays a string,
// others are read as JSON of their type. Without: a value that is a JSON number, boolean, object or
// array keeps that type, anything else is a string.
func argValue(val string, types []string) json.RawMessage {
	str := func() json.RawMessage { b, _ := json.Marshal(val); return b }
	t := strings.TrimSpace(val)
	has := func(x string) bool {
		for _, y := range types {
			if y == x {
				return true
			}
		}
		return false
	}
	if len(types) > 0 {
		if len(types) == 1 && types[0] == "string" {
			return str()
		}
		if raw, kind := jsonKind(t); raw != nil {
			switch kind {
			case 'n':
				if has("number") || has("integer") {
					return raw
				}
			case 'b':
				if has("boolean") {
					return raw
				}
			case 'o':
				if has("object") {
					return raw
				}
			case 'a':
				if has("array") {
					return raw
				}
			case 'z':
				if has("null") {
					return raw
				}
			}
		}
		return str()
	}
	if raw, kind := jsonKind(t); raw != nil && kind != 'z' {
		return raw
	}
	return str()
}

// jsonKind returns t as raw JSON and its kind (n number, b bool, o object, a array, z null, s string)
// when t is exactly one valid JSON value.
func jsonKind(t string) (json.RawMessage, byte) {
	if t == "" || !json.Valid([]byte(t)) {
		return nil, 0
	}
	var kind byte
	switch c := t[0]; {
	case c == '{':
		kind = 'o'
	case c == '[':
		kind = 'a'
	case c == '"':
		kind = 's'
	case c == 't' || c == 'f':
		kind = 'b'
	case c == 'n':
		kind = 'z'
	default:
		kind = 'n'
	}
	if kind == 's' {
		return nil, 0 // a quoted string is the text the model wrote
	}
	var buf bytes.Buffer
	json.Compact(&buf, []byte(t))
	return json.RawMessage(buf.Bytes()), kind
}

// argsObject builds {"p":v,…} in the order written (a repeated parameter keeps its last value).
func argsObject(params []param, types map[string][]string) json.RawMessage {
	var order []string
	vals := map[string]json.RawMessage{}
	for _, p := range params {
		if _, dup := vals[p.name]; !dup {
			order = append(order, p.name)
		}
		vals[p.name] = argValue(p.value, types[p.name])
	}
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range order {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		b.Write(kb)
		b.WriteByte(':')
		b.Write(vals[k])
	}
	b.WriteByte('}')
	return b.Bytes()
}
