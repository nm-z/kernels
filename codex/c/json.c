/* json.c: see json.h */
#include "json.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- arena: a list of blocks, freed together ---- */
struct arena { struct blk *head; };
struct blk { struct blk *next; size_t used, cap; _Alignas(16) char data[]; };
arena *arena_new(void) { return calloc(1, sizeof(arena)); }
void arena_free(arena *a) { if (!a) return; for (struct blk *b = a->head, *n; b; b = n) { n = b->next; free(b); } free(a); }
void *arena_alloc(arena *a, size_t n) {
	n = (n + 15) & ~(size_t)15;
	if (!a->head || a->head->used + n > a->head->cap) {
		size_t cap = n > 65536 ? n : 65536;
		struct blk *b = malloc(sizeof *b + cap); b->next = a->head; b->used = 0; b->cap = cap; a->head = b;
	}
	void *p = a->head->data + a->head->used; a->head->used += n;
	memset(p, 0, n);
	return p;
}
char *arena_strndup(arena *a, const char *s, size_t n) { char *p = arena_alloc(a, n + 1); memcpy(p, s, n); p[n] = 0; return p; }

/* ---- sbuf ---- */
static void sb_grow(sbuf *b, size_t add) {
	if (b->n + add + 1 <= b->cap) return;
	size_t c = b->cap ? b->cap : 256; while (c < b->n + add + 1) c *= 2;
	b->s = realloc(b->s, c); b->cap = c;
}
void sb_put(sbuf *b, const char *s, size_t n) { sb_grow(b, n); memcpy(b->s + b->n, s, n); b->n += n; b->s[b->n] = 0; }
void sb_puts(sbuf *b, const char *s) { sb_put(b, s, strlen(s)); }
void sb_putc(sbuf *b, char c) { sb_put(b, &c, 1); }
void sb_printf(sbuf *b, const char *fmt, ...) {
	va_list ap; va_start(ap, fmt); int n = vsnprintf(NULL, 0, fmt, ap); va_end(ap);
	sb_grow(b, (size_t)n); va_start(ap, fmt); vsnprintf(b->s + b->n, (size_t)n + 1, fmt, ap); va_end(ap); b->n += (size_t)n;
}
void sb_free(sbuf *b) { free(b->s); b->s = NULL; b->n = b->cap = 0; }
void sb_json_str(sbuf *b, const char *s, size_t n) {
	sb_putc(b, '"');
	for (size_t i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];
		switch (c) {
		case '"': sb_put(b, "\\\"", 2); break;
		case '\\': sb_put(b, "\\\\", 2); break;
		case '\n': sb_put(b, "\\n", 2); break;
		case '\r': sb_put(b, "\\r", 2); break;
		case '\t': sb_put(b, "\\t", 2); break;
		case '\b': sb_put(b, "\\b", 2); break;
		case '\f': sb_put(b, "\\f", 2); break;
		default: if (c < 0x20 || c == 0x7f) sb_printf(b, "\\u%04x", c); else sb_putc(b, (char)c);
		}
	}
	sb_putc(b, '"');
}

/* ---- values ---- */
jv *jv_new(arena *a, int k) { jv *v = arena_alloc(a, sizeof *v); v->k = k; return v; }
jv *jv_str(arena *a, const char *s, size_t n) { jv *v = jv_new(a, J_STR); v->s.s = arena_strndup(a, s, n); v->s.n = n; return v; }
jv *jv_strref(arena *a, const char *s, size_t n) { jv *v = jv_new(a, J_STR); v->s.s = s; v->s.n = n; (void)a; return v; }
jv *jv_int(arena *a, long long i) { jv *v = jv_new(a, J_INT); v->i = i; return v; }
jv *jv_bool(arena *a, int b) { jv *v = jv_new(a, J_BOOL); v->b = !!b; return v; }
void jv_push(arena *a, jv *l, jv *v) {
	if (l->l.n == l->l.cap) { int c = l->l.cap ? 2 * l->l.cap : 8; jv **n = arena_alloc(a, sizeof *n * (size_t)c); if (l->l.n) memcpy(n, l->l.a, sizeof *n * (size_t)l->l.n); l->l.a = n; l->l.cap = c; }
	l->l.a[l->l.n++] = v;
}
jv *jv_get(const jv *d, const char *key, size_t kl) {
	if (!d || (d->k != J_DICT && d->k != J_NS)) return NULL;
	for (int i = 0; i < d->o.n; i++) if (d->o.klen[i] == kl && !memcmp(d->o.key[i], key, kl)) return d->o.val[i];
	return NULL;
}
jv *jv_getz(const jv *d, const char *key) { return jv_get(d, key, strlen(key)); }
void jv_set(arena *a, jv *d, const char *key, size_t kl, jv *v) {
	for (int i = 0; i < d->o.n; i++) if (d->o.klen[i] == kl && !memcmp(d->o.key[i], key, kl)) { d->o.val[i] = v; return; }
	if (d->o.n == d->o.cap) {
		int c = d->o.cap ? 2 * d->o.cap : 8;
		const char **k = arena_alloc(a, sizeof *k * (size_t)c); size_t *l = arena_alloc(a, sizeof *l * (size_t)c); jv **vv = arena_alloc(a, sizeof *vv * (size_t)c);
		if (d->o.n) { memcpy(k, d->o.key, sizeof *k * (size_t)d->o.n); memcpy(l, d->o.klen, sizeof *l * (size_t)d->o.n); memcpy(vv, d->o.val, sizeof *vv * (size_t)d->o.n); }
		d->o.key = k; d->o.klen = l; d->o.val = vv; d->o.cap = c;
	}
	d->o.key[d->o.n] = arena_strndup(a, key, kl); d->o.klen[d->o.n] = kl; d->o.val[d->o.n] = v; d->o.n++;
}

/* ---- JSON reader ---- */
typedef struct { const char *p, *e; arena *a; const char *err; int depth; } jp;
static void ws(jp *j) { while (j->p < j->e && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++; }
static int hex4(const char *p, unsigned *v) {
	*v = 0;
	for (int i = 0; i < 4; i++) { char c = p[i]; *v <<= 4; if (c >= '0' && c <= '9') *v |= (unsigned)(c - '0'); else if (c >= 'a' && c <= 'f') *v |= (unsigned)(c - 'a' + 10); else if (c >= 'A' && c <= 'F') *v |= (unsigned)(c - 'A' + 10); else return -1; }
	return 0;
}
static int put_utf8(unsigned c, char *o) {
	if (c < 0x80) { o[0] = (char)c; return 1; }
	if (c < 0x800) { o[0] = (char)(0xc0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3f)); return 2; }
	if (c < 0x10000) { o[0] = (char)(0xe0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3f)); o[2] = (char)(0x80 | (c & 0x3f)); return 3; }
	o[0] = (char)(0xf0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3f)); o[2] = (char)(0x80 | ((c >> 6) & 0x3f)); o[3] = (char)(0x80 | (c & 0x3f)); return 4;
}
static jv *jstring(jp *j) {
	j->p++;   /* " */
	const char *s = j->p; size_t esc = 0;
	while (j->p < j->e && *j->p != '"') { if (*j->p == '\\') { j->p++; esc++; } j->p++; }
	if (j->p >= j->e) { j->err = "unterminated string"; return NULL; }
	const char *end = j->p++;
	if (!esc) return jv_str(j->a, s, (size_t)(end - s));
	char *o = arena_alloc(j->a, (size_t)(end - s) + 1); size_t n = 0;
	for (const char *q = s; q < end; q++) {
		if (*q != '\\') { o[n++] = *q; continue; }
		q++;
		switch (*q) {
		case 'n': o[n++] = '\n'; break; case 't': o[n++] = '\t'; break; case 'r': o[n++] = '\r'; break;
		case 'b': o[n++] = '\b'; break; case 'f': o[n++] = '\f'; break; case '/': o[n++] = '/'; break;
		case '\\': o[n++] = '\\'; break; case '"': o[n++] = '"'; break;
		case 'u': {
			unsigned c, c2;
			if (end - q < 5 || hex4(q + 1, &c)) { j->err = "bad \\u escape"; return NULL; }
			q += 4;
			if (c >= 0xd800 && c < 0xdc00 && end - q >= 7 && q[1] == '\\' && q[2] == 'u' && !hex4(q + 3, &c2) && c2 >= 0xdc00 && c2 < 0xe000) { c = 0x10000 + ((c - 0xd800) << 10) + (c2 - 0xdc00); q += 6; }
			else if (c >= 0xd800 && c < 0xe000) c = 0xfffd;
			n += (size_t)put_utf8(c, o + n);
			break;
		}
		default: j->err = "bad escape"; return NULL;
		}
	}
	jv *v = jv_new(j->a, J_STR); v->s.s = o; v->s.n = n;
	return v;
}
static jv *jvalue(jp *j) {
	ws(j);
	if (j->p >= j->e) { j->err = "unexpected end"; return NULL; }
	if (++j->depth > 64) { j->err = "nested too deeply"; return NULL; }
	jv *v = NULL; char c = *j->p;
	if (c == '"') v = jstring(j);
	else if (c == '{') {
		v = jv_new(j->a, J_DICT); j->p++; ws(j);
		if (j->p < j->e && *j->p == '}') { j->p++; }
		else for (;;) {
			ws(j); if (j->p >= j->e || *j->p != '"') { j->err = "expected a key"; return NULL; }
			jv *k = jstring(j); if (!k) return NULL;
			ws(j); if (j->p >= j->e || *j->p != ':') { j->err = "expected ':'"; return NULL; }
			j->p++;
			jv *x = jvalue(j); if (!x) return NULL;
			jv_set(j->a, v, k->s.s, k->s.n, x);
			ws(j); if (j->p < j->e && *j->p == ',') { j->p++; continue; }
			if (j->p < j->e && *j->p == '}') { j->p++; break; }
			j->err = "expected ',' or '}'"; return NULL;
		}
	} else if (c == '[') {
		v = jv_new(j->a, J_LIST); j->p++; ws(j);
		if (j->p < j->e && *j->p == ']') { j->p++; }
		else for (;;) {
			jv *x = jvalue(j); if (!x) return NULL;
			jv_push(j->a, v, x);
			ws(j); if (j->p < j->e && *j->p == ',') { j->p++; continue; }
			if (j->p < j->e && *j->p == ']') { j->p++; break; }
			j->err = "expected ',' or ']'"; return NULL;
		}
	} else if (j->e - j->p >= 4 && !memcmp(j->p, "true", 4)) { v = jv_bool(j->a, 1); j->p += 4; }
	else if (j->e - j->p >= 5 && !memcmp(j->p, "false", 5)) { v = jv_bool(j->a, 0); j->p += 5; }
	else if (j->e - j->p >= 4 && !memcmp(j->p, "null", 4)) { v = jv_new(j->a, J_NULL); j->p += 4; }
	else if (c == '-' || (c >= '0' && c <= '9')) {
		const char *s = j->p; int flt = 0;
		while (j->p < j->e && strchr("+-0123456789.eE", *j->p)) { if (strchr(".eE", *j->p)) flt = 1; j->p++; }
		char tmp[64]; size_t n = (size_t)(j->p - s); if (n >= sizeof tmp) { j->err = "number too long"; return NULL; }
		memcpy(tmp, s, n); tmp[n] = 0;
		if (flt) { v = jv_new(j->a, J_NUM); v->d = strtod(tmp, NULL); } else v = jv_int(j->a, strtoll(tmp, NULL, 10));
	} else j->err = "unexpected character";
	j->depth--;
	return v;
}
jv *json_parse(arena *a, const char *s, size_t n, const char **err) {
	jp j = { s, s + n, a, NULL, 0 };
	jv *v = jvalue(&j);
	if (v) { ws(&j); if (j.p != j.e) { j.err = "trailing characters"; v = NULL; } }
	if (!v && err) *err = j.err ? j.err : "syntax error";
	return v;
}

/* ---- JSON writer ---- */
void json_dump(sbuf *b, const jv *v, int sep) {
	const char *comma = sep ? ", " : ",", *colon = sep ? ": " : ":";
	switch (v ? v->k : J_NULL) {
	case J_BOOL: sb_puts(b, v->b ? "true" : "false"); break;
	case J_INT: sb_printf(b, "%lld", v->i); break;
	case J_NUM: sb_printf(b, "%.17g", v->d); break;
	case J_STR: sb_json_str(b, v->s.s, v->s.n); break;
	case J_LIST: case J_TUPLE:
		sb_putc(b, '[');
		for (int i = 0; i < v->l.n; i++) { if (i) sb_puts(b, comma); json_dump(b, v->l.a[i], sep); }
		sb_putc(b, ']'); break;
	case J_DICT: case J_NS:
		sb_putc(b, '{');
		for (int i = 0; i < v->o.n; i++) { if (i) sb_puts(b, comma); sb_json_str(b, v->o.key[i], v->o.klen[i]); sb_puts(b, colon); json_dump(b, v->o.val[i], sep); }
		sb_putc(b, '}'); break;
	default: sb_puts(b, "null");
	}
}
