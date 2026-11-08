#define _GNU_SOURCE
/* tok.c: see tok.h. Ported from llama.cpp (MIT): unicode_cpts_from_utf8, the qwen35 custom
 * regex split (unicode_regex_split_custom_qwen35), the byte-level encoding, the BPE merge loop
 * (llm_tokenizer_bpe_session::tokenize) and tokenizer_st_partition. */
#include "tok.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ucd.h"

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };

typedef struct { const char *s; uint32_t n; int32_t v; } hent;   /* open addressing, v = id or rank */
typedef struct { hent *e; uint32_t mask; } htab;

struct tok_vocab {
	gguf_file g;
	int n;
	const char **text; uint32_t *tlen; int32_t *type;
	htab ids;              /* token text -> id */
	htab ranks;            /* "left right" merge text -> rank */
	int *special; int nspecial;   /* control / user-defined / unknown ids, longest text first */
	uint32_t eos, bos; int add_bos;
	char *tmpl;
	char byte_utf8[256][3]; uint8_t byte_len[256];   /* byte -> its byte-level character */
	int16_t cpt_byte[512];                           /* byte-level codepoint -> byte (-1: none) */
};

/* ---- hashing ---- */
static uint32_t fnv(const char *s, uint32_t n) { uint32_t h = 2166136261u; for (uint32_t i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u; return h; }
static void ht_init(htab *t, uint32_t n) { uint32_t c = 1; while (c < 2 * n) c <<= 1; t->e = calloc(c, sizeof *t->e); t->mask = c - 1; }
static void ht_put(htab *t, const char *s, uint32_t n, int32_t v) {
	for (uint32_t i = fnv(s, n) & t->mask;; i = (i + 1) & t->mask) {
		if (!t->e[i].s) { t->e[i] = (hent){ s, n, v }; return; }
		if (t->e[i].n == n && !memcmp(t->e[i].s, s, n)) return;   /* first wins */
	}
}
static int32_t ht_get(const htab *t, const char *s, uint32_t n) {
	for (uint32_t i = fnv(s, n) & t->mask;; i = (i + 1) & t->mask) {
		if (!t->e[i].s) return -1;
		if (t->e[i].n == n && !memcmp(t->e[i].s, s, n)) return t->e[i].v;
	}
}

/* ---- unicode ---- */
static int put_utf8(uint32_t c, char *o) {
	if (c < 0x80) { o[0] = (char)c; return 1; }
	if (c < 0x800) { o[0] = (char)(0xc0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3f)); return 2; }
	if (c < 0x10000) { o[0] = (char)(0xe0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3f)); o[2] = (char)(0x80 | (c & 0x3f)); return 3; }
	o[0] = (char)(0xf0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3f)); o[2] = (char)(0x80 | ((c >> 6) & 0x3f)); o[3] = (char)(0x80 | (c & 0x3f)); return 4;
}
/* llama.cpp unicode_cpt_from_utf8 + unicode_cpts_from_utf8: an invalid sequence yields U+FFFD and skips one byte */
static size_t cpts_from_utf8(const uint8_t *s, size_t n, uint32_t *out) {
	size_t k = 0, i = 0;
	while (i < n) {
		uint8_t b = s[i]; uint32_t r;
		if (!(b & 0x80)) { out[k++] = b; i++; continue; }
		if (!(b & 0x40)) goto bad;
		if (!(b & 0x20)) {
			if (i + 1 >= n || (s[i + 1] & 0xc0) != 0x80) goto bad;
			r = ((uint32_t)(b & 0x1f) << 6) | (s[i + 1] & 0x3f); if (r < 0x80) goto bad;
			out[k++] = r; i += 2; continue;
		}
		if (!(b & 0x10)) {
			if (i + 2 >= n || (s[i + 1] & 0xc0) != 0x80 || (s[i + 2] & 0xc0) != 0x80) goto bad;
			r = ((uint32_t)(b & 0x0f) << 12) | ((uint32_t)(s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f); if (r < 0x800) goto bad;
			out[k++] = r; i += 3; continue;
		}
		if (!(b & 0x08)) {
			if (i + 3 >= n || (s[i + 1] & 0xc0) != 0x80 || (s[i + 2] & 0xc0) != 0x80 || (s[i + 3] & 0xc0) != 0x80) goto bad;
			r = ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3f) << 12) | ((uint32_t)(s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
			if (r < 0x10000 || r > 0x10ffff) goto bad;
			out[k++] = r; i += 4; continue;
		}
	bad:
		out[k++] = 0xfffd; i++;
	}
	return k;
}
enum { F_UNDEF = 0x01, F_NUMBER = 0x02, F_LETTER = 0x04, F_MARK = 0x10, F_WS = 0x100 };
static unsigned cpt_flags(uint32_t c) {
	if (c >= 0x110000) return 0;
	int lo = 0, hi = (int)(sizeof ucd_ranges / sizeof ucd_ranges[0]) - 1;   /* last entry starts at 0x110000 */
	while (hi - lo > 1) { int m = (lo + hi) / 2; if (ucd_ranges[m].start <= c) lo = m; else hi = m; }
	unsigned f = ucd_ranges[lo].flags;
	for (size_t i = 0; i < sizeof ucd_whitespace / sizeof ucd_whitespace[0]; i++) if (ucd_whitespace[i] == c) { f |= F_WS; break; }
	return f;
}
static uint32_t lower_ascii(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

/* unicode_regex_split_custom_qwen35 over cp[0..n): word lengths (in codepoints) into len[] */
static size_t split_qwen35(const uint32_t *cp, size_t n, size_t *len) {
	const uint32_t OUT = 0xffffffffu; size_t nw = 0, prev = 0;
#define CPT(p) ((p) < n ? cp[p] : OUT)            /* p must not have side effects */
#define FL(p) ((p) < n ? cpt_flags(cp[p]) : 0u)
#define ADD(e) do { size_t e_ = (e); if (e_ > prev) len[nw++] = e_ - prev; prev = e_; } while (0)
	for (size_t pos = 0; pos < n;) {
		uint32_t c = CPT(pos); unsigned f = FL(pos);
		if (c == '\'' && pos + 1 < n) {
			uint32_t c1 = lower_ascii(CPT(pos + 1));
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { ADD(pos + 2); pos += 2; continue; }
			if (pos + 2 < n) {
				uint32_t c2 = lower_ascii(CPT(pos + 2));
				if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { ADD(pos + 3); pos += 3; continue; }
			}
		}
		if (!(c == '\r' || c == '\n' || (f & F_NUMBER))) {
			if ((f & (F_LETTER | F_MARK)) || (FL(pos + 1) & (F_LETTER | F_MARK))) {
				pos++;
				while (FL(pos) & (F_LETTER | F_MARK)) pos++;
				ADD(pos); continue;
			}
		}
		if (f & F_NUMBER) { pos++; ADD(pos); continue; }
		unsigned f2 = c == ' ' ? FL(pos + 1) : f;
		if (!(f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) && f) {
			pos += c == ' ';
			while (!(f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) && f2) { pos++; f2 = FL(pos); }
			uint32_t c2 = CPT(pos);
			while (c2 == '\r' || c2 == '\n') { pos++; c2 = CPT(pos); }
			ADD(pos); continue;
		}
		size_t nws = 0, last_rn = 0;
		while (FL(pos + nws) & F_WS) { uint32_t c2 = CPT(pos + nws); if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1; nws++; }
		if (last_rn > 0) { pos = last_rn; ADD(pos); continue; }
		if (nws > 1 && CPT(pos + nws) != OUT) { pos += nws - 1; ADD(pos); continue; }
		if (nws > 0) { pos += nws; ADD(pos); continue; }
		ADD(++pos);
	}
#undef CPT
#undef FL
#undef ADD
	return nw;
}

/* ---- BPE (llm_tokenizer_bpe_session::tokenize for one word) ---- */
typedef struct { int prev, next; const char *t; uint32_t n; } sym_t;
typedef struct { int left, right, rank; uint32_t n; } bigram_t;   /* n: text length of the pair */
typedef struct { bigram_t *b; int n, cap; } heap_t;
static int bg_before(const bigram_t *a, const bigram_t *b) { return a->rank < b->rank || (a->rank == b->rank && a->left < b->left); }
static void hpush(heap_t *h, bigram_t x) {
	if (h->n == h->cap) { h->cap = h->cap ? 2 * h->cap : 64; h->b = realloc(h->b, sizeof *h->b * (size_t)h->cap); }
	int i = h->n++; h->b[i] = x;
	while (i > 0) { int p = (i - 1) / 2; if (!bg_before(&h->b[i], &h->b[p])) break; bigram_t t = h->b[i]; h->b[i] = h->b[p]; h->b[p] = t; i = p; }
}
static bigram_t hpop(heap_t *h) {
	bigram_t top = h->b[0]; h->b[0] = h->b[--h->n]; int i = 0;
	for (;;) {
		int l = 2 * i + 1, r = l + 1, m = i;
		if (l < h->n && bg_before(&h->b[l], &h->b[m])) m = l;
		if (r < h->n && bg_before(&h->b[r], &h->b[m])) m = r;
		if (m == i) break;
		bigram_t t = h->b[i]; h->b[i] = h->b[m]; h->b[m] = t; i = m;
	}
	return top;
}
static void add_bigram(const tok_vocab *v, sym_t *sy, heap_t *h, int l, int r, char *scratch) {
	if (l < 0 || r < 0) return;
	memcpy(scratch, sy[l].t, sy[l].n); scratch[sy[l].n] = ' '; memcpy(scratch + sy[l].n + 1, sy[r].t, sy[r].n);
	int32_t rank = ht_get(&v->ranks, scratch, sy[l].n + 1 + sy[r].n);
	if (rank < 0) return;
	hpush(h, (bigram_t){ l, r, rank, sy[l].n + sy[r].n });
}
static int utf8_len(uint8_t b) { static const int L[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 }; return L[b >> 4]; }

typedef struct { uint32_t *out; int n, cap; } outbuf;
static void emit(outbuf *o, uint32_t id) { if (o->n < o->cap) o->out[o->n] = id; o->n++; }

static void bpe_word(const tok_vocab *v, const char *w, uint32_t wn, outbuf *o) {
	/* (one copy a thread: flashd tokenizes requests on their own threads, FLASHD_CONCURRENT of them at once) */
	static _Thread_local sym_t *sy; static _Thread_local int sycap; static _Thread_local heap_t h;
	static _Thread_local char *scratch; static _Thread_local uint32_t scap;
	if ((int)wn > sycap) { sycap = (int)wn * 2; sy = realloc(sy, sizeof *sy * (size_t)sycap); }
	if (2 * wn + 2 > scap) { scap = 2 * wn + 2; scratch = realloc(scratch, scap); }
	int ns = 0; h.n = 0;
	for (uint32_t off = 0; off < wn;) {
		uint32_t cl = (uint32_t)utf8_len((uint8_t)w[off]); if (cl > wn - off) cl = wn - off;
		sy[ns] = (sym_t){ ns - 1, off + cl == wn ? -1 : ns + 1, w + off, cl }; ns++; off += cl;
	}
	for (int i = 1; i < ns; i++) add_bigram(v, sy, &h, i - 1, i, scratch);
	while (h.n) {
		bigram_t b = hpop(&h);
		sym_t *L = &sy[b.left], *R = &sy[b.right];
		if (L->n == 0 || R->n == 0 || L->n + R->n != b.n || L->next != b.right) continue;   /* outdated */
		L->n += R->n; R->n = 0; L->next = R->next;
		if (R->next >= 0) sy[R->next].prev = b.left;
		add_bigram(v, sy, &h, L->prev, b.left, scratch);
		add_bigram(v, sy, &h, b.left, L->next, scratch);
	}
	for (int i = 0; i != -1 && ns; i = sy[i].next) {
		if (!sy[i].n) continue;
		int32_t id = ht_get(&v->ids, sy[i].t, sy[i].n);
		if (id >= 0) { emit(o, (uint32_t)id); continue; }
		for (uint32_t j = 0; j < sy[i].n; j++) { int32_t b1 = ht_get(&v->ids, sy[i].t + j, 1); if (b1 >= 0) emit(o, (uint32_t)b1); }
	}
}

/* one raw fragment: split, byte-encode each word, BPE */
static void encode_raw(const tok_vocab *v, const char *s, size_t n, outbuf *o) {
	if (!n) return;
	uint32_t *cp = malloc(sizeof *cp * n); size_t *wl = malloc(sizeof *wl * n);
	size_t nc = cpts_from_utf8((const uint8_t *)s, n, cp), nw = split_qwen35(cp, nc, wl), at = 0;
	char *w = malloc(16 * n + 16);
	for (size_t k = 0; k < nw; k++) {
		uint32_t wn = 0; char u[4];
		for (size_t i = at; i < at + wl[k]; i++) {
			int ul = put_utf8(cp[i], u);
			for (int j = 0; j < ul; j++) { uint8_t b = (uint8_t)u[j]; memcpy(w + wn, v->byte_utf8[b], v->byte_len[b]); wn += v->byte_len[b]; }
		}
		at += wl[k];
		bpe_word(v, w, wn, o);
	}
	free(w); free(cp); free(wl);
}

int tok_encode(const tok_vocab *v, const char *text, size_t len, int parse_special, uint32_t *out, int cap) {
	outbuf o = { out, 0, cap };
	/* fragments: id >= 0 a special token, else raw text [off, off + n) */
	typedef struct { int id; size_t off, n; } frag;
	int nf = 1, fcap = 64; frag *f = malloc(sizeof *f * (size_t)fcap); f[0] = (frag){ -1, 0, len };
	if (v->add_bos) emit(&o, v->bos);
	for (int k = 0; k < v->nspecial; k++) {
		int id = v->special[k];
		if (!parse_special && (v->type[id] == TT_CONTROL || v->type[id] == TT_UNKNOWN)) continue;
		const char *st = v->text[id]; size_t sl = v->tlen[id];
		if (!sl) continue;
		for (int i = 0; i < nf; i++) {
			if (f[i].id >= 0) continue;
			const char *base = text + f[i].off; size_t bn = f[i].n;
			const char *m = bn >= sl ? memmem(base, bn, st, sl) : NULL;
			if (!m) continue;
			size_t left = (size_t)(m - base), rest = bn - left - sl;
			/* replace f[i] by [left] [special] [right]; the right part is searched next (i + 2) */
			if (nf + 2 > fcap) { fcap *= 2; f = realloc(f, sizeof *f * (size_t)fcap); }
			frag L = { -1, f[i].off, left }, S = { id, 0, 0 }, R = { -1, f[i].off + left + sl, rest };
			int add = (left > 0) + 1 + (rest > 0);
			memmove(f + i + add, f + i + 1, sizeof *f * (size_t)(nf - i - 1));
			int j = i; if (left > 0) f[j++] = L; f[j++] = S; if (rest > 0) f[j++] = R;
			nf += add - 1;
			i = j - 1 - (rest > 0);   /* continue with R (if any) on the next iteration */
		}
	}
	for (int i = 0; i < nf; i++) {
		if (f[i].id >= 0) emit(&o, (uint32_t)f[i].id);
		else encode_raw(v, text + f[i].off, f[i].n, &o);
	}
	free(f);
	return o.n <= cap ? o.n : -1;
}

int tok_piece(const tok_vocab *v, uint32_t id, char *buf, int cap) {
	if ((int)id >= v->n) return -1;
	const char *t = v->text[id]; uint32_t n = v->tlen[id]; int k = 0;
	if (v->type[id] != TT_NORMAL) { if ((int)n > cap) return -1; memcpy(buf, t, n); return (int)n; }
	uint32_t cp[256]; size_t nc = n <= 256 ? cpts_from_utf8((const uint8_t *)t, n, cp) : 0;
	for (size_t i = 0; i < nc; i++) {
		int b = cp[i] < 512 ? v->cpt_byte[cp[i]] : -1;
		if (b < 0) { char u[4]; int ul = put_utf8(cp[i], u); if (k + ul > cap) return -1; memcpy(buf + k, u, (size_t)ul); k += ul; continue; }
		if (k + 1 > cap) return -1;
		buf[k++] = (char)b;
	}
	return k;
}

int tok_count(const tok_vocab *v) { return v->n; }
int tok_find(const tok_vocab *v, const char *text) { return ht_get(&v->ids, text, (uint32_t)strlen(text)); }
uint32_t tok_eos(const tok_vocab *v) { return v->eos; }
uint32_t tok_bos(const tok_vocab *v) { return v->bos; }
int tok_add_bos(const tok_vocab *v) { return v->add_bos; }
const char *tok_template(const tok_vocab *v) { return v->tmpl; }
const char *tok_text(const tok_vocab *v, uint32_t id, size_t *len) { if ((int)id >= v->n) return NULL; if (len) *len = v->tlen[id]; return v->text[id]; }

static const tok_vocab *cmp_v;
static int by_len_desc(const void *a, const void *b) {
	int x = *(const int *)a, y = *(const int *)b;
	if (cmp_v->tlen[x] != cmp_v->tlen[y]) return cmp_v->tlen[x] > cmp_v->tlen[y] ? -1 : 1;
	return x < y ? -1 : x > y;
}

tok_vocab *tok_load(const char *path) {
	tok_vocab *v = calloc(1, sizeof *v);
	if (gguf_open(&v->g, path)) { free(v); return NULL; }
	uint64_t len; const char *s = gguf_get_str(&v->g, "tokenizer.ggml.model", &len);
	if (!s || len != 4 || memcmp(s, "gpt2", 4)) { fprintf(stderr, "%s: tokenizer.ggml.model is not gpt2\n", path); return NULL; }
	s = gguf_get_str(&v->g, "tokenizer.ggml.pre", &len);
	if (!s || len != 6 || memcmp(s, "qwen35", 6)) { fprintf(stderr, "%s: tokenizer.ggml.pre is %.*s, only qwen35 is implemented\n", path, s ? (int)len : 4, s ? s : "none"); return NULL; }
	uint32_t et; uint64_t n; const uint8_t *p;
	if (gguf_get_arr(&v->g, "tokenizer.ggml.tokens", &et, &n, &p) || et != 8) { fprintf(stderr, "%s: no tokenizer.ggml.tokens\n", path); return NULL; }
	v->n = (int)n; v->text = malloc(sizeof *v->text * n); v->tlen = malloc(sizeof *v->tlen * n); v->type = malloc(sizeof *v->type * n);
	ht_init(&v->ids, (uint32_t)n);
	for (uint64_t i = 0; i < n; i++) { uint64_t l; memcpy(&l, p, 8); v->text[i] = (const char *)p + 8; v->tlen[i] = (uint32_t)l; p += 8 + l; ht_put(&v->ids, v->text[i], v->tlen[i], (int32_t)i); }
	if (gguf_get_arr(&v->g, "tokenizer.ggml.token_type", &et, &n, &p) || et != 5 || (int)n != v->n) { fprintf(stderr, "%s: no tokenizer.ggml.token_type\n", path); return NULL; }
	memcpy(v->type, p, 4 * n);
	if (gguf_get_arr(&v->g, "tokenizer.ggml.merges", &et, &n, &p) || et != 8) { fprintf(stderr, "%s: no tokenizer.ggml.merges\n", path); return NULL; }
	ht_init(&v->ranks, (uint32_t)n);
	for (uint64_t i = 0; i < n; i++) { uint64_t l; memcpy(&l, p, 8); ht_put(&v->ranks, (const char *)p + 8, (uint32_t)l, (int32_t)i); p += 8 + l; }
	int64_t x;
	v->eos = gguf_get_int(&v->g, "tokenizer.ggml.eos_token_id", &x) ? 0xffffffffu : (uint32_t)x;
	v->bos = gguf_get_int(&v->g, "tokenizer.ggml.bos_token_id", &x) ? 0xffffffffu : (uint32_t)x;
	v->add_bos = !gguf_get_int(&v->g, "tokenizer.ggml.add_bos_token", &x) && x;
	s = gguf_get_str(&v->g, "tokenizer.chat_template", &len);
	if (s) { v->tmpl = malloc(len + 1); memcpy(v->tmpl, s, len); v->tmpl[len] = 0; }
	v->special = malloc(sizeof *v->special * (size_t)v->n);
	for (int i = 0; i < v->n; i++) if (v->type[i] == TT_CONTROL || v->type[i] == TT_USER || v->type[i] == TT_UNKNOWN) v->special[v->nspecial++] = i;
	cmp_v = v; qsort(v->special, (size_t)v->nspecial, sizeof *v->special, by_len_desc);
	/* GPT-2 bytes_to_unicode (llama.cpp unicode_byte_to_utf8_map) */
	for (int i = 0; i < 512; i++) v->cpt_byte[i] = -1;
	int extra = 0;
	for (int b = 0; b < 256; b++) {
		int keep = (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
		uint32_t c = keep ? (uint32_t)b : 256u + (uint32_t)extra++;
		v->byte_len[b] = (uint8_t)put_utf8(c, v->byte_utf8[b]); v->cpt_byte[c] = (int16_t)b;
	}
	return v;
}

int tok_is_special(const tok_vocab *v, uint32_t id) { return (int)id < v->n && v->type[id] != TT_NORMAL && v->type[id] != TT_BYTE; }

/* what the output path must not show: control and unknown tokens (llama.cpp suppresses UNKNOWN | CONTROL). USER_DEFINED tokens such as <tool_call> and
   </tool_call> are text the model wrote and are shown (issue #3); tok_is_special is the tokenizer-side notion and includes them. */
int tok_is_hidden(const tok_vocab *v, uint32_t id) { return (int)id < v->n && (v->type[id] == TT_CONTROL || v->type[id] == TT_UNKNOWN); }

int tok_meta_num(const tok_vocab *v, const char *key, double *out) {
	const gguf_kv *kv = gguf_get(&v->g, key);
	if (!kv) return -1;
	if (kv->type == 6) { float f; memcpy(&f, kv->val, 4); *out = f; return 0; }
	if (kv->type == 12) { memcpy(out, kv->val, 8); return 0; }
	int64_t i;
	if (gguf_get_int(&v->g, key, &i)) return -1;
	*out = (double)i; return 0;
}
