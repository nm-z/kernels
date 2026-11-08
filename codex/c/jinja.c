/* jinja.c: see jinja.h. Three stages: a scanner that turns the source into text / expression /
 * statement items (applying -, trim_blocks and lstrip_blocks), a recursive-descent parser that
 * builds statement and expression trees, and a tree-walking evaluator over the jv values of json.h. */
#include "jinja.h"
#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct expr expr;
typedef struct stmt stmt;

enum { E_LIT, E_NAME, E_LIST, E_TUPLE, E_DICT, E_ATTR, E_SUB, E_SLICE, E_CALL, E_FILTER, E_TEST, E_NEG, E_POS, E_NOT, E_BIN, E_COND, E_AND, E_OR };
enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_IDIV, OP_MOD, OP_POW, OP_CAT, OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE, OP_IN, OP_NOTIN };

struct expr {
	int k, op, neg;
	jv *lit;
	const char *name; size_t nlen;
	expr *a, *b, *c, *x;                 /* operands; E_SLICE: a[b:c:x]; E_COND: b if a else c */
	expr **args; int nargs;              /* E_CALL, E_FILTER, E_TEST arguments */
	const char **kn; size_t *kl; expr **kv; int nkw;   /* keyword arguments */
	expr **items; int nitems;            /* E_LIST, E_TUPLE; E_DICT: key, value, key, value ... */
};

enum { S_TEXT, S_OUT, S_IF, S_FOR, S_SET, S_MACRO, S_BREAK, S_CONTINUE };
struct stmt {
	int k; stmt *next;
	const char *text; size_t tlen;       /* S_TEXT */
	expr *e;                             /* S_OUT / S_SET value; S_IF condition; S_FOR iterable */
	expr *target;                        /* S_SET: name, attribute or tuple of names */
	expr *cond;                          /* S_FOR: the "if" filter */
	stmt *body, *els;
	const char **names; size_t *nlens; int nnames;   /* S_FOR targets; S_MACRO parameters */
	expr **defs;                         /* S_MACRO: default values (NULL: none) */
	const char *mname; size_t mlen;
};

struct jinja { arena *ar; stmt *body; char *src; jmp_buf jb; char err[256]; };

/* ================= scanner ================= */
typedef struct { int k; const char *s; size_t n; } item;   /* k: 0 text, 1 {{ }}, 2 {% %} */

static void cfail(jinja *t, const char *fmt, ...) {
	va_list ap; va_start(ap, fmt); vsnprintf(t->err, sizeof t->err, fmt, ap); va_end(ap);
	longjmp(t->jb, 1);
}
static int isws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

static const char *find_close(const char *p, const char *e, char kind) {
	char q = 0;
	for (; p + 1 < e; p++) {
		if (kind != '#') {
			if (q) { if (*p == '\\') p++; else if (*p == q) q = 0; continue; }
			if (*p == '"' || *p == '\'') { q = *p; continue; }
		}
		if (*p == kind && p[1] == '}') return p;
	}
	return NULL;
}

static item *scan(jinja *t, const char *src, size_t n, int *nout) {
	int cap = 64, ni = 0; item *it = malloc(sizeof *it * (size_t)cap);
	const char *e = src + n, *p = src, *text_start = src;
#define PUSH(kk, ss, nn) do { if (ni == cap) { cap *= 2; it = realloc(it, sizeof *it * (size_t)cap); } it[ni++] = (item){ kk, ss, nn }; } while (0)
	if (n && e[-1] == '\n') e--;   /* keep_trailing_newline off */
	for (;;) {
		const char *tag = NULL;
		for (const char *q = p; q + 1 < e; q++) if (*q == '{' && (q[1] == '{' || q[1] == '%' || q[1] == '#')) { tag = q; break; }
		if (!tag) { if (e > text_start) PUSH(0, text_start, (size_t)(e - text_start)); break; }
		char kind = tag[1];
		const char *tend = tag, *op = tag + 2; int dash_l = 0, plus = 0;
		if (op < e && *op == '-') { dash_l = 1; op++; } else if (kind == '%' && op < e && *op == '+') { plus = 1; op++; }
		if (dash_l) { while (tend > text_start && isws(tend[-1])) tend--; }
		else if (kind != '{' && !plus) {
			const char *b = tag; while (b > src && (b[-1] == ' ' || b[-1] == '\t')) b--;
			if ((b == src || b[-1] == '\n') && b >= text_start) tend = b;   /* lstrip_blocks: only blanks since the line start */
		}
		if (tend > text_start) PUSH(0, text_start, (size_t)(tend - text_start));
		const char *cl = find_close(op, e, kind == '{' ? '}' : kind == '%' ? '%' : '#');
		if (!cl) { free(it); cfail(t, "unterminated %s", kind == '{' ? "{{" : kind == '%' ? "{%" : "{#"); }
		int dash_r = cl > op && cl[-1] == '-';
		if (kind == '{') PUSH(1, op, (size_t)(cl - op - dash_r));
		else if (kind == '%') PUSH(2, op, (size_t)(cl - op - dash_r));
		p = cl + 2;
		if (dash_r) { while (p < e && isws(*p)) p++; }
		else if (kind != '{') { if (p < e && *p == '\n') p++; else if (p + 1 < e && p[0] == '\r' && p[1] == '\n') p += 2; }   /* trim_blocks */
		text_start = p;
	}
#undef PUSH
	*nout = ni;
	return it;
}

/* ================= expression parser ================= */
typedef struct { const char *p, *e; jinja *t; } ps;

static void skipws(ps *s) { while (s->p < s->e && isws(*s->p)) s->p++; }
static int idc(char c) { return isalnum((unsigned char)c) || c == '_'; }
static int peek_word(ps *s, const char *w) {
	skipws(s); size_t n = strlen(w);
	if ((size_t)(s->e - s->p) < n || memcmp(s->p, w, n)) return 0;
	return !(s->p + n < s->e && idc(s->p[n]));
}
static int accept_word(ps *s, const char *w) { if (!peek_word(s, w)) return 0; s->p += strlen(w); return 1; }
static int peek_op(ps *s, const char *o) { skipws(s); size_t n = strlen(o); return (size_t)(s->e - s->p) >= n && !memcmp(s->p, o, n); }
static int accept_op(ps *s, const char *o) { if (!peek_op(s, o)) return 0; s->p += strlen(o); return 1; }
static void expect_op(ps *s, const char *o) { if (!accept_op(s, o)) cfail(s->t, "expected '%s' before '%.20s'", o, s->p < s->e ? s->p : ""); }

static expr *mk(ps *s, int k) { expr *x = arena_alloc(s->t->ar, sizeof *x); x->k = k; return x; }
static expr *mk2(ps *s, int k, expr *a, expr *b) { expr *x = mk(s, k); x->a = a; x->b = b; return x; }
static expr *p_expr(ps *s);
static expr *p_or(ps *s);

/* an identifier at the cursor, or NULL */
static const char *ident(ps *s, size_t *n) {
	skipws(s);
	if (s->p >= s->e || !(isalpha((unsigned char)*s->p) || *s->p == '_')) return NULL;
	const char *b = s->p; while (s->p < s->e && idc(*s->p)) s->p++;
	*n = (size_t)(s->p - b);
	return b;
}

static jv *p_string(ps *s) {
	char q = *s->p++; sbuf b = {0};
	for (;;) {
		if (s->p >= s->e) cfail(s->t, "unterminated string");
		char c = *s->p++;
		if (c == q) break;
		if (c != '\\') { sb_putc(&b, c); continue; }
		if (s->p >= s->e) cfail(s->t, "unterminated string");
		c = *s->p++;
		switch (c) {
		case 'n': sb_putc(&b, '\n'); break; case 't': sb_putc(&b, '\t'); break; case 'r': sb_putc(&b, '\r'); break;
		case 'b': sb_putc(&b, '\b'); break; case 'f': sb_putc(&b, '\f'); break; case '0': sb_putc(&b, 0); break;
		case 'x': {
			unsigned v = 0; for (int i = 0; i < 2 && s->p < s->e && isxdigit((unsigned char)*s->p); i++) v = v * 16 + (unsigned)(isdigit((unsigned char)*s->p) ? *s->p - '0' : (tolower(*s->p) - 'a' + 10)), s->p++;
			sb_putc(&b, (char)v); break;
		}
		case 'u': {
			unsigned v = 0; for (int i = 0; i < 4 && s->p < s->e && isxdigit((unsigned char)*s->p); i++) v = v * 16 + (unsigned)(isdigit((unsigned char)*s->p) ? *s->p - '0' : (tolower(*s->p) - 'a' + 10)), s->p++;
			char u[4]; int ul;
			if (v < 0x80) { u[0] = (char)v; ul = 1; } else if (v < 0x800) { u[0] = (char)(0xc0 | (v >> 6)); u[1] = (char)(0x80 | (v & 0x3f)); ul = 2; }
			else { u[0] = (char)(0xe0 | (v >> 12)); u[1] = (char)(0x80 | ((v >> 6) & 0x3f)); u[2] = (char)(0x80 | (v & 0x3f)); ul = 3; }
			sb_put(&b, u, (size_t)ul); break;
		}
		default: sb_putc(&b, c);   /* \\ \' \" and unknown escapes */
		}
	}
	jv *v = jv_str(s->t->ar, b.s ? b.s : "", b.n); sb_free(&b);
	return v;
}

static void p_args(ps *s, expr *x) {   /* after '(' ... through ')' */
	int cap = 0;
	for (;;) {
		skipws(s);
		if (accept_op(s, ")")) return;
		size_t kl = 0; const char *save = s->p; const char *kn = ident(s, &kl);
		if (kn && peek_op(s, "=") && !peek_op(s, "==")) {
			s->p++;
			expr *v = p_expr(s);
			x->kn = realloc(x->kn, sizeof *x->kn * (size_t)(x->nkw + 1)); x->kl = realloc(x->kl, sizeof *x->kl * (size_t)(x->nkw + 1)); x->kv = realloc(x->kv, sizeof *x->kv * (size_t)(x->nkw + 1));
			x->kn[x->nkw] = arena_strndup(s->t->ar, kn, kl); x->kl[x->nkw] = kl; x->kv[x->nkw] = v; x->nkw++;
		} else {
			s->p = save;
			expr *v = p_expr(s);
			if (x->nargs == cap) { cap = cap ? 2 * cap : 4; x->args = realloc(x->args, sizeof *x->args * (size_t)cap); }
			x->args[x->nargs++] = v;
		}
		if (accept_op(s, ",")) continue;
		expect_op(s, ")"); return;
	}
}

static expr *p_primary(ps *s) {
	skipws(s);
	if (s->p >= s->e) cfail(s->t, "unexpected end of expression");
	char c = *s->p;
	if (c == '"' || c == '\'') { expr *x = mk(s, E_LIT); x->lit = p_string(s); return x; }
	if (isdigit((unsigned char)c)) {
		const char *b = s->p; int flt = 0;
		while (s->p < s->e && isdigit((unsigned char)*s->p)) s->p++;
		if (s->p + 1 < s->e && *s->p == '.' && isdigit((unsigned char)s->p[1])) { flt = 1; s->p++; while (s->p < s->e && isdigit((unsigned char)*s->p)) s->p++; }
		char tmp[64]; size_t n = (size_t)(s->p - b); if (n >= sizeof tmp) cfail(s->t, "number too long");
		memcpy(tmp, b, n); tmp[n] = 0;
		expr *x = mk(s, E_LIT);
		if (flt) { x->lit = jv_new(s->t->ar, J_NUM); x->lit->d = strtod(tmp, NULL); } else x->lit = jv_int(s->t->ar, strtoll(tmp, NULL, 10));
		return x;
	}
	if (c == '[') {
		s->p++; expr *x = mk(s, E_LIST);
		for (;;) {
			if (accept_op(s, "]")) break;
			expr *v = p_expr(s);
			x->items = realloc(x->items, sizeof *x->items * (size_t)(x->nitems + 1)); x->items[x->nitems++] = v;
			if (accept_op(s, ",")) continue;
			expect_op(s, "]"); break;
		}
		return x;
	}
	if (c == '(') {
		s->p++;
		if (accept_op(s, ")")) return mk(s, E_TUPLE);
		expr *first = p_expr(s);
		if (accept_op(s, ")")) return first;
		expr *x = mk(s, E_TUPLE); x->items = malloc(sizeof *x->items); x->items[0] = first; x->nitems = 1;
		while (accept_op(s, ",")) {
			if (peek_op(s, ")")) break;
			expr *v = p_expr(s);
			x->items = realloc(x->items, sizeof *x->items * (size_t)(x->nitems + 1)); x->items[x->nitems++] = v;
		}
		expect_op(s, ")");
		return x;
	}
	if (c == '{') {
		s->p++; expr *x = mk(s, E_DICT);
		for (;;) {
			if (accept_op(s, "}")) break;
			expr *k = p_expr(s); expect_op(s, ":"); expr *v = p_expr(s);
			x->items = realloc(x->items, sizeof *x->items * (size_t)(x->nitems + 2)); x->items[x->nitems++] = k; x->items[x->nitems++] = v;
			if (accept_op(s, ",")) continue;
			expect_op(s, "}"); break;
		}
		return x;
	}
	size_t n; const char *id = ident(s, &n);
	if (!id) cfail(s->t, "unexpected '%.20s'", s->p);
	expr *x = mk(s, E_LIT);
	if ((n == 4 && (!memcmp(id, "true", 4) || !memcmp(id, "True", 4)))) x->lit = jv_bool(s->t->ar, 1);
	else if ((n == 5 && (!memcmp(id, "false", 5) || !memcmp(id, "False", 5)))) x->lit = jv_bool(s->t->ar, 0);
	else if ((n == 4 && (!memcmp(id, "none", 4) || !memcmp(id, "None", 4)))) x->lit = jv_new(s->t->ar, J_NULL);
	else { x->k = E_NAME; x->name = arena_strndup(s->t->ar, id, n); x->nlen = n; }
	return x;
}

static expr *p_postfix(ps *s) {
	expr *x = p_primary(s);
	for (;;) {
		skipws(s);
		if (s->p >= s->e) return x;
		char c = *s->p;
		if (c == '.' && s->p + 1 < s->e && (isalpha((unsigned char)s->p[1]) || s->p[1] == '_')) {
			s->p++; size_t n; const char *id = ident(s, &n);
			expr *y = mk2(s, E_ATTR, x, NULL); y->name = arena_strndup(s->t->ar, id, n); y->nlen = n; x = y;
		} else if (c == '[') {
			s->p++;
			expr *start = NULL, *stop = NULL, *step = NULL; int slice = 0;
			if (!peek_op(s, ":")) start = p_expr(s);
			if (accept_op(s, ":")) {
				slice = 1;
				if (!peek_op(s, ":") && !peek_op(s, "]")) stop = p_expr(s);
				if (accept_op(s, ":")) { if (!peek_op(s, "]")) step = p_expr(s); }
			}
			expect_op(s, "]");
			if (slice) { expr *y = mk(s, E_SLICE); y->a = x; y->b = start; y->c = stop; y->x = step; x = y; }
			else x = mk2(s, E_SUB, x, start);
		} else if (c == '(') {
			s->p++; expr *y = mk2(s, E_CALL, x, NULL); p_args(s, y); x = y;
		} else if (c == '|') {
			s->p++; size_t n; const char *id = ident(s, &n); if (!id) cfail(s->t, "filter name expected");
			expr *y = mk2(s, E_FILTER, x, NULL); y->name = arena_strndup(s->t->ar, id, n); y->nlen = n;
			if (peek_op(s, "(")) { s->p++; p_args(s, y); }
			x = y;
		} else if (peek_word(s, "is")) {
			s->p += 2; int neg = accept_word(s, "not");
			size_t n; const char *id = ident(s, &n); if (!id) cfail(s->t, "test name expected");
			expr *y = mk2(s, E_TEST, x, NULL); y->neg = neg; y->name = arena_strndup(s->t->ar, id, n); y->nlen = n;
			if (peek_op(s, "(")) { s->p++; p_args(s, y); }
			x = y;
		} else return x;
	}
}

static expr *p_unary(ps *s) {
	if (accept_op(s, "-")) return mk2(s, E_NEG, p_unary(s), NULL);
	if (accept_op(s, "+")) return mk2(s, E_POS, p_unary(s), NULL);
	return p_postfix(s);
}
static expr *p_pow(ps *s) {
	expr *l = p_unary(s);
	if (accept_op(s, "**")) { expr *x = mk2(s, E_BIN, l, p_pow(s)); x->op = OP_POW; return x; }
	return l;
}
static expr *p_mul(ps *s) {
	expr *l = p_pow(s);
	for (;;) {
		int op;
		if (peek_op(s, "//")) { s->p += 2; op = OP_IDIV; }
		else if (peek_op(s, "/")) { s->p++; op = OP_DIV; }
		else if (peek_op(s, "*") && !peek_op(s, "**")) { s->p++; op = OP_MUL; }
		else if (peek_op(s, "%")) { s->p++; op = OP_MOD; }
		else return l;
		expr *x = mk2(s, E_BIN, l, p_pow(s)); x->op = op; l = x;
	}
}
static expr *p_cat(ps *s) {
	expr *l = p_mul(s);
	while (peek_op(s, "~")) { s->p++; expr *x = mk2(s, E_BIN, l, p_mul(s)); x->op = OP_CAT; l = x; }
	return l;
}
static expr *p_add(ps *s) {
	expr *l = p_cat(s);
	for (;;) {
		int op;
		if (peek_op(s, "+")) op = OP_ADD; else if (peek_op(s, "-")) op = OP_SUB; else return l;
		s->p++;
		expr *x = mk2(s, E_BIN, l, p_cat(s)); x->op = op; l = x;
	}
}
static expr *p_cmp(ps *s) {
	expr *l = p_add(s);
	for (;;) {
		int op;
		if (peek_op(s, "==")) { s->p += 2; op = OP_EQ; }
		else if (peek_op(s, "!=")) { s->p += 2; op = OP_NE; }
		else if (peek_op(s, "<=")) { s->p += 2; op = OP_LE; }
		else if (peek_op(s, ">=")) { s->p += 2; op = OP_GE; }
		else if (peek_op(s, "<")) { s->p++; op = OP_LT; }
		else if (peek_op(s, ">")) { s->p++; op = OP_GT; }
		else if (accept_word(s, "in")) op = OP_IN;
		else if (peek_word(s, "not")) {
			const char *save = s->p; s->p += 3;
			if (accept_word(s, "in")) op = OP_NOTIN; else { s->p = save; return l; }
		} else return l;
		expr *x = mk2(s, E_BIN, l, p_add(s)); x->op = op; l = x;
	}
}
static expr *p_not(ps *s) {
	if (accept_word(s, "not")) return mk2(s, E_NOT, p_not(s), NULL);
	return p_cmp(s);
}
static expr *p_and(ps *s) {
	expr *l = p_not(s);
	while (accept_word(s, "and")) l = mk2(s, E_AND, l, p_not(s));
	return l;
}
static expr *p_or(ps *s) {
	expr *l = p_and(s);
	while (accept_word(s, "or")) l = mk2(s, E_OR, l, p_and(s));
	return l;
}
static expr *p_expr(ps *s) {
	expr *v = p_or(s);
	if (accept_word(s, "if")) {
		expr *c = p_or(s), *o = NULL;
		if (accept_word(s, "else")) o = p_expr(s);
		expr *x = mk(s, E_COND); x->a = c; x->b = v; x->c = o; return x;
	}
	return v;
}
static void p_end(ps *s) { skipws(s); if (s->p < s->e) cfail(s->t, "unexpected '%.20s'", s->p); }
static expr *parse_whole(jinja *t, const char *src, size_t n) { ps s = { src, src + n, t }; expr *x = p_expr(&s); p_end(&s); return x; }

/* ================= statement parser ================= */
typedef struct { jinja *t; item *it; int n, i; const char *tk; const char *rest; size_t restn; } sp;

static stmt *smk(sp *P, int k) { stmt *s = arena_alloc(P->t->ar, sizeof *s); s->k = k; return s; }

/* the keyword of a {% %} source and the text after it */
static const char *keyword(const item *b, size_t *kn, const char **rest, size_t *restn) {
	const char *p = b->s, *e = b->s + b->n;
	while (p < e && isws(*p)) p++;
	const char *k = p; while (p < e && idc(*p)) p++;
	*kn = (size_t)(p - k); *rest = p; *restn = (size_t)(e - p);
	return k;
}

static stmt *plist(sp *P, const char *const *terms);

static stmt *parse_if(sp *P, const char *rest, size_t restn) {
	stmt *s = smk(P, S_IF);
	s->e = parse_whole(P->t, rest, restn);
	static const char *const T[] = { "elif", "else", "endif", NULL };
	s->body = plist(P, T);
	if (!strcmp(P->tk, "elif")) { s->els = parse_if(P, P->rest, P->restn); return s; }
	if (!strcmp(P->tk, "else")) { static const char *const E[] = { "endif", NULL }; s->els = plist(P, E); }
	return s;
}

static stmt *plist(sp *P, const char *const *terms) {
	stmt *head = NULL, **tail = &head;
	while (P->i < P->n) {
		const item *b = &P->it[P->i++];
		stmt *s = NULL;
		if (b->k == 0) { s = smk(P, S_TEXT); s->text = b->s; s->tlen = b->n; }
		else if (b->k == 1) { s = smk(P, S_OUT); s->e = parse_whole(P->t, b->s, b->n); }
		else {
			size_t kn, restn; const char *rest; const char *k = keyword(b, &kn, &rest, &restn);
			for (int j = 0; terms && terms[j]; j++)
				if (strlen(terms[j]) == kn && !memcmp(k, terms[j], kn)) { P->tk = terms[j]; P->rest = rest; P->restn = restn; return head; }
			if (kn == 2 && !memcmp(k, "if", 2)) s = parse_if(P, rest, restn);
			else if (kn == 3 && !memcmp(k, "for", 3)) {
				s = smk(P, S_FOR);
				ps q = { rest, rest + restn, P->t };
				for (;;) {
					size_t n; const char *id = ident(&q, &n); if (!id) cfail(P->t, "loop variable expected");
					s->names = realloc(s->names, sizeof *s->names * (size_t)(s->nnames + 1)); s->nlens = realloc(s->nlens, sizeof *s->nlens * (size_t)(s->nnames + 1));
					s->names[s->nnames] = arena_strndup(P->t->ar, id, n); s->nlens[s->nnames++] = n;
					if (!accept_op(&q, ",")) break;
				}
				if (!accept_word(&q, "in")) cfail(P->t, "'in' expected in for");
				s->e = p_or(&q);
				if (accept_word(&q, "if")) s->cond = p_or(&q);
				p_end(&q);
				static const char *const T[] = { "else", "endfor", NULL };
				s->body = plist(P, T);
				if (!strcmp(P->tk, "else")) { static const char *const E[] = { "endfor", NULL }; s->els = plist(P, E); }
			} else if (kn == 3 && !memcmp(k, "set", 3)) {
				s = smk(P, S_SET);
				ps q = { rest, rest + restn, P->t };
				expr *tg = p_postfix(&q);
				if (peek_op(&q, ",")) {   /* a, b = ... */
					expr *tp = mk2(&q, E_TUPLE, NULL, NULL); tp->items = malloc(sizeof *tp->items); tp->items[0] = tg; tp->nitems = 1;
					while (accept_op(&q, ",")) { tp->items = realloc(tp->items, sizeof *tp->items * (size_t)(tp->nitems + 1)); tp->items[tp->nitems++] = p_postfix(&q); }
					tg = tp;
				}
				s->target = tg;
				if (accept_op(&q, "=") ) { s->e = p_expr(&q); p_end(&q); }
				else {   /* {% set x %} ... {% endset %} */
					p_end(&q);
					static const char *const E[] = { "endset", NULL };
					s->body = plist(P, E);
				}
			} else if (kn == 5 && !memcmp(k, "macro", 5)) {
				s = smk(P, S_MACRO);
				ps q = { rest, rest + restn, P->t };
				size_t n; const char *id = ident(&q, &n); if (!id) cfail(P->t, "macro name expected");
				s->mname = arena_strndup(P->t->ar, id, n); s->mlen = n;
				expect_op(&q, "(");
				for (;;) {
					if (accept_op(&q, ")")) break;
					size_t pn; const char *pid = ident(&q, &pn); if (!pid) cfail(P->t, "parameter name expected");
					s->names = realloc(s->names, sizeof *s->names * (size_t)(s->nnames + 1)); s->nlens = realloc(s->nlens, sizeof *s->nlens * (size_t)(s->nnames + 1)); s->defs = realloc(s->defs, sizeof *s->defs * (size_t)(s->nnames + 1));
					s->names[s->nnames] = arena_strndup(P->t->ar, pid, pn); s->nlens[s->nnames] = pn; s->defs[s->nnames] = NULL;
					if (accept_op(&q, "=")) s->defs[s->nnames] = p_expr(&q);
					s->nnames++;
					if (accept_op(&q, ",")) continue;
					expect_op(&q, ")"); break;
				}
				p_end(&q);
				static const char *const E[] = { "endmacro", NULL };
				s->body = plist(P, E);
			} else if (kn == 5 && !memcmp(k, "break", 5)) s = smk(P, S_BREAK);
			else if (kn == 8 && !memcmp(k, "continue", 8)) s = smk(P, S_CONTINUE);
			else cfail(P->t, "unsupported or unexpected tag '%.*s'", (int)kn, k);
		}
		*tail = s; tail = &s->next;
	}
	if (terms) cfail(P->t, "missing {%% %s %%}", terms[0]);
	return head;
}

jinja *jinja_compile(const char *src, size_t n, char *err, size_t errcap) {
	jinja *t = calloc(1, sizeof *t); t->ar = arena_new(); t->src = malloc(n + 1); memcpy(t->src, src, n); t->src[n] = 0;
	item *volatile it = NULL;
	if (setjmp(t->jb)) { if (err) snprintf(err, errcap, "%s", t->err); free(it); jinja_free(t); return NULL; }
	int ni; it = scan(t, t->src, n, &ni);
	sp P = { t, it, ni, 0, NULL, NULL, 0 };
	t->body = plist(&P, NULL);
	free(it);
	return t;
}
void jinja_free(jinja *t) { if (!t) return; arena_free(t->ar); free(t->src); free(t); }

/* ================= evaluator ================= */
typedef struct scope { const char **nm; size_t *nl; jv **val; int n, cap; struct scope *up; } scope;
typedef struct { arena *a; jmp_buf jb; char err[512]; int brk, cont, depth; } rc;

static void rfail(rc *c, const char *fmt, ...) __attribute__((noreturn, format(printf, 2, 3)));
static void rfail(rc *c, const char *fmt, ...) {
	va_list ap; va_start(ap, fmt); vsnprintf(c->err, sizeof c->err, fmt, ap); va_end(ap);
	longjmp(c->jb, 1);
}
static scope *scope_new(rc *c, scope *up) { scope *s = arena_alloc(c->a, sizeof *s); s->up = up; return s; }
static jv *scope_find(scope *s, const char *nm, size_t nl) {
	for (; s; s = s->up) for (int i = s->n - 1; i >= 0; i--) if (s->nl[i] == nl && !memcmp(s->nm[i], nm, nl)) return s->val[i];
	return NULL;
}
static void scope_set(rc *c, scope *s, const char *nm, size_t nl, jv *v) {
	for (int i = 0; i < s->n; i++) if (s->nl[i] == nl && !memcmp(s->nm[i], nm, nl)) { s->val[i] = v; return; }
	if (s->n == s->cap) {
		int cap = s->cap ? 2 * s->cap : 8;
		const char **a = arena_alloc(c->a, sizeof *a * (size_t)cap); size_t *l = arena_alloc(c->a, sizeof *l * (size_t)cap); jv **vv = arena_alloc(c->a, sizeof *vv * (size_t)cap);
		if (s->n) { memcpy(a, s->nm, sizeof *a * (size_t)s->n); memcpy(l, s->nl, sizeof *l * (size_t)s->n); memcpy(vv, s->val, sizeof *vv * (size_t)s->n); }
		s->nm = a; s->nl = l; s->val = vv; s->cap = cap;
	}
	s->nm[s->n] = nm; s->nl[s->n] = nl; s->val[s->n] = v; s->n++;
}

static jv *undef(rc *c) { return jv_new(c->a, J_UNDEF); }
static jv *mkbool(rc *c, int b) { return jv_bool(c->a, b); }
static int is_num(const jv *v) { return v->k == J_INT || v->k == J_NUM || v->k == J_BOOL; }
static double as_d(const jv *v) { return v->k == J_NUM ? v->d : v->k == J_BOOL ? v->b : (double)v->i; }
static long long as_i(const jv *v) { return v->k == J_BOOL ? v->b : v->k == J_INT ? v->i : (long long)v->d; }
static int is_seq(const jv *v) { return v->k == J_LIST || v->k == J_TUPLE; }
static int is_map(const jv *v) { return v->k == J_DICT || v->k == J_NS; }

static int truthy(const jv *v) {
	switch (v->k) {
	case J_UNDEF: case J_NULL: return 0;
	case J_BOOL: return v->b;
	case J_INT: return v->i != 0;
	case J_NUM: return v->d != 0;
	case J_STR: return v->s.n > 0;
	case J_LIST: case J_TUPLE: return v->l.n > 0;
	case J_DICT: case J_NS: return v->o.n > 0;
	default: return 1;
	}
}

static void numstr(sbuf *b, double d) {
	char t[40]; snprintf(t, sizeof t, "%.15g", d);
	if (strtod(t, NULL) != d) snprintf(t, sizeof t, "%.17g", d);
	sb_puts(b, t);
	if (!strpbrk(t, ".eEn")) sb_puts(b, ".0");
}
static void put_repr(sbuf *b, const jv *v);
static void put_str(sbuf *b, const jv *v) {
	switch (v->k) {
	case J_UNDEF: break;
	case J_NULL: sb_puts(b, "None"); break;
	case J_BOOL: sb_puts(b, v->b ? "True" : "False"); break;
	case J_INT: sb_printf(b, "%lld", v->i); break;
	case J_NUM: numstr(b, v->d); break;
	case J_STR: sb_put(b, v->s.s, v->s.n); break;
	case J_MACRO: case J_FUNC: sb_puts(b, "<function>"); break;
	default: put_repr(b, v);
	}
}
static void put_repr(sbuf *b, const jv *v) {
	switch (v->k) {
	case J_STR: {
		char q = memchr(v->s.s, '\'', v->s.n) && !memchr(v->s.s, '"', v->s.n) ? '"' : '\'';
		sb_putc(b, q);
		for (size_t i = 0; i < v->s.n; i++) {
			char c = v->s.s[i];
			if (c == q || c == '\\') { sb_putc(b, '\\'); sb_putc(b, c); }
			else if (c == '\n') sb_puts(b, "\\n"); else if (c == '\t') sb_puts(b, "\\t"); else if (c == '\r') sb_puts(b, "\\r");
			else sb_putc(b, c);
		}
		sb_putc(b, q); break;
	}
	case J_LIST: case J_TUPLE:
		sb_putc(b, v->k == J_LIST ? '[' : '(');
		for (int i = 0; i < v->l.n; i++) { if (i) sb_puts(b, ", "); put_repr(b, v->l.a[i]); }
		if (v->k == J_TUPLE && v->l.n == 1) sb_putc(b, ',');
		sb_putc(b, v->k == J_LIST ? ']' : ')'); break;
	case J_DICT: case J_NS:
		sb_putc(b, '{');
		for (int i = 0; i < v->o.n; i++) { if (i) sb_puts(b, ", "); sb_putc(b, '\''); sb_put(b, v->o.key[i], v->o.klen[i]); sb_puts(b, "': "); put_repr(b, v->o.val[i]); }
		sb_putc(b, '}'); break;
	default: put_str(b, v);
	}
}
static jv *tostr(rc *c, const jv *v) { if (v->k == J_STR) return (jv *)v; sbuf b = {0}; put_str(&b, v); jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r; }

static int jeq(const jv *a, const jv *b) {
	if (is_num(a) && is_num(b)) return (a->k == J_NUM || b->k == J_NUM) ? as_d(a) == as_d(b) : as_i(a) == as_i(b);
	if (a->k != b->k && !(is_seq(a) && is_seq(b)) && !(is_map(a) && is_map(b))) return 0;
	switch (a->k) {
	case J_UNDEF: case J_NULL: return 1;
	case J_STR: return a->s.n == b->s.n && !memcmp(a->s.s, b->s.s, a->s.n);
	case J_LIST: case J_TUPLE:
		if (a->l.n != b->l.n) return 0;
		for (int i = 0; i < a->l.n; i++) if (!jeq(a->l.a[i], b->l.a[i])) return 0;
		return 1;
	case J_DICT: case J_NS:
		if (a->o.n != b->o.n) return 0;
		for (int i = 0; i < a->o.n; i++) { jv *o = jv_get(b, a->o.key[i], a->o.klen[i]); if (!o || !jeq(a->o.val[i], o)) return 0; }
		return 1;
	default: return a == b;
	}
}
static int jcmp(rc *c, const jv *a, const jv *b) {
	if (is_num(a) && is_num(b)) { double x = as_d(a), y = as_d(b); return x < y ? -1 : x > y; }
	if (a->k == J_STR && b->k == J_STR) { size_t n = a->s.n < b->s.n ? a->s.n : b->s.n; int r = memcmp(a->s.s, b->s.s, n); return r ? (r < 0 ? -1 : 1) : (a->s.n < b->s.n ? -1 : a->s.n > b->s.n); }
	rfail(c, "cannot compare these values");
}
static int jin(rc *c, const jv *a, const jv *b) {
	if (b->k == J_STR) {
		if (a->k != J_STR) rfail(c, "'in <string>' needs a string");
		return a->s.n == 0 || (b->s.n >= a->s.n && memmem(b->s.s, b->s.n, a->s.s, a->s.n) != NULL);
	}
	if (is_seq(b)) { for (int i = 0; i < b->l.n; i++) if (jeq(a, b->l.a[i])) return 1; return 0; }
	if (is_map(b)) return a->k == J_STR && jv_get(b, a->s.s, a->s.n) != NULL;
	if (b->k == J_UNDEF || b->k == J_NULL) return 0;
	rfail(c, "argument of type is not iterable");
}

static size_t u8len(const char *s, size_t n) { size_t k = 0; for (size_t i = 0; i < n; i++) k += ((unsigned char)s[i] & 0xc0) != 0x80; return k; }

static jv *arith(rc *c, int op, jv *a, jv *b) {
	if (op == OP_ADD && a->k == J_STR && b->k == J_STR) { sbuf s = {0}; sb_put(&s, a->s.s, a->s.n); sb_put(&s, b->s.s, b->s.n); jv *r = jv_str(c->a, s.s, s.n); sb_free(&s); return r; }
	if (op == OP_ADD && is_seq(a) && is_seq(b)) { jv *r = jv_new(c->a, a->k); for (int i = 0; i < a->l.n; i++) jv_push(c->a, r, a->l.a[i]); for (int i = 0; i < b->l.n; i++) jv_push(c->a, r, b->l.a[i]); return r; }
	if (op == OP_MUL && a->k == J_STR && b->k == J_INT) { sbuf s = {0}; for (long long i = 0; i < b->i; i++) sb_put(&s, a->s.s, a->s.n); jv *r = jv_str(c->a, s.s ? s.s : "", s.n); sb_free(&s); return r; }
	if (!is_num(a) || !is_num(b)) rfail(c, "unsupported operand types for arithmetic");
	int ints = a->k != J_NUM && b->k != J_NUM;
	if (ints && op != OP_DIV && op != OP_POW) {
		long long x = as_i(a), y = as_i(b), r;
		switch (op) {
		case OP_ADD: r = x + y; break; case OP_SUB: r = x - y; break; case OP_MUL: r = x * y; break;
		case OP_IDIV: if (!y) rfail(c, "division by zero"); r = x / y; if ((x % y) && ((x < 0) != (y < 0))) r--; break;
		case OP_MOD: if (!y) rfail(c, "modulo by zero"); r = x % y; if (r && ((r < 0) != (y < 0))) r += y; break;
		default: rfail(c, "bad operator");
		}
		return jv_int(c->a, r);
	}
	double x = as_d(a), y = as_d(b), r;
	switch (op) {
	case OP_ADD: r = x + y; break; case OP_SUB: r = x - y; break; case OP_MUL: r = x * y; break;
	case OP_DIV: if (y == 0) rfail(c, "division by zero"); r = x / y; break;
	case OP_IDIV: if (y == 0) rfail(c, "division by zero"); r = floor(x / y); break;
	case OP_MOD: if (y == 0) rfail(c, "modulo by zero"); r = fmod(x, y); if (r != 0 && ((r < 0) != (y < 0))) r += y; break;
	case OP_POW: r = pow(x, y); if (ints && y >= 0) return jv_int(c->a, (long long)r); break;
	default: rfail(c, "bad operator");
	}
	jv *v = jv_new(c->a, J_NUM); v->d = r; return v;
}

/* ---- values from iteration ---- */
static void chars_of(rc *c, jv *s, jv *out) {
	for (size_t i = 0; i < s->s.n;) { size_t j = i + 1; while (j < s->s.n && ((unsigned char)s->s.s[j] & 0xc0) == 0x80) j++; jv_push(c->a, out, jv_str(c->a, s->s.s + i, j - i)); i = j; }
}
static jv *as_list(rc *c, jv *v) {   /* the items a for loop or list() visits */
	if (is_seq(v)) return v;
	jv *l = jv_new(c->a, J_LIST);
	if (is_map(v)) for (int i = 0; i < v->o.n; i++) jv_push(c->a, l, jv_str(c->a, v->o.key[i], v->o.klen[i]));
	else if (v->k == J_STR) chars_of(c, v, l);
	else if (v->k != J_UNDEF && v->k != J_NULL) rfail(c, "value is not iterable");
	return l;
}

/* ---- evaluation ---- */
static jv *ev(rc *c, scope *sc, const expr *x);
static void exec_list(rc *c, scope *sc, const stmt *s, sbuf *out);

enum { F_NAMESPACE = 1, F_RAISE, F_RANGE, F_STRFTIME, F_DICT };
static int builtin_id(const char *n, size_t l) {
	if (l == 9 && !memcmp(n, "namespace", 9)) return F_NAMESPACE;
	if (l == 15 && !memcmp(n, "raise_exception", 15)) return F_RAISE;
	if (l == 5 && !memcmp(n, "range", 5)) return F_RANGE;
	if (l == 12 && !memcmp(n, "strftime_now", 12)) return F_STRFTIME;
	if (l == 4 && !memcmp(n, "dict", 4)) return F_DICT;
	return 0;
}

typedef struct { jv **a; int n; const char **kn; size_t *kl; jv **kv; int nk; } argv_t;
static jv *kwarg(const argv_t *g, const char *name) { size_t l = strlen(name); for (int i = 0; i < g->nk; i++) if (g->kl[i] == l && !memcmp(g->kn[i], name, l)) return g->kv[i]; return NULL; }
static jv *arg_at(const argv_t *g, int i, const char *name) { if (i < g->n) return g->a[i]; return name ? kwarg(g, name) : NULL; }

static jv *call_macro(rc *c, jv *m, const argv_t *g) {
	const stmt *s = m->f.p;
	if (++c->depth > 100) rfail(c, "macro recursion too deep");
	scope *ms = scope_new(c, (scope *)m->f.sc);
	if (g->n > s->nnames) rfail(c, "too many arguments for macro '%.*s'", (int)s->mlen, s->mname);
	for (int i = 0; i < s->nnames; i++) {
		jv *val = i < g->n ? g->a[i] : NULL;
		for (int j = 0; !val && j < g->nk; j++) if (g->kl[j] == s->nlens[i] && !memcmp(g->kn[j], s->names[i], g->kl[j])) val = g->kv[j];
		if (!val) val = s->defs[i] ? ev(c, ms, s->defs[i]) : undef(c);
		scope_set(c, ms, s->names[i], s->nlens[i], val);
	}
	sbuf b = {0};
	exec_list(c, ms, s->body, &b);
	jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b);
	c->depth--;
	return r;
}

static jv *call_builtin(rc *c, int id, const argv_t *g) {
	switch (id) {
	case F_NAMESPACE: {
		jv *ns = jv_new(c->a, J_NS);
		if (g->n && is_map(g->a[0])) for (int i = 0; i < g->a[0]->o.n; i++) jv_set(c->a, ns, g->a[0]->o.key[i], g->a[0]->o.klen[i], g->a[0]->o.val[i]);
		for (int i = 0; i < g->nk; i++) jv_set(c->a, ns, g->kn[i], g->kl[i], g->kv[i]);
		return ns;
	}
	case F_RAISE: { sbuf b = {0}; if (g->n) put_str(&b, g->a[0]); snprintf(c->err, sizeof c->err, "%s", b.s ? b.s : "template error"); sb_free(&b); longjmp(c->jb, 1); }
	case F_RANGE: {
		long long lo = 0, hi = 0, st = 1;
		if (g->n == 1) hi = as_i(g->a[0]); else if (g->n >= 2) { lo = as_i(g->a[0]); hi = as_i(g->a[1]); if (g->n >= 3) st = as_i(g->a[2]); }
		if (!st) rfail(c, "range() step is zero");
		jv *l = jv_new(c->a, J_LIST);
		for (long long i = lo; st > 0 ? i < hi : i > hi; i += st) { if (l->l.n > 100000) rfail(c, "range too large"); jv_push(c->a, l, jv_int(c->a, i)); }
		return l;
	}
	case F_STRFTIME: {
		time_t tt = time(NULL); struct tm tm; localtime_r(&tt, &tm); char buf[128];
		size_t n = strftime(buf, sizeof buf, g->n ? (const char *)g->a[0]->s.s : "%Y-%m-%d", &tm);
		return jv_str(c->a, buf, n);
	}
	case F_DICT: {
		jv *d = jv_new(c->a, J_DICT);
		for (int i = 0; i < g->nk; i++) jv_set(c->a, d, g->kn[i], g->kl[i], g->kv[i]);
		return d;
	}
	}
	rfail(c, "unknown function");
}

static void strip_range(const char *s, size_t n, const jv *chars, int left, int right, size_t *b, size_t *e) {
	size_t i = 0, j = n;
#define IN_SET(ch) (chars ? memchr(chars->s.s, (ch), chars->s.n) != NULL : isws(ch))
	if (left) while (i < j && IN_SET(s[i])) i++;
	if (right) while (j > i && IN_SET(s[j - 1])) j--;
#undef IN_SET
	*b = i; *e = j;
}

/* methods of str / list / dict; returns NULL if there is no such method */
static jv *call_method(rc *c, jv *o, const char *nm, size_t nl, const argv_t *g) {
#define IS(s) (nl == sizeof(s) - 1 && !memcmp(nm, s, nl))
	if (o->k == J_STR) {
		const char *s = o->s.s; size_t n = o->s.n;
		if (IS("strip") || IS("lstrip") || IS("rstrip")) {
			jv *ch = g->n ? g->a[0] : NULL; if (ch && ch->k != J_STR) ch = NULL;
			size_t b, e; strip_range(s, n, ch, nm[0] != 'r', nm[0] != 'l', &b, &e);
			return jv_str(c->a, s + b, e - b);
		}
		if (IS("startswith") || IS("endswith")) {
			int st = nm[0] == 's'; jv *p = g->n ? g->a[0] : undef(c); int ok = 0;
			int np = p->k == J_STR ? 1 : is_seq(p) ? p->l.n : 0;
			for (int i = 0; i < np; i++) {
				jv *q = p->k == J_STR ? p : p->l.a[i]; if (q->k != J_STR) continue;
				if (q->s.n <= n && !memcmp(st ? s : s + n - q->s.n, q->s.s, q->s.n)) ok = 1;
			}
			return mkbool(c, ok);
		}
		if (IS("upper") || IS("lower")) { char *t = arena_strndup(c->a, s, n); for (size_t i = 0; i < n; i++) t[i] = (char)(nm[0] == 'u' ? toupper((unsigned char)t[i]) : tolower((unsigned char)t[i])); return jv_strref(c->a, t, n); }
		if (IS("capitalize") || IS("title")) {
			char *t = arena_strndup(c->a, s, n); int word = 1;
			for (size_t i = 0; i < n; i++) {
				if (isalpha((unsigned char)t[i])) { t[i] = (char)(word || (nm[0] == 'c' && i == 0) ? toupper((unsigned char)t[i]) : tolower((unsigned char)t[i])); if (nm[0] == 'c') word = 0; else word = 0; }
				else if (nm[0] == 't') word = 1;
				if (nm[0] == 'c' && i == 0) word = 0;
			}
			return jv_strref(c->a, t, n);
		}
		if (IS("replace")) {
			if (g->n < 2 || g->a[0]->k != J_STR || g->a[1]->k != J_STR) rfail(c, "replace() takes two strings");
			jv *from = g->a[0], *to = g->a[1]; long long cnt = g->n > 2 ? as_i(g->a[2]) : -1; sbuf b = {0};
			if (!from->s.n) { sb_put(&b, s, n); }
			else for (size_t i = 0; i < n;) {
				if (cnt && i + from->s.n <= n && !memcmp(s + i, from->s.s, from->s.n)) { sb_put(&b, to->s.s, to->s.n); i += from->s.n; if (cnt > 0) cnt--; }
				else sb_putc(&b, s[i++]);
			}
			jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r;
		}
		if (IS("split") || IS("rsplit")) {
			jv *l = jv_new(c->a, J_LIST); jv *sep = g->n && g->a[0]->k == J_STR ? g->a[0] : NULL; long long mx = g->n > 1 ? as_i(g->a[1]) : -1;
			if (!sep) {
				size_t i = 0;
				while (i < n) { while (i < n && isws(s[i])) i++; if (i >= n) break; size_t j = i; while (j < n && !isws(s[j])) j++; jv_push(c->a, l, jv_str(c->a, s + i, j - i)); i = j; }
			} else if (sep->s.n) {
				size_t i = 0;
				for (;;) {
					const char *m = mx != 0 && n - i >= sep->s.n ? memmem(s + i, n - i, sep->s.s, sep->s.n) : NULL;
					if (!m) { jv_push(c->a, l, jv_str(c->a, s + i, n - i)); break; }
					jv_push(c->a, l, jv_str(c->a, s + i, (size_t)(m - s) - i)); i = (size_t)(m - s) + sep->s.n; if (mx > 0) mx--;
				}
			}
			return l;
		}
		if (IS("splitlines")) {
			jv *l = jv_new(c->a, J_LIST); size_t i = 0;
			while (i < n) { size_t j = i; while (j < n && s[j] != '\n') j++; size_t e = j; if (e > i && s[e - 1] == '\r') e--; jv_push(c->a, l, jv_str(c->a, s + i, e - i)); i = j + 1; }
			return l;
		}
		if (IS("find") || IS("count")) {
			if (!g->n || g->a[0]->k != J_STR) rfail(c, "%.*s() takes a string", (int)nl, nm);
			jv *q = g->a[0];
			if (nm[0] == 'f') { const char *m = q->s.n <= n ? memmem(s, n, q->s.s, q->s.n) : NULL; return jv_int(c->a, m ? (long long)(m - s) : -1); }
			long long k = 0; if (q->s.n) for (size_t i = 0; i + q->s.n <= n;) { if (!memcmp(s + i, q->s.s, q->s.n)) { k++; i += q->s.n; } else i++; }
			return jv_int(c->a, k);
		}
		if (IS("join")) {
			if (!g->n) rfail(c, "join() needs a list");
			jv *l = as_list(c, g->a[0]); sbuf b = {0};
			for (int i = 0; i < l->l.n; i++) { if (i) sb_put(&b, s, n); put_str(&b, l->l.a[i]); }
			jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r;
		}
		return NULL;
	}
	if (is_map(o)) {
		if (IS("items") || IS("keys") || IS("values")) {
			jv *l = jv_new(c->a, J_LIST);
			for (int i = 0; i < o->o.n; i++) {
				jv *k = jv_str(c->a, o->o.key[i], o->o.klen[i]);
				if (nm[0] == 'k') jv_push(c->a, l, k);
				else if (nm[0] == 'v') jv_push(c->a, l, o->o.val[i]);
				else { jv *t = jv_new(c->a, J_TUPLE); jv_push(c->a, t, k); jv_push(c->a, t, o->o.val[i]); jv_push(c->a, l, t); }
			}
			return l;
		}
		if (IS("get")) { jv *k = g->n ? g->a[0] : NULL; jv *r = k && k->k == J_STR ? jv_get(o, k->s.s, k->s.n) : NULL; return r ? r : g->n > 1 ? g->a[1] : jv_new(c->a, J_NULL); }
		if (IS("update") && g->n && is_map(g->a[0])) { for (int i = 0; i < g->a[0]->o.n; i++) jv_set(c->a, o, g->a[0]->o.key[i], g->a[0]->o.klen[i], g->a[0]->o.val[i]); return jv_new(c->a, J_NULL); }
		return NULL;
	}
	if (o->k == J_LIST) {
		if (IS("append") && g->n) { jv_push(c->a, o, g->a[0]); return jv_new(c->a, J_NULL); }
		if (IS("pop")) { if (!o->l.n) rfail(c, "pop from empty list"); return o->l.a[--o->l.n]; }
		return NULL;
	}
	return NULL;
#undef IS
}

static jv *do_tojson(rc *c, jv *v, const argv_t *g) {
	(void)g; sbuf b = {0}; json_dump(&b, v, 1);
	jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r;
}

static jv *do_filter(rc *c, const expr *f, jv *v, const argv_t *g) {
#define IS(s) (f->nlen == sizeof(s) - 1 && !memcmp(f->name, s, f->nlen))
	if (IS("trim")) { jv *s = tostr(c, v); size_t b, e; strip_range(s->s.s, s->s.n, g->n && g->a[0]->k == J_STR ? g->a[0] : NULL, 1, 1, &b, &e); return jv_str(c->a, s->s.s + b, e - b); }
	if (IS("default") || IS("d")) {
		jv *d = arg_at(g, 0, "default_value"); jv *bo = arg_at(g, 1, "boolean");
		if (!d) d = jv_str(c->a, "", 0);
		return (bo && truthy(bo)) ? (truthy(v) ? v : d) : (v->k == J_UNDEF ? d : v);
	}
	if (IS("length") || IS("count")) {
		if (v->k == J_STR) return jv_int(c->a, (long long)u8len(v->s.s, v->s.n));
		if (is_seq(v)) return jv_int(c->a, v->l.n);
		if (is_map(v)) return jv_int(c->a, v->o.n);
		rfail(c, "object has no length");
	}
	if (IS("string")) return tostr(c, v);
	if (IS("tojson")) return do_tojson(c, v, g);
	if (IS("safe")) return v;
	if (IS("items") || IS("keys") || IS("values")) { if (v->k == J_UNDEF) return jv_new(c->a, J_LIST); jv *r = call_method(c, v, f->name, f->nlen, g); if (!r) rfail(c, "%.*s needs a mapping", (int)f->nlen, f->name); return r; }
	if (IS("list")) { jv *l = as_list(c, v); if (l == v) { jv *r = jv_new(c->a, J_LIST); for (int i = 0; i < l->l.n; i++) jv_push(c->a, r, l->l.a[i]); return r; } return l; }
	if (IS("join")) {
		jv *sep = g->n ? g->a[0] : kwarg(g, "d"); jv *l = as_list(c, v); sbuf b = {0};
		for (int i = 0; i < l->l.n; i++) { if (i && sep && sep->k == J_STR) sb_put(&b, sep->s.s, sep->s.n); put_str(&b, l->l.a[i]); }
		jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r;
	}
	if (IS("first") || IS("last")) {
		jv *l = as_list(c, v); if (!l->l.n) return undef(c);
		return l->l.a[f->name[0] == 'f' ? 0 : l->l.n - 1];
	}
	if (IS("reverse")) {
		if (v->k == J_STR) { jv *l = jv_new(c->a, J_LIST); chars_of(c, v, l); sbuf b = {0}; for (int i = l->l.n - 1; i >= 0; i--) sb_put(&b, l->l.a[i]->s.s, l->l.a[i]->s.n); jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r; }
		jv *l = as_list(c, v), *r = jv_new(c->a, J_LIST); for (int i = l->l.n - 1; i >= 0; i--) jv_push(c->a, r, l->l.a[i]); return r;
	}
	if (IS("upper") || IS("lower") || IS("capitalize") || IS("title")) { jv *s = tostr(c, v); jv *r = call_method(c, s, f->name, f->nlen, g); return r; }
	if (IS("replace")) { jv *s = tostr(c, v); return call_method(c, s, f->name, f->nlen, g); }
	if (IS("int")) {
		if (v->k == J_STR) { char *e; long long i = strtoll(v->s.s, &e, 10); return jv_int(c->a, e == v->s.s ? (g->n ? as_i(g->a[0]) : 0) : i); }
		return is_num(v) ? jv_int(c->a, as_i(v)) : jv_int(c->a, g->n ? as_i(g->a[0]) : 0);
	}
	if (IS("float")) { jv *r = jv_new(c->a, J_NUM); r->d = v->k == J_STR ? strtod(v->s.s, NULL) : is_num(v) ? as_d(v) : 0; return r; }
	if (IS("abs")) { if (v->k == J_NUM) { jv *r = jv_new(c->a, J_NUM); r->d = fabs(v->d); return r; } return jv_int(c->a, llabs(as_i(v))); }
	if (IS("indent")) {   /* indent(width = 4, first = false) */
		jv *s = tostr(c, v); long long w = g->n ? as_i(g->a[0]) : 4; int first = g->n > 1 && truthy(g->a[1]); sbuf b = {0};
		for (size_t i = 0; i < s->s.n; i++) {
			if ((i == 0 && first) || (i > 0 && s->s.s[i - 1] == '\n' && s->s.s[i] != '\n')) for (long long k = 0; k < w; k++) sb_putc(&b, ' ');
			sb_putc(&b, s->s.s[i]);
		}
		jv *r = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return r;
	}
	rfail(c, "unknown filter '%.*s'", (int)f->nlen, f->name);
#undef IS
}

static int do_test(rc *c, const expr *t, jv *v, const argv_t *g) {
#define IS(s) (t->nlen == sizeof(s) - 1 && !memcmp(t->name, s, t->nlen))
	if (IS("defined")) return v->k != J_UNDEF;
	if (IS("undefined")) return v->k == J_UNDEF;
	if (IS("none")) return v->k == J_NULL;
	if (IS("string")) return v->k == J_STR;
	if (IS("number")) return v->k == J_INT || v->k == J_NUM;
	if (IS("integer")) return v->k == J_INT;
	if (IS("float")) return v->k == J_NUM;
	if (IS("mapping")) return is_map(v);
	if (IS("sequence")) return is_seq(v) || v->k == J_STR;
	if (IS("iterable")) return is_seq(v) || is_map(v) || v->k == J_STR;
	if (IS("boolean")) return v->k == J_BOOL;
	if (IS("true")) return v->k == J_BOOL && v->b;
	if (IS("false")) return v->k == J_BOOL && !v->b;
	if (IS("callable")) return v->k == J_MACRO || v->k == J_FUNC;
	if (IS("odd")) return is_num(v) && (as_i(v) & 1);
	if (IS("even")) return is_num(v) && !(as_i(v) & 1);
	if (IS("divisibleby")) return g->n && is_num(v) && as_i(g->a[0]) && as_i(v) % as_i(g->a[0]) == 0;
	if (IS("eq") || IS("equalto") || IS("sameas")) return g->n && jeq(v, g->a[0]);
	if (IS("ne")) return g->n && !jeq(v, g->a[0]);
	if (IS("lower") || IS("upper")) { if (v->k != J_STR) return 0; for (size_t i = 0; i < v->s.n; i++) if (isalpha((unsigned char)v->s.s[i]) && (IS("lower") ? isupper((unsigned char)v->s.s[i]) : islower((unsigned char)v->s.s[i]))) return 0; return 1; }
	if (IS("in")) return g->n && jin(c, v, g->a[0]);
	rfail(c, "unknown test '%.*s'", (int)t->nlen, t->name);
#undef IS
}

static jv *slice_of(rc *c, jv *o, jv *lo, jv *hi, jv *st) {
	long long step = st && st->k != J_NULL && st->k != J_UNDEF ? as_i(st) : 1; if (!step) rfail(c, "slice step cannot be zero");
	int isstr = o->k == J_STR; jv *chars = o;
	if (isstr) { chars = jv_new(c->a, J_LIST); chars_of(c, o, chars); }
	else if (!is_seq(o)) rfail(c, "value cannot be sliced");
	long long n = chars->l.n, a, b;
	if (step > 0) { a = lo && lo->k != J_NULL && lo->k != J_UNDEF ? as_i(lo) : 0; b = hi && hi->k != J_NULL && hi->k != J_UNDEF ? as_i(hi) : n; if (a < 0) a += n; if (b < 0) b += n; if (a < 0) a = 0; if (b > n) b = n; }
	else { a = lo && lo->k != J_NULL && lo->k != J_UNDEF ? as_i(lo) : n - 1; b = hi && hi->k != J_NULL && hi->k != J_UNDEF ? as_i(hi) : -n - 1; if (a < 0) a += n; if (b < -n - 1) b = -n - 1; if (b < 0 && !(hi && hi->k != J_NULL && hi->k != J_UNDEF)) b = -1; else if (b < 0) b += n; if (a >= n) a = n - 1; }
	jv *r = jv_new(c->a, isstr ? J_LIST : o->k);
	if (step > 0) for (long long i = a; i < b; i += step) jv_push(c->a, r, chars->l.a[i]);
	else for (long long i = a; i > b; i += step) if (i >= 0 && i < n) jv_push(c->a, r, chars->l.a[i]);
	if (!isstr) return r;
	sbuf s = {0}; for (int i = 0; i < r->l.n; i++) sb_put(&s, r->l.a[i]->s.s, r->l.a[i]->s.n);
	jv *out = jv_str(c->a, s.s ? s.s : "", s.n); sb_free(&s); return out;
}

static void eval_args(rc *c, scope *sc, const expr *x, argv_t *g, jv ***stack) {
	g->n = x->nargs; g->nk = x->nkw; g->kn = x->kn; g->kl = x->kl;
	jv **buf = arena_alloc(c->a, sizeof(jv *) * (size_t)(x->nargs + x->nkw + 1)); *stack = buf;
	g->a = buf; g->kv = buf + x->nargs;
	for (int i = 0; i < x->nargs; i++) buf[i] = ev(c, sc, x->args[i]);
	for (int i = 0; i < x->nkw; i++) g->kv[i] = ev(c, sc, x->kv[i]);
}

static jv *ev(rc *c, scope *sc, const expr *x) {
	switch (x->k) {
	case E_LIT: return x->lit;
	case E_NAME: {
		jv *v = scope_find(sc, x->name, x->nlen);
		if (v) return v;
		int id = builtin_id(x->name, x->nlen);
		if (id) { v = jv_new(c->a, J_FUNC); v->f.id = id; return v; }
		return undef(c);
	}
	case E_LIST: case E_TUPLE: { jv *l = jv_new(c->a, x->k == E_LIST ? J_LIST : J_TUPLE); for (int i = 0; i < x->nitems; i++) jv_push(c->a, l, ev(c, sc, x->items[i])); return l; }
	case E_DICT: {
		jv *d = jv_new(c->a, J_DICT);
		for (int i = 0; i < x->nitems; i += 2) { jv *k = tostr(c, ev(c, sc, x->items[i])); jv_set(c->a, d, k->s.s, k->s.n, ev(c, sc, x->items[i + 1])); }
		return d;
	}
	case E_ATTR: {
		jv *o = ev(c, sc, x->a);
		if (is_map(o)) { jv *r = jv_get(o, x->name, x->nlen); return r ? r : undef(c); }
		return undef(c);
	}
	case E_SUB: {
		jv *o = ev(c, sc, x->a), *i = ev(c, sc, x->b);
		if (is_seq(o) && is_num(i)) { long long k = as_i(i); if (k < 0) k += o->l.n; return k >= 0 && k < o->l.n ? o->l.a[k] : undef(c); }
		if (is_map(o) && i->k == J_STR) { jv *r = jv_get(o, i->s.s, i->s.n); return r ? r : undef(c); }
		if (o->k == J_STR && is_num(i)) { jv *l = jv_new(c->a, J_LIST); chars_of(c, o, l); long long k = as_i(i); if (k < 0) k += l->l.n; return k >= 0 && k < l->l.n ? l->l.a[k] : undef(c); }
		return undef(c);
	}
	case E_SLICE: {
		jv *o = ev(c, sc, x->a);
		return slice_of(c, o, x->b ? ev(c, sc, x->b) : NULL, x->c ? ev(c, sc, x->c) : NULL, x->x ? ev(c, sc, x->x) : NULL);
	}
	case E_CALL: {
		argv_t g; jv **st;
		const expr *f = x->a;
		if (f->k == E_ATTR) {
			jv *o = ev(c, sc, f->a);
			eval_args(c, sc, x, &g, &st);
			jv *r = call_method(c, o, f->name, f->nlen, &g);
			if (r) return r;
			jv *fn = is_map(o) ? jv_get(o, f->name, f->nlen) : NULL;
			if (fn && fn->k == J_MACRO) return call_macro(c, fn, &g);
			if (fn && fn->k == J_FUNC) return call_builtin(c, fn->f.id, &g);
			rfail(c, "no method '%.*s'", (int)f->nlen, f->name);
		}
		jv *fn = ev(c, sc, f);
		eval_args(c, sc, x, &g, &st);
		if (fn->k == J_MACRO) return call_macro(c, fn, &g);
		if (fn->k == J_FUNC) return call_builtin(c, fn->f.id, &g);
		if (f->k == E_NAME) rfail(c, "'%.*s' is not callable", (int)f->nlen, f->name);
		rfail(c, "value is not callable");
	}
	case E_FILTER: { jv *v = ev(c, sc, x->a); argv_t g; jv **st; eval_args(c, sc, x, &g, &st); return do_filter(c, x, v, &g); }
	case E_TEST: { jv *v = ev(c, sc, x->a); argv_t g; jv **st; eval_args(c, sc, x, &g, &st); return mkbool(c, do_test(c, x, v, &g) != x->neg); }
	case E_NEG: { jv *v = ev(c, sc, x->a); if (v->k == J_NUM) { jv *r = jv_new(c->a, J_NUM); r->d = -v->d; return r; } if (!is_num(v)) rfail(c, "cannot negate"); return jv_int(c->a, -as_i(v)); }
	case E_POS: return ev(c, sc, x->a);
	case E_NOT: return mkbool(c, !truthy(ev(c, sc, x->a)));
	case E_AND: { jv *l = ev(c, sc, x->a); return truthy(l) ? ev(c, sc, x->b) : l; }
	case E_OR: { jv *l = ev(c, sc, x->a); return truthy(l) ? l : ev(c, sc, x->b); }
	case E_COND: return truthy(ev(c, sc, x->a)) ? ev(c, sc, x->b) : x->c ? ev(c, sc, x->c) : undef(c);
	case E_BIN: {
		jv *l = ev(c, sc, x->a), *r = ev(c, sc, x->b);
		switch (x->op) {
		case OP_EQ: return mkbool(c, jeq(l, r));
		case OP_NE: return mkbool(c, !jeq(l, r));
		case OP_LT: return mkbool(c, jcmp(c, l, r) < 0);
		case OP_GT: return mkbool(c, jcmp(c, l, r) > 0);
		case OP_LE: return mkbool(c, jcmp(c, l, r) <= 0);
		case OP_GE: return mkbool(c, jcmp(c, l, r) >= 0);
		case OP_IN: return mkbool(c, jin(c, l, r));
		case OP_NOTIN: return mkbool(c, !jin(c, l, r));
		case OP_CAT: { sbuf b = {0}; put_str(&b, l); put_str(&b, r); jv *s = jv_str(c->a, b.s ? b.s : "", b.n); sb_free(&b); return s; }
		default: return arith(c, x->op, l, r);
		}
	}
	}
	rfail(c, "bad expression");
}

static void assign(rc *c, scope *sc, const expr *t, jv *v) {
	if (t->k == E_NAME) { scope_set(c, sc, t->name, t->nlen, v); return; }
	if (t->k == E_ATTR) {
		jv *o = ev(c, sc, t->a);
		if (o->k != J_NS) rfail(c, "cannot assign attribute '%.*s' of a non-namespace", (int)t->nlen, t->name);
		jv_set(c->a, o, t->name, t->nlen, v); return;
	}
	if (t->k == E_TUPLE) {
		jv *l = as_list(c, v);
		if (l->l.n != t->nitems) rfail(c, "cannot unpack %d values into %d names", l->l.n, t->nitems);
		for (int i = 0; i < t->nitems; i++) assign(c, sc, t->items[i], l->l.a[i]);
		return;
	}
	rfail(c, "bad assignment target");
}

static void exec_list(rc *c, scope *sc, const stmt *s, sbuf *out) {
	for (; s; s = s->next) {
		switch (s->k) {
		case S_TEXT: sb_put(out, s->text, s->tlen); break;
		case S_OUT: put_str(out, ev(c, sc, s->e)); break;
		case S_IF: if (truthy(ev(c, sc, s->e))) exec_list(c, sc, s->body, out); else if (s->els) exec_list(c, sc, s->els, out); break;
		case S_SET:
			if (s->e) assign(c, sc, s->target, ev(c, sc, s->e));
			else { sbuf b = {0}; exec_list(c, sc, s->body, &b); assign(c, sc, s->target, jv_str(c->a, b.s ? b.s : "", b.n)); sb_free(&b); }
			break;
		case S_MACRO: { jv *m = jv_new(c->a, J_MACRO); m->f.p = s; m->f.sc = sc; scope_set(c, sc, s->mname, s->mlen, m); break; }
		case S_BREAK: c->brk = 1; return;
		case S_CONTINUE: c->cont = 1; return;
		case S_FOR: {
			jv *src = as_list(c, ev(c, sc, s->e)), *items = src;
			if (s->cond) {
				items = jv_new(c->a, J_LIST);
				for (int i = 0; i < src->l.n; i++) {
					scope *t = scope_new(c, sc);
					if (s->nnames == 1) scope_set(c, t, s->names[0], s->nlens[0], src->l.a[i]);
					else { jv *u = as_list(c, src->l.a[i]); for (int j = 0; j < s->nnames && j < u->l.n; j++) scope_set(c, t, s->names[j], s->nlens[j], u->l.a[j]); }
					if (truthy(ev(c, t, s->cond))) jv_push(c->a, items, src->l.a[i]);
				}
			}
			int n = items->l.n;
			for (int i = 0; i < n; i++) {
				scope *t = scope_new(c, sc);
				if (s->nnames == 1) scope_set(c, t, s->names[0], s->nlens[0], items->l.a[i]);
				else {
					jv *u = as_list(c, items->l.a[i]);
					if (u->l.n != s->nnames) rfail(c, "cannot unpack %d values into %d names", u->l.n, s->nnames);
					for (int j = 0; j < s->nnames; j++) scope_set(c, t, s->names[j], s->nlens[j], u->l.a[j]);
				}
				jv *loop = jv_new(c->a, J_DICT);
				jv_set(c->a, loop, "index", 5, jv_int(c->a, i + 1)); jv_set(c->a, loop, "index0", 6, jv_int(c->a, i));
				jv_set(c->a, loop, "revindex", 8, jv_int(c->a, n - i)); jv_set(c->a, loop, "revindex0", 9, jv_int(c->a, n - i - 1));
				jv_set(c->a, loop, "first", 5, mkbool(c, i == 0)); jv_set(c->a, loop, "last", 4, mkbool(c, i == n - 1));
				jv_set(c->a, loop, "length", 6, jv_int(c->a, n));
				jv_set(c->a, loop, "previtem", 8, i > 0 ? items->l.a[i - 1] : undef(c)); jv_set(c->a, loop, "nextitem", 8, i < n - 1 ? items->l.a[i + 1] : undef(c));
				scope_set(c, t, "loop", 4, loop);
				exec_list(c, t, s->body, out);
				if (c->brk) { c->brk = 0; break; }
				c->cont = 0;
			}
			if (!n && s->els) exec_list(c, sc, s->els, out);
			break;
		}
		}
	}
}

char *jinja_render(const jinja *t, arena *a, const jv *vars, size_t *outn, char *err, size_t errcap) {
	rc *c = calloc(1, sizeof *c); c->a = a;
	sbuf out = {0};
	if (setjmp(c->jb)) { if (err) snprintf(err, errcap, "%s", c->err); sb_free(&out); free(c); return NULL; }
	scope *top = scope_new(c, NULL);
	if (vars) for (int i = 0; i < vars->o.n; i++) scope_set(c, top, vars->o.key[i], vars->o.klen[i], vars->o.val[i]);
	exec_list(c, top, t->body, &out);
	free(c);
	if (!out.s) { out.s = malloc(1); out.s[0] = 0; }
	if (outn) *outn = out.n;
	return out.s;
}
