/* json.h: values shared by the JSON reader/writer and the Jinja interpreter (jinja.c), an arena
 * that owns them, and a growable string buffer. */
#ifndef FLASH_JSON_H
#define FLASH_JSON_H
#include <stddef.h>
#include <stdint.h>

typedef struct arena arena;
arena *arena_new(void);
void arena_free(arena *a);
void *arena_alloc(arena *a, size_t n);          /* zeroed */
char *arena_strndup(arena *a, const char *s, size_t n);

typedef struct { char *s; size_t n, cap; } sbuf;
void sb_put(sbuf *b, const char *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_putc(sbuf *b, char c);
void sb_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void sb_json_str(sbuf *b, const char *s, size_t n);   /* a JSON string literal, quotes included */
void sb_free(sbuf *b);

/* J_NS: a Jinja namespace (a dict whose attributes can be set); J_MACRO, J_FUNC: Jinja callables */
enum { J_UNDEF, J_NULL, J_BOOL, J_INT, J_NUM, J_STR, J_LIST, J_TUPLE, J_DICT, J_NS, J_MACRO, J_FUNC };
typedef struct jv jv;
struct jv {
	int k;
	union {
		int b;
		long long i;
		double d;
		struct { const char *s; size_t n; } s;
		struct { jv **a; int n, cap; } l;                       /* J_LIST, J_TUPLE */
		struct { const char **key; size_t *klen; jv **val; int n, cap; } o;   /* J_DICT, J_NS (insertion order) */
		struct { const void *p; const void *sc; int id; } f;   /* J_MACRO: node and defining scope; J_FUNC: builtin id */
	};
};

jv *jv_new(arena *a, int k);
jv *jv_str(arena *a, const char *s, size_t n);    /* copies */
jv *jv_strref(arena *a, const char *s, size_t n); /* refers */
jv *jv_int(arena *a, long long i);
jv *jv_bool(arena *a, int b);
void jv_push(arena *a, jv *list, jv *v);
void jv_set(arena *a, jv *dict, const char *key, size_t klen, jv *v);   /* replaces an existing key */
jv *jv_get(const jv *dict, const char *key, size_t klen);              /* NULL if absent */
jv *jv_getz(const jv *dict, const char *key);

/* JSON text -> value (NULL and *err on a syntax error) */
jv *json_parse(arena *a, const char *s, size_t n, const char **err);
/* value -> JSON; sep: 0 compact (",", ":"), 1 Python json.dumps style (", ", ": ") */
void json_dump(sbuf *b, const jv *v, int sep);

#endif
