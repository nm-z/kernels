/* jinja.h: a Jinja2 interpreter for model chat templates (tokenizer.chat_template in the GGUF).
 *
 * Syntax: {{ }}, {% %}, {# #} with - whitespace control, trim_blocks and lstrip_blocks on (as
 * llama.cpp and transformers set them for chat templates); if/elif/else, for (loop.*, else, if
 * filter, break/continue), set (names, namespace attributes, tuples, blocks), macro with default
 * arguments; expressions with ternaries, tests (is [not] defined/string/mapping/...), filters
 * (trim, default, length, tojson, items, join, ...), slices, str/dict/list methods, and the
 * globals namespace(), raise_exception(), range(), dict(), strftime_now(). */
#ifndef FLASH_JINJA_H
#define FLASH_JINJA_H
#include "json.h"

typedef struct jinja jinja;

/* NULL and a message in err on a syntax error */
jinja *jinja_compile(const char *src, size_t n, char *err, size_t errcap);
/* Render with the top-level variables in vars (a J_DICT). Values made while rendering live in a.
 * Returns a malloc'ed string (*outn bytes), or NULL and a message in err (a raise_exception()
 * call, or a runtime error). */
char *jinja_render(const jinja *t, arena *a, const jv *vars, size_t *outn, char *err, size_t errcap);
void jinja_free(jinja *t);

#endif
