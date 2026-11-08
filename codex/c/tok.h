/* tok.h: the model's tokenizer, read from the GGUF (tokenizer.ggml.*): byte-level BPE with the
 * qwen35 pre-tokenizer, the same splits, merges and special-token handling as llama.cpp's
 * (src/llama-vocab.cpp, src/unicode.cpp), so prompts tokenize to the ids the reference saw. */
#ifndef FLASH_TOK_H
#define FLASH_TOK_H
#include <stddef.h>
#include <stdint.h>

typedef struct tok_vocab tok_vocab;

tok_vocab *tok_load(const char *gguf_path);              /* NULL (and a message) on failure */
int tok_count(const tok_vocab *v);
/* text -> ids; parse_special: control tokens written as text (<|im_start|>) become their ids
 * (user-defined ones always do). Returns the count, or -1 if out (of cap) is too small. */
int tok_encode(const tok_vocab *v, const char *text, size_t len, int parse_special, uint32_t *out, int cap);
/* id -> bytes (special tokens as their text); returns the length, or -1 for a bad id */
int tok_piece(const tok_vocab *v, uint32_t id, char *buf, int cap);
int tok_find(const tok_vocab *v, const char *text);      /* exact token text -> id, or -1 */
uint32_t tok_eos(const tok_vocab *v);                    /* tokenizer.ggml.eos_token_id */
int tok_add_bos(const tok_vocab *v); uint32_t tok_bos(const tok_vocab *v);
const char *tok_template(const tok_vocab *v);            /* tokenizer.chat_template (NUL-terminated) or NULL */
const char *tok_text(const tok_vocab *v, uint32_t id, size_t *len);   /* the vocabulary entry as stored */
int tok_is_special(const tok_vocab *v, uint32_t id);   /* control, user-defined or unknown: what the tokenizer recognises as special (not what the output hides, see tok_is_hidden) */
int tok_is_hidden(const tok_vocab *v, uint32_t id);    /* control or unknown: not shown in a response; user-defined tokens (<tool_call>) are */
int tok_meta_num(const tok_vocab *v, const char *key, double *out);   /* a numeric GGUF metadata value (int, bool or float); 0 if present */

#endif
