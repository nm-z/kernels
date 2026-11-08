/* Minimal read-only GGUF access: mmap the file, index the tensors, skip the metadata. */
#ifndef FLASH_GGUF_H
#define FLASH_GGUF_H
#include <stddef.h>
#include <stdint.h>

enum {
	GG_F32 = 0, GG_F16 = 1, GG_Q4_0 = 2, GG_Q4_1 = 3, GG_Q5_0 = 6, GG_Q5_1 = 7, GG_Q8_0 = 8, GG_Q8_1 = 9,
	GG_Q2_K = 10, GG_Q3_K = 11, GG_Q4_K = 12, GG_Q5_K = 13, GG_Q6_K = 14, GG_Q8_K = 15,
	GG_IQ4_NL = 20, GG_IQ4_XS = 23, GG_BF16 = 30, GG_Q2_0 = 42
};

typedef struct {
	const char *name;
	uint32_t type;
	uint32_t nd;
	uint64_t dims[4];      /* dims[0] is the contiguous (K) dimension */
	uint64_t elems;
	uint64_t off;          /* relative to data_start */
	uint64_t nbytes;       /* from the gap to the next tensor, or to the end of the file */
	const uint8_t *data;   /* NULL when the mapping does not cover it (truncated file) */
} gguf_tensor;

/* one metadata entry: val points at the value as stored (for an array: u32 element type, u64 count,
   then the elements; a string is u64 length + bytes) */
typedef struct {
	const char *key;
	uint64_t klen;
	uint32_t type;         /* 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array 10 u64 11 i64 12 f64 */
	const uint8_t *val;
} gguf_kv;

typedef struct {
	int fd;
	uint8_t *map;
	size_t len;
	uint64_t ntensors;
	gguf_tensor *t;
	uint64_t data_start;
	uint64_t align;
	uint64_t nkv;
	gguf_kv *kv;
} gguf_file;

int gguf_open(gguf_file *g, const char *path);
void gguf_close(gguf_file *g);
const gguf_tensor *gguf_find(const gguf_file *g, const char *name);
/* elements per block and bytes per block for a type; 0,0 if unknown */
void gguf_block(uint32_t type, uint32_t *elems, uint32_t *bytes);
const char *gguf_type_name(uint32_t type);

/* metadata */
const gguf_kv *gguf_get(const gguf_file *g, const char *key);
int gguf_get_int(const gguf_file *g, const char *key, int64_t *out);                  /* any integer or bool; 0 if found */
const char *gguf_get_str(const gguf_file *g, const char *key, uint64_t *len);         /* not NUL-terminated */
int gguf_get_arr(const gguf_file *g, const char *key, uint32_t *etype, uint64_t *n, const uint8_t **data);   /* 0 if found */

#endif
