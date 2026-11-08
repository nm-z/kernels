#define _DEFAULT_SOURCE
#include "gguf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct cur { const uint8_t *p, *end; int bad; };

static uint64_t rd(struct cur *c, int n) {
	uint64_t v = 0;
	if (c->p + n > c->end) { c->bad = 1; return 0; }
	memcpy(&v, c->p, n);
	c->p += n;
	return v;
}
static const char *rds(struct cur *c, uint64_t *len) {
	uint64_t n = rd(c, 8);
	if (c->bad || c->p + n > c->end) { c->bad = 1; return NULL; }
	const char *s = (const char *)c->p;
	c->p += n;
	if (len) *len = n;
	return s;
}
static int scalar_size(uint32_t t) {
	switch (t) {
	case 0: case 1: case 7: return 1;
	case 2: case 3: return 2;
	case 4: case 5: case 6: return 4;
	case 10: case 11: case 12: return 8;
	default: return 0;
	}
}
static void skip_value(struct cur *c, uint32_t t) {
	if (t == 8) { rds(c, NULL); return; }
	if (t == 9) {
		uint32_t et = (uint32_t)rd(c, 4);
		uint64_t n = rd(c, 8);
		if (et == 8 || et == 9) { for (uint64_t i = 0; i < n && !c->bad; i++) skip_value(c, et); }
		else { int s = scalar_size(et); if (!s) { c->bad = 1; return; } if (c->p + n * s > c->end) { c->bad = 1; return; } c->p += n * s; }
		return;
	}
	int s = scalar_size(t);
	if (!s) { c->bad = 1; return; }
	rd(c, s);
}

int gguf_open(gguf_file *g, const char *path) {
	memset(g, 0, sizeof *g);
	g->fd = open(path, O_RDONLY);
	if (g->fd < 0) { perror(path); return -1; }
	struct stat st;
	if (fstat(g->fd, &st)) { perror("fstat"); return -1; }
	g->len = (size_t)st.st_size;
	g->map = mmap(NULL, g->len, PROT_READ, MAP_PRIVATE, g->fd, 0);
	if (g->map == MAP_FAILED) { perror("mmap"); return -1; }
	struct cur c = { g->map, g->map + g->len, 0 };
	if (memcmp(c.p, "GGUF", 4)) { fprintf(stderr, "%s: not GGUF\n", path); return -1; }
	c.p += 4;
	uint32_t ver = (uint32_t)rd(&c, 4);
	if (ver != 3 && ver != 2) { fprintf(stderr, "%s: gguf v%u unsupported\n", path, ver); return -1; }
	g->ntensors = rd(&c, 8);
	uint64_t nkv = rd(&c, 8);
	g->align = 32;
	g->kv = calloc(nkv ? nkv : 1, sizeof *g->kv);
	for (uint64_t i = 0; i < nkv && !c.bad; i++) {
		uint64_t kl;
		const char *k = rds(&c, &kl);
		if (c.bad) break;
		uint32_t t = (uint32_t)rd(&c, 4);
		g->kv[i] = (gguf_kv){ k, kl, t, c.p }; g->nkv = i + 1;
		if (kl == 17 && !memcmp(k, "general.alignment", 17) && t == 4) { g->align = rd(&c, 4); continue; }
		skip_value(&c, t);
	}
	if (c.bad) { fprintf(stderr, "%s: bad metadata\n", path); return -1; }
	g->t = calloc(g->ntensors, sizeof *g->t);
	for (uint64_t i = 0; i < g->ntensors && !c.bad; i++) {
		gguf_tensor *t = &g->t[i];
		uint64_t nl;
		const char *n = rds(&c, &nl);
		if (c.bad) break;
		char *name = malloc(nl + 1);
		memcpy(name, n, nl);
		name[nl] = 0;
		t->name = name;
		t->nd = (uint32_t)rd(&c, 4);
		if (t->nd > 4) { c.bad = 1; break; }
		t->elems = 1;
		for (uint32_t d = 0; d < t->nd; d++) { t->dims[d] = rd(&c, 8); t->elems *= t->dims[d]; }
		t->type = (uint32_t)rd(&c, 4);
		t->off = rd(&c, 8);
	}
	if (c.bad) { fprintf(stderr, "%s: bad tensor table\n", path); return -1; }
	uint64_t hdr = (uint64_t)(c.p - g->map);
	g->data_start = (hdr + g->align - 1) / g->align * g->align;
	for (uint64_t i = 0; i < g->ntensors; i++) {
		uint64_t next = UINT64_MAX;
		for (uint64_t j = 0; j < g->ntensors; j++)
			if (g->t[j].off > g->t[i].off && g->t[j].off < next) next = g->t[j].off;
		uint64_t end = next == UINT64_MAX ? g->len - g->data_start : next;
		g->t[i].nbytes = end - g->t[i].off;
		uint64_t abs = g->data_start + g->t[i].off;
		g->t[i].data = abs + g->t[i].nbytes <= g->len ? g->map + abs : NULL;
	}
	return 0;
}

void gguf_close(gguf_file *g) {
	if (g->t) { for (uint64_t i = 0; i < g->ntensors; i++) free((char *)g->t[i].name); free(g->t); }
	free(g->kv);
	if (g->map && g->map != MAP_FAILED) munmap(g->map, g->len);
	if (g->fd >= 0) close(g->fd);
	memset(g, 0, sizeof *g);
}

const gguf_tensor *gguf_find(const gguf_file *g, const char *name) {
	for (uint64_t i = 0; i < g->ntensors; i++) if (!strcmp(g->t[i].name, name)) return &g->t[i];
	return NULL;
}

void gguf_block(uint32_t type, uint32_t *elems, uint32_t *bytes) {
	uint32_t e = 0, b = 0;
	switch (type) {
	case GG_F32: e = 1; b = 4; break;
	case GG_F16: case GG_BF16: e = 1; b = 2; break;
	case GG_Q4_0: e = 32; b = 18; break;
	case GG_Q4_1: e = 32; b = 20; break;
	case GG_Q5_0: e = 32; b = 22; break;
	case GG_Q5_1: e = 32; b = 24; break;
	case GG_Q8_0: e = 32; b = 34; break;
	case GG_Q8_1: e = 32; b = 36; break;
	case GG_Q2_K: e = 256; b = 84; break;
	case GG_Q3_K: e = 256; b = 110; break;
	case GG_Q4_K: e = 256; b = 144; break;
	case GG_Q5_K: e = 256; b = 176; break;
	case GG_Q6_K: e = 256; b = 210; break;
	case GG_Q8_K: e = 256; b = 292; break;
	case GG_IQ4_NL: e = 32; b = 18; break;
	case GG_IQ4_XS: e = 256; b = 136; break;
	case GG_Q2_0: e = 64; b = 18; break;
	}
	*elems = e;
	*bytes = b;
}

const char *gguf_type_name(uint32_t type) {
	switch (type) {
	case GG_F32: return "F32"; case GG_F16: return "F16"; case GG_Q4_0: return "Q4_0"; case GG_Q4_1: return "Q4_1";
	case GG_Q5_0: return "Q5_0"; case GG_Q5_1: return "Q5_1"; case GG_Q8_0: return "Q8_0"; case GG_Q8_1: return "Q8_1";
	case GG_Q2_K: return "Q2_K"; case GG_Q3_K: return "Q3_K"; case GG_Q4_K: return "Q4_K"; case GG_Q5_K: return "Q5_K";
	case GG_Q6_K: return "Q6_K"; case GG_Q8_K: return "Q8_K"; case GG_IQ4_NL: return "IQ4_NL"; case GG_IQ4_XS: return "IQ4_XS";
	case GG_BF16: return "BF16"; case GG_Q2_0: return "Q2_0";
	}
	return "?";
}

/* ---- metadata ---- */
const gguf_kv *gguf_get(const gguf_file *g, const char *key) {
	size_t kl = strlen(key);
	for (uint64_t i = 0; i < g->nkv; i++)
		if (g->kv[i].klen == kl && !memcmp(g->kv[i].key, key, kl)) return &g->kv[i];
	return NULL;
}
int gguf_get_int(const gguf_file *g, const char *key, int64_t *out) {
	const gguf_kv *kv = gguf_get(g, key);
	if (!kv) return -1;
	const uint8_t *p = kv->val;
	switch (kv->type) {
	case 0: case 7: *out = p[0]; return 0;
	case 1: *out = (int8_t)p[0]; return 0;
	case 2: { uint16_t v; memcpy(&v, p, 2); *out = v; return 0; }
	case 3: { int16_t v; memcpy(&v, p, 2); *out = v; return 0; }
	case 4: { uint32_t v; memcpy(&v, p, 4); *out = v; return 0; }
	case 5: { int32_t v; memcpy(&v, p, 4); *out = v; return 0; }
	case 10: case 11: { int64_t v; memcpy(&v, p, 8); *out = v; return 0; }
	default: return -1;
	}
}
const char *gguf_get_str(const gguf_file *g, const char *key, uint64_t *len) {
	const gguf_kv *kv = gguf_get(g, key);
	if (!kv || kv->type != 8) return NULL;
	uint64_t n; memcpy(&n, kv->val, 8);
	if (len) *len = n;
	return (const char *)kv->val + 8;
}
int gguf_get_arr(const gguf_file *g, const char *key, uint32_t *etype, uint64_t *n, const uint8_t **data) {
	const gguf_kv *kv = gguf_get(g, key);
	if (!kv || kv->type != 9) return -1;
	memcpy(etype, kv->val, 4); memcpy(n, kv->val + 4, 8);
	*data = kv->val + 12;
	return 0;
}
