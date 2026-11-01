/* flashd: the HTTP front-end of the Flash-Next engine, speaking the subset of llama-server's API
 * that https://nm-z.com/chat/ uses:
 *
 *   GET  /v1/models            {"data":[{"id": name}]}
 *   GET  /props               {"default_generation_settings":{"n_ctx": positions}, "chat_template": ...}
 *   GET  /health
 *   POST /v1/chat/completions  {"messages":[...], "stream":bool, "max_tokens":n,
 *                               "chat_template_kwargs":{"enable_thinking":bool}}
 *
 * The prompt is the GGUF's own chat template (jinja.c) over the request's messages, tokenized
 * with the GGUF's vocabulary (tok.c), and run on the engine (flash, FLASH_SOCK) greedily until
 * <|im_end|>/EOS or max_tokens. The reply streams as server-sent events: reasoning tokens as
 * delta.reasoning_content up to </think>, the answer as delta.content, and the final event
 * carries timings (prompt_per_second, predicted_per_second). One request runs at a time; the
 * others wait in a FIFO queue (see "the request queue"; GET endpoints never wait).
 *
 *   flashd <shard-1.gguf> <flash-socket> <host:port>      env FLASHD_NAME, FLASHD_THINKING (0: off by default),
 *   FLASHD_QUEUE_MAX (waiting requests, default 8), FLASHD_QUEUE_WAIT_S (default 600), FLASHD_CONCURRENT (requests running at once, default 1),
 *   FLASHD_REQUEST_MAX_S (default 300 without vision, 2400 with FLASHVIS_SOCK; 0 disables the limit) */
#define _GNU_SOURCE
#include "jinja.h"
#include "timings.h"
#include "tok.h"
#include "vision.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

static tok_vocab *V;
static jinja *TPL;
static const char *SOCK, *NAME = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-sass";
static int POSITIONS = 2048, THINK_DEFAULT = 1;
enum { FLASHD_TEXT_REQUEST_MAX_S = 300, FLASHD_VISION_REQUEST_MAX_S = 2400 };
static int request_max_seconds(const char *configured, int vision_enabled) {
    if (configured) return atoi(configured);
    return vision_enabled ? FLASHD_VISION_REQUEST_MAX_S : FLASHD_TEXT_REQUEST_MAX_S;
}
// Output budgets in thinking mode: up to 262,144 tokens of reasoning and 131,072 of final answer (the context bounds both).
static long REASONING_MAX = 262144, ANSWER_MAX = 131072;
static double DEF_TEMP = 0.0, DEF_TOPP = 1.0; static int DEF_TOPK = 0;   /* general.sampling.* from the GGUF; temperature 0 = greedy */
static uint32_t ID_END, ID_THINK_END;
static unsigned long long request_sequence;
static double CANCEL_DRAIN_MS = 60000.;
static double http_clock_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static void http_optional_ms(sbuf *b, const char *key, double stamp, double origin) {
    sb_printf(b, ",\"%s\":", key);
    if (stamp > 0) sb_printf(b, "%.17g", stamp - origin); else sb_puts(b, "null");
}
static void http_write_log(unsigned long long request_id,double origin,int attempted,int failed,int closed) {
    char elapsed[48];
    flash_timing_number(elapsed,http_clock_ms()-origin,attempted);
    fprintf(stderr,"flashd_request_timing: {\"request_id\":\"%ld-%llu\",\"clock\":\"CLOCK_MONOTONIC\",\"response_write_returned_ms\":%s,\"write_attempted\":%s,\"failed\":%s,\"client_closed\":%s}\n",
        (long)getpid(),request_id,elapsed,attempted?"true":"false",failed?"true":"false",closed?"true":"false");
}

/* ---- output ---- */
/* Each HTTP worker owns one absolute budget. Partial I/O never renews it. */
static __thread double request_deadline;
static int deadline_expired(double deadline) {
    return deadline > 0 && http_clock_ms() >= deadline;
}
static int io_ready(int fd, short events, double deadline) {
    for (;;) {
        double left = deadline - http_clock_ms();
        if (left <= 0) { errno = ETIMEDOUT; return -1; }
        int wait_ms = left >= INT_MAX ? INT_MAX : (int)ceil(left);
        struct pollfd p = { .fd = fd, .events = events };
        int ready = poll(&p, 1, wait_ms);
        if (ready > 0) return 0;
        if (ready < 0 && errno == EINTR) continue;
        if (!ready) errno = ETIMEDOUT;
        return -1;
    }
}
static int wr_until(int fd, const char *s, size_t n, double deadline) {
    while (n) {
        if (deadline_expired(deadline)) { errno = ETIMEDOUT; return -1; }
        ssize_t k = send(fd, s, n, MSG_NOSIGNAL | (deadline > 0 ? MSG_DONTWAIT : 0));
        if (k > 0) { s += k; n -= (size_t)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && deadline > 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!io_ready(fd, POLLOUT, deadline)) continue;
        }
        return -1;
    }
    return 0;
}
static int wr(int fd, const char *s, size_t n) {
    return wr_until(fd, s, n, request_deadline);
}
static ssize_t rd(int fd, void *buf, size_t n) {
    for (;;) {
        if (deadline_expired(request_deadline)) { errno = ETIMEDOUT; return -1; }
        ssize_t got = recv(fd, buf, n, request_deadline > 0 ? MSG_DONTWAIT : 0);
        if (got >= 0) return got;
        if (errno == EINTR) continue;
        if (request_deadline > 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!io_ready(fd, POLLIN, request_deadline)) continue;
        }
        return -1;
    }
}
static int wrs(int fd, const char *s) { return wr(fd, s, strlen(s)); }
static int chunk(int fd, const char *s, size_t n) { char h[24]; int k = snprintf(h, sizeof h, "%zx\r\n", n); return wr(fd, h, (size_t)k) || wr(fd, s, n) || wr(fd, "\r\n", 2); }
static int reply(int fd, int status, const char *ctype, const char *body, size_t n) {
	char h[256]; const char *r = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : status == 413 ? "Payload Too Large" : status == 504 ? "Gateway Timeout" : "Internal Server Error";
	int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nCache-Control: no-store\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, r, ctype, n);
	return wr(fd, h, (size_t)k) || wr(fd, body, n) ? -1 : 0;
}
static void error(int fd, int status, const char *msg) {
	sbuf b = {0}; sb_puts(&b, "{\"error\":{\"message\":"); sb_json_str(&b, msg, strlen(msg)); sb_printf(&b, ",\"code\":%d}}", status);
	reply(fd, status, "application/json", b.s, b.n); sb_free(&b);
}


/* FIFO keepalives may already have committed SSE headers. Do not send a
   second HTTP status line when admission is followed by a validation error. */
static void admitted_error(int fd, int headers_sent, int status, const char *msg) {
    if (!headers_sent) { error(fd, status, msg); return; }
    sbuf b = {0};
    sb_puts(&b, "data: {\"error\":{\"message\":");
    sb_json_str(&b, msg, strlen(msg));
    sb_printf(&b, ",\"code\":%d}}\n\n", status);
    if (!chunk(fd, b.s, b.n) && !chunk(fd, "data: [DONE]\n\n", 14))
        wrs(fd, "0\r\n\r\n");
    sb_free(&b);
}

/* ---- the request queue ----
   FLASHD_CONCURRENT requests run at once (default 1: an engine of one sequence), the others wait here in arrival order. A request whose client goes away
   leaves the queue (or, running, stops the engine) instead of being served to nobody; a full queue or a long wait is a 503 with Retry-After.
   FLASHD_QUEUE_MAX (waiting requests, default 8), FLASHD_QUEUE_WAIT_S (default 600). GET /queue shows the depth.
   FLASHD_REQUEST_MAX_S (default 300 without vision, 2400 with FLASHVIS_SOCK; 0: no limit) bounds a whole request from its arrival, the wait and the run together: a request still waiting
   leaves the queue, a running one closes its engine connection (the core aborts the turn when its client goes) and the next one starts. */
typedef struct waiter { struct waiter *next; } waiter;
static pthread_mutex_t QM = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t QC = PTHREAD_COND_INITIALIZER;
static waiter *QHEAD, *QTAIL; static int QBUSY, QLEN, QMAX = 8, QWAIT_S = 600, REQ_MAX_S = FLASHD_TEXT_REQUEST_MAX_S;
static int QCONC = 1;   /* FLASHD_CONCURRENT: requests running at once (an engine that batches sequences); QBUSY counts them */
static int client_gone(int fd) {   /* the peer closed or reset its end */
	struct pollfd p = { .fd = fd, .events = POLLRDHUP };
	return poll(&p, 1, 0) > 0 && (p.revents & (POLLRDHUP | POLLHUP | POLLERR));
}
static void q_remove(waiter *w) {   /* QM held */
	waiter *prev = NULL; for (waiter *x = QHEAD; x; prev = x, x = x->next) if (x == w) { if (prev) prev->next = x->next; else QHEAD = x->next; if (QTAIL == x) QTAIL = prev; QLEN--; break; }
}
static double mono_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int try_send(int fd, const char *s, size_t n) { return send(fd, s, n, MSG_NOSIGNAL | MSG_DONTWAIT) == (ssize_t)n ? 0 : -1; }
/* 0: the engine is yours (queue_leave when done); -1: the client left; -2: the queue is full; -3: waited too long; -4: the request deadline
   (mono_ms, 0: none) passed. *hdr is set once SSE headers have gone out while waiting. */
static int queue_enter(int fd, int stream, int *hdr, double *waited_ms, double deadline) {
	double t0 = mono_ms(), last_note = t0; waiter me = { 0 };
	pthread_mutex_lock(&QM);
    if (deadline_expired(deadline)) { pthread_mutex_unlock(&QM); *waited_ms = 0; return -4; }
	if (QBUSY < QCONC && !QHEAD) { QBUSY++; pthread_mutex_unlock(&QM); *waited_ms = 0; return 0; }
	if (QLEN >= QMAX) { pthread_mutex_unlock(&QM); *waited_ms = 0; return -2; }
	if (QTAIL) QTAIL->next = &me; else QHEAD = &me;
	QTAIL = &me; QLEN++;
	int rc = 0;
	while (!(QHEAD == &me && QBUSY < QCONC)) {
        double wait_ms = 1000.;
        if (deadline > 0) {
            double left = deadline - mono_ms();
            if (left <= 0) { rc = -4; break; }
            if (left < wait_ms) wait_ms = left;
        }
		struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += (long)(wait_ms * 1e6);
        ts.tv_sec += ts.tv_nsec / 1000000000L; ts.tv_nsec %= 1000000000L;
		pthread_cond_timedwait(&QC, &QM, &ts);
		if (QHEAD == &me && QBUSY < QCONC) break;
		double now = mono_ms();
		if (client_gone(fd)) { rc = -1; break; }
		if (QWAIT_S > 0 && now - t0 > QWAIT_S * 1e3) { rc = -3; break; }
		if (deadline > 0 && now >= deadline) { rc = -4; break; }
		if (stream && now - last_note > 10e3) {   /* keep the connection open through proxies, and say where it stands */
			int pos = 1; for (waiter *x = QHEAD; x && x != &me; x = x->next) pos++;
			char c[160], m[96]; int mn = snprintf(m, sizeof m, ": queued, position %d\n\n", pos), k = 0;
			if (!*hdr) { const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"; k = snprintf(c, sizeof c, "%s", h); }
			k += snprintf(c + k, sizeof c - (size_t)k, "%x\r\n%s\r\n", mn, m);
			if (try_send(fd, c, (size_t)k)) { rc = -1; break; }
			*hdr = 1; last_note = now;
		}
	}
	if (!rc && deadline > 0 && mono_ms() >= deadline) rc = -4;   /* its turn came too late: leave rather than start a cancelled run */
	if (rc) q_remove(&me);
	else { QHEAD = me.next; if (!QHEAD) QTAIL = NULL; QLEN--; QBUSY++; }
	pthread_cond_broadcast(&QC); pthread_mutex_unlock(&QM);
	*waited_ms = mono_ms() - t0;
	return rc;
}
static void queue_leave(void) { pthread_mutex_lock(&QM); QBUSY--; pthread_cond_broadcast(&QC); pthread_mutex_unlock(&QM); }
static void busy(int fd, const char *msg, int retry_after) {
	sbuf b = {0}; sb_puts(&b, "{\"error\":{\"message\":"); sb_json_str(&b, msg, strlen(msg)); sb_puts(&b, ",\"code\":503}}");
	char h[256]; int k = snprintf(h, sizeof h, "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: %d\r\nCache-Control: no-store\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", retry_after, b.n);
	wr(fd, h, (size_t)k); wr(fd, b.s, b.n); sb_free(&b);
}
/* FLASHD_REQUEST_MAX_S ran out: an error event ending the stream (whose headers went out once hdr is set), or a 504 */
static void timed_out_record(int fd, int stream, int hdr, const char *timings) {
    double saved_deadline = request_deadline;
    request_deadline = http_clock_ms() + 1000.; /* bounded best-effort error delivery */
	char m[96]; snprintf(m, sizeof m, "the request took longer than %d s and was cancelled", REQ_MAX_S);
    sbuf b = {0};
	if (stream && hdr) {
        sb_puts(&b, "data: ");
    }
    sb_puts(&b, "{\"error\":{\"message\":"); sb_json_str(&b, m, strlen(m));
    sb_puts(&b, ",\"type\":\"request_timeout\",\"code\":504}");
    if (timings) sb_printf(&b, ",\"timings\":%s", timings);
    sb_putc(&b, '}');
    if (stream && hdr) {
        sb_puts(&b, "\n\n");
        if (!chunk(fd, b.s, b.n) && !chunk(fd, "data: [DONE]\n\n", 14)) wrs(fd, "0\r\n\r\n");
    } else reply(fd, 504, "application/json", b.s, b.n);
    sb_free(&b);
    request_deadline = saved_deadline;
}
static void timed_out(int fd, int stream, int hdr) {
    timed_out_record(fd, stream, hdr, NULL);
}

/* ---- the engine socket ---- */
/* A resident-engine validation can pin every backend connection to the
   process covered by its external GPU lease. Ordinary serving is unchanged. */
static int engine_peer_matches(int fd) {
    const char *expected = getenv("FLASHD_EXPECT_CORE_PID");
    if (!expected) return 1;
    if (!*expected || strspn(expected, "0123456789") != strlen(expected)) return 0;
    char *end; errno = 0; long pid = strtol(expected, &end, 10);
    if (errno || *end || pid <= 1 || pid > INT_MAX) return 0;
    struct ucred peer; socklen_t size = sizeof peer;
    return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0
        && size == sizeof peer && peer.pid == pid && peer.uid == getuid();
}
static int unix_open(const char *path) {
    int s; struct sockaddr_un sa = { .sun_family = AF_UNIX };
	snprintf(sa.sun_path, sizeof sa.sun_path, "%s", path);
    for (;;) {
        if (deadline_expired(request_deadline)) { errno = ETIMEDOUT; return -1; }
        s = socket(AF_UNIX, SOCK_STREAM | (request_deadline > 0 ? SOCK_NONBLOCK : 0), 0);
        if (s < 0) return -1;
        if (!connect(s, (struct sockaddr *)&sa, sizeof sa)) break;
        int error_code = errno;
        if (request_deadline > 0 && (error_code == EINPROGRESS || error_code == EALREADY)) {
            if (!io_ready(s, POLLOUT, request_deadline)) {
                socklen_t size = sizeof error_code;
                if (!getsockopt(s, SOL_SOCKET, SO_ERROR, &error_code, &size) && !error_code) break;
                if (!error_code) error_code = errno;
            } else error_code = errno;
        }
        close(s);
        /* AF_UNIX EAGAIN means the listener backlog is full, not a pending
           connection. Recreate and retry; SO_ERROR==0 would not prove connection. */
        if (request_deadline > 0 && (error_code == EAGAIN || error_code == EWOULDBLOCK || error_code == EINTR)) {
            double left = request_deadline - http_clock_ms();
            if (left <= 0) { errno = ETIMEDOUT; return -1; }
            poll(NULL, 0, left > 10. ? 10 : (int)ceil(left));
            continue;
        }
        errno = error_code; return -1;
    }
    if (request_deadline > 0) {
        int flags = fcntl(s, F_GETFL, 0);
        if (flags < 0 || fcntl(s, F_SETFL, flags & ~O_NONBLOCK)) { int e = errno; close(s); errno = e; return -1; }
    }
    return s;
}
static FILE *engine_open(FILE **out) {
    *out = NULL;
    int s = unix_open(SOCK);
    if (s < 0) return NULL;
	if (!engine_peer_matches(s)) { close(s); errno = EPERM; return NULL; }
    int output_fd = dup(s);
    if (output_fd < 0) { close(s); return NULL; }
    FILE *in = fdopen(s, "r");
    if (!in) { close(output_fd); close(s); return NULL; }
    *out = fdopen(output_fd, "w");
    if (!*out) { close(output_fd); fclose(in); return NULL; }
    return in;
}

/* Read complete protocol lines with one absolute deadline, including partial
   writes. stdio fgets + SO_RCVTIMEO can otherwise wait forever on a slow line. */
typedef struct { char bytes[8192]; size_t used; } engine_reader;
static int engine_line(int fd,engine_reader *r,char *line,size_t cap,double deadline) {
    for(;;) {
        double left=deadline-http_clock_ms();if(left<=0)return -2;
        char *end=memchr(r->bytes,'\n',r->used);
        if(end) {
            size_t n=(size_t)(end-r->bytes)+1;if(n>=cap)return -1;
            memcpy(line,r->bytes,n);line[n]=0;r->used-=n;
            memmove(r->bytes,r->bytes+n,r->used);return 1;
        }
        if(r->used==sizeof r->bytes)return -1;
        struct pollfd p={.fd=fd,.events=POLLIN};
        int ready=poll(&p,1,(int)ceil(left));
        if(ready<0){if(errno==EINTR)continue;return -1;}
        if(!ready)return -2;
        ssize_t n=recv(fd,r->bytes+r->used,sizeof r->bytes-r->used,MSG_DONTWAIT);
        if(n<0){if(errno==EINTR||errno==EAGAIN||errno==EWOULDBLOCK)continue;return -1;}
        if(!n)return r->used?-1:0;
        r->used+=(size_t)n;
    }
}

/* ---- one reply being decoded ---- */
typedef struct {
	int fd, stream, in_think, gone, started_content, started_reasoning;
	unsigned char pend[8]; int npend;
	sbuf content, reasoning;
	long created;
	double request_start_ms, first_text_send_ms;
	long n_reason, n_answer; int cap_hit;   /* tokens so far in each part; set when one reaches its budget */
} gen;

static void emit_delta(gen *g, const char *field, const char *s, size_t n) {
	if (!n) return;
	if (g->stream && !g->gone) {
		sbuf b = {0};
		sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"%s\":", field); sb_json_str(&b, s, n);
		sb_printf(&b, "},\"finish_reason\":null}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\"}\n\n", g->created, NAME);
		if (chunk(g->fd, b.s, b.n)) g->gone = 1;
        else if (!g->first_text_send_ms) g->first_text_send_ms = http_clock_ms();
		sb_free(&b);
	}
	sbuf *acc = field[0] == 'r' ? &g->reasoning : &g->content;
	sb_put(acc, s, n);
}
/* text from the model: drop the whitespace that opens each part, hold back an incomplete UTF-8 tail */
static void emit_text(gen *g, const char *s, size_t n) {
	unsigned char buf[16 + 4096]; size_t k = 0;
	if (g->npend) memcpy(buf, g->pend, (size_t)g->npend), k = (size_t)g->npend;
	if (n > 4096) n = 4096;
	memcpy(buf + k, s, n); k += n;
	size_t keep = k;   /* an incomplete trailing character waits for its next bytes */
	for (size_t back = 1; back <= 3 && back <= k; back++) {
		unsigned char c = buf[k - back];
		if ((c & 0xc0) == 0x80) continue;
		size_t need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
		if (need > back) keep = k - back;
		break;
	}
	g->npend = (int)(k - keep); memcpy(g->pend, buf + keep, (size_t)g->npend);
	const char *p = (const char *)buf; size_t m = keep;
	int *started = g->in_think ? &g->started_reasoning : &g->started_content;
	while (!*started && m && (*p == '\n' || *p == ' ' || *p == '\r' || *p == '\t')) { p++; m--; }
	if (m) *started = 1;
	emit_delta(g, g->in_think ? "reasoning_content" : "content", p, m);
}
static void on_token(gen *g, uint32_t id) {
	if (id == ID_THINK_END) { g->in_think = 0; return; }
	if (tok_is_hidden(V, id)) return;   /* control and unknown tokens only: <tool_call> and </tool_call> are USER_DEFINED and are part of the answer */
	if (g->in_think ? ++g->n_reason > REASONING_MAX : ++g->n_answer > ANSWER_MAX) { g->cap_hit = 1; return; }
	char piece[256]; int n = tok_piece(V, id, piece, sizeof piece);
	if (n > 0) emit_text(g, piece, (size_t)n);
}

/* ---- images: the vision encoder daemon (flashvis) and its cache ----
   A request's image parts (data: URLs) are decoded, run through flashvis (GPU Flash Attention) and the embeddings are kept in files
   under IMGDIR keyed by a hash of the image bytes, so the images of earlier turns, which the page sends again with every question, cost nothing the
   second time. The engine reads those files (the `img` command). */
static const char *VIS_SOCK;               /* FLASHVIS_SOCK: the encoder; NULL: no images */
static char IMGDIR[200] = "/var/tmp/flash-img";
static int ID_IMGPAD = -1;
static unsigned IMG_ROWS = 0;              /* image tokens one response can hold (the engine's FLASH_MAXIMG), from `info` */
static int b64val(int c) { return c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1; }
/* base64 (standard or URL alphabet, padding optional) -> bytes; returns the length or -1 */
static long b64decode(const char *s, size_t n, unsigned char *out) {
	size_t o = 0; unsigned acc = 0; int bits = 0;
	for (size_t i = 0; i < n; i++) {
		if (s[i] == '=' || s[i] == '\n' || s[i] == '\r' || s[i] == ' ') continue;
		int v = b64val((unsigned char)s[i]); if (v < 0) return -1;
		acc = (acc << 6) | (unsigned)v; bits += 6;
		if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)(acc >> bits); acc &= (1u << bits) - 1; }
	}
	return (long)o;
}
static unsigned long long fnv64(const unsigned char *p, size_t n) { unsigned long long h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h ^ ((unsigned long long)n << 1); }
/* the cache file of an image (written when missing): its header gives the token count and the positions the image takes.
 * Returns 0, or an error text. */
typedef struct { char path[240]; unsigned ntok, npos; } imgrec;
static const char *vis_image(const unsigned char *bytes, size_t n, imgrec *r) {
	if (!VIS_SOCK) return "this engine was started without a vision encoder";
	char *dir = IMGDIR; mkdir(dir, 0755);
	snprintf(r->path, sizeof r->path, "%s/" FLASH_IMAGE_CACHE_PREFIX "%016llx.bin", dir, fnv64(bytes, n));
	FILE *f = fopen(r->path, "rb"); uint32_t hd[5];
	if (f) { int ok = fread(hd, 4, 5, f) == 5; fclose(f); if (ok) { r->ntok = hd[0]; r->npos = hd[3]; utime(r->path, NULL); return NULL; } }
	int s = unix_open(VIS_SOCK);
	if (s < 0) return "the vision encoder is not running";
	char head[64]; int hn = snprintf(head, sizeof head, "enc %zu\n", n);
	if (wr(s, head, (size_t)hn) || wr(s, (const char *)bytes, n)) { close(s); return "the vision encoder closed the connection"; }
	char line[160]; size_t ll = 0;
	while (ll + 1 < sizeof line) { char c; if (rd(s, &c, 1) != 1) { close(s); return "the vision encoder closed the connection"; } if (c == '\n') break; line[ll++] = c; }
	line[ll] = 0;
	unsigned ntok, nx, ny, dim; long npos;
	if (sscanf(line, "ok %u %u %u %ld %u", &ntok, &nx, &ny, &npos, &dim) != 5) { close(s); static char em[200]; snprintf(em, sizeof em, "the vision encoder: %.150s", line); return em; }
	size_t body = (size_t)ntok * dim * 4 + (size_t)ntok * 12; unsigned char *buf = malloc(body); size_t got = 0;
	while (got < body) { ssize_t k = rd(s, buf + got, body - got); if (k <= 0) break; got += (size_t)k; }
	close(s);
	if (got != body || dim != 2560) { free(buf); return "the vision encoder sent a short answer"; }
	char tmp[260]; snprintf(tmp, sizeof tmp, "%s.%d.tmp", r->path, (int)getpid());
	f = fopen(tmp, "wb"); uint32_t h5[5] = { ntok, nx, ny, (uint32_t)npos, dim };
	int ok = f && fwrite(h5, 4, 5, f) == 5 && fwrite(buf, 1, body, f) == body; if (f) fclose(f);
	free(buf);
	if (!ok || rename(tmp, r->path)) { unlink(tmp); return "cannot write the image cache"; }
	r->ntok = ntok; r->npos = (unsigned)npos;
	/* keep the newest 24 files */
	{ DIR *d = opendir(dir); struct dirent *e; struct { time_t t; char name[64]; } keep[64]; int nk = 0;
	  while (d && (e = readdir(d))) { if (e->d_name[0] == '.' || strstr(e->d_name, ".tmp")) continue; struct stat st; char p[300]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name); if (stat(p, &st)) continue; if (nk < 64) { keep[nk].t = st.st_mtime; snprintf(keep[nk].name, sizeof keep[nk].name, "%s", e->d_name); nk++; } }
	  if (d) closedir(d);
	  for (int guard = 0; nk > 24 && guard < 64; guard++) { int old = 0; for (int i = 1; i < nk; i++) if (keep[i].t < keep[old].t) old = i; char p[300]; snprintf(p, sizeof p, "%s/%s", dir, keep[old].name); unlink(p); keep[old] = keep[--nk]; } }
	return NULL;
}

/* ---- POST /v1/chat/completions ---- */
static void chat(int fd, const char *body, size_t bn, double request_start_ms) {
    unsigned long long request_id = __sync_add_and_fetch(&request_sequence, 1);
    double chat_start_ms = http_clock_ms(), tokenization_ms = 0;
    double deadline = REQ_MAX_S > 0 ? request_start_ms + REQ_MAX_S * 1e3 : 0;
    request_deadline = deadline;
	arena *a = arena_new(); const char *je = NULL;
	jv *req = json_parse(a, body, bn, &je);
	if (!req || req->k != J_DICT) { error(fd, 400, "the body is not a JSON object"); arena_free(a); return; }
	jv *msgs = jv_getz(req, "messages");
	if (!msgs || msgs->k != J_LIST || !msgs->l.n) { error(fd, 400, "messages must be a non-empty list"); arena_free(a); return; }
	/* The GGUF chat template takes one system message, first; Codex sends several (its developer instructions later in the
	   conversation). Every system and developer message is folded, in order, into one leading system message. */
	{
		sbuf sys = {0}; int nsys = 0; jv *rest = jv_new(a, J_LIST);
		for (int i = 0; i < msgs->l.n; i++) {
			jv *m = msgs->l.a[i], *r = jv_getz(m, "role");
			int issys = r && r->k == J_STR && ((r->s.n == 6 && !memcmp(r->s.s, "system", 6)) || (r->s.n == 9 && !memcmp(r->s.s, "developer", 9)));
			if (!issys) { jv_push(a, rest, m); continue; }
			jv *c = jv_getz(m, "content"); nsys++;
			if (c && c->k == J_STR && c->s.n) { if (sys.n) sb_puts(&sys, "\n\n"); sb_put(&sys, c->s.s, c->s.n); }
			else if (c && c->k == J_LIST)
				for (int j = 0; j < c->l.n; j++) {
					jv *t = jv_getz(c->l.a[j], "text");
					if (t && t->k == J_STR && t->s.n) { if (sys.n) sb_puts(&sys, "\n\n"); sb_put(&sys, t->s.s, t->s.n); }
				}
		}
		if (nsys) {
			jv *l2 = jv_new(a, J_LIST), *sm = jv_new(a, J_DICT);
			jv_set(a, sm, "role", 4, jv_str(a, "system", 6)); jv_set(a, sm, "content", 7, jv_str(a, sys.s ? sys.s : "", sys.n));
			jv_push(a, l2, sm);
			for (int i = 0; i < rest->l.n; i++) jv_push(a, l2, rest->l.a[i]);
			msgs = l2;
		}
		sb_free(&sys);
	}
	/* The sampler has no penalties: the thinking-mode settings presence_penalty 0.0 and repetition_penalty 1.0 are what it does, so
	   those values are accepted and any other is refused rather than silently ignored. */
	{
		static const struct { const char *key; double fixed; } pen[] = { { "presence_penalty", 0.0 }, { "frequency_penalty", 0.0 }, { "repetition_penalty", 1.0 }, { "repeat_penalty", 1.0 } };
		for (size_t i = 0; i < sizeof pen / sizeof pen[0]; i++) {
			jv *x = jv_getz(req, pen[i].key); double v = x && x->k == J_NUM ? x->d : x && x->k == J_INT ? (double)x->i : pen[i].fixed;
			if (v != pen[i].fixed) { char m[160]; snprintf(m, sizeof m, "%s is fixed at %g (the engine has no penalties)", pen[i].key, pen[i].fixed); error(fd, 400, m); arena_free(a); return; }
		}
	}
	/* Image parts: data: URLs, decoded and encoded by flashvis (cached by hash); the template only needs to see that there is an image,
	   so the data is dropped from the variables before rendering. */
	imgrec imgs[8]; int nim = 0;
	for (int i = 0; i < msgs->l.n; i++) {
		jv *c = jv_getz(msgs->l.a[i], "content");
		if (!c || c->k != J_LIST) continue;
		for (int j = 0; j < c->l.n; j++) {
			jv *part = c->l.a[j], *ty = jv_getz(part, "type");
			if (!(ty && ty->k == J_STR && ((ty->s.n == 9 && !memcmp(ty->s.s, "image_url", 9)) || (ty->s.n == 5 && !memcmp(ty->s.s, "image", 5))))) continue;
			if (nim >= 8) { error(fd, 400, "at most 8 images per request"); arena_free(a); return; }
			jv *iu = jv_getz(part, "image_url"); const char *url = NULL; size_t un = 0;
			if (iu && iu->k == J_STR) { url = iu->s.s; un = iu->s.n; } else if (iu && iu->k == J_DICT) { jv *u = jv_getz(iu, "url"); if (u && u->k == J_STR) { url = u->s.s; un = u->s.n; } }
			const char *comma = url && un > 5 && !memcmp(url, "data:", 5) ? memchr(url, ',', un < 200 ? un : 200) : NULL;
			if (!comma || !memmem(url, (size_t)(comma - url), ";base64", 7)) { error(fd, 400, "an image must be a data: URL with base64 data"); arena_free(a); return; }
			unsigned char *raw = malloc(un); long rn = b64decode(comma + 1, un - (size_t)(comma + 1 - url), raw);
			if (rn <= 0) { free(raw); error(fd, 400, "the image is not valid base64"); arena_free(a); return; }
			const char *ie = vis_image(raw, (size_t)rn, &imgs[nim]); free(raw);
            if (deadline_expired(deadline)) { timed_out(fd, 0, 0); arena_free(a); return; }
			if (ie) { error(fd, 503, ie); arena_free(a); return; }
			if (iu->k == J_DICT) jv_set(a, iu, "url", 3, jv_str(a, "", 0)); else jv_set(a, part, "image_url", 9, jv_str(a, "", 0));
			nim++;
		}
	}
	unsigned img_tokens = 0; for (int i = 0; i < nim; i++) img_tokens += imgs[i].ntok;
	if (img_tokens > IMG_ROWS) { char m[160]; snprintf(m, sizeof m, "the images take %u tokens; one response holds %u", img_tokens, IMG_ROWS); error(fd, 400, m); arena_free(a); return; }
	/* OpenAI clients send a tool call's arguments as a JSON string; the template iterates them as a mapping */
	for (int i = 0; i < msgs->l.n; i++) {
		jv *tcs = jv_getz(msgs->l.a[i], "tool_calls");
		if (!tcs || tcs->k != J_LIST) continue;
		for (int j = 0; j < tcs->l.n; j++) {
			jv *f = jv_getz(tcs->l.a[j], "function"); if (!f || f->k != J_DICT) f = tcs->l.a[j];
			jv *ar = jv_getz(f, "arguments"); const char *e2 = NULL;
			if (!ar || ar->k != J_STR) continue;
			jv *d = ar->s.n ? json_parse(a, ar->s.s, ar->s.n, &e2) : jv_new(a, J_DICT);
			if (d && d->k == J_DICT) jv_set(a, f, "arguments", 9, d);
		}
	}
	jv *vars = jv_new(a, J_DICT);
	jv_set(a, vars, "messages", 8, msgs);
	jv_set(a, vars, "add_generation_prompt", 21, jv_bool(a, 1));
	jv *tools = jv_getz(req, "tools"); if (tools && tools->k == J_LIST && tools->l.n) jv_set(a, vars, "tools", 5, tools);
	jv *kw = jv_getz(req, "chat_template_kwargs");
	jv *think = kw ? jv_getz(kw, "enable_thinking") : NULL;
	if (think && think->k == J_BOOL) jv_set(a, vars, "enable_thinking", 15, think);
	else if (!THINK_DEFAULT) jv_set(a, vars, "enable_thinking", 15, jv_bool(a, 0));
	jv *eff = kw ? jv_getz(kw, "reasoning_effort") : NULL;
	if (!eff) eff = jv_getz(req, "reasoning_effort");   /* the OpenAI spelling */
	if (eff && eff->k == J_STR) jv_set(a, vars, "reasoning_effort", 16, eff);
	double temp = DEF_TEMP, topp = DEF_TOPP; long topk = DEF_TOPK; unsigned long long seed = 0; int have_seed = 0;
	{ jv *x;
	  if ((x = jv_getz(req, "temperature")) && (x->k == J_NUM || x->k == J_INT)) temp = x->k == J_NUM ? x->d : (double)x->i;
	  if ((x = jv_getz(req, "top_p")) && (x->k == J_NUM || x->k == J_INT)) topp = x->k == J_NUM ? x->d : (double)x->i;
	  if ((x = jv_getz(req, "top_k")) && x->k == J_INT) topk = (long)x->i;
	  if ((x = jv_getz(req, "seed")) && x->k == J_INT && x->i >= 0) { seed = (unsigned long long)x->i; have_seed = 1; } }
	if (!have_seed) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); seed = (unsigned long long)ts.tv_nsec * 1000003ull ^ (unsigned long long)ts.tv_sec * 0x9e3779b97f4a7c15ull ^ (unsigned long long)(uintptr_t)body; }
	int stream = 0; jv *sv = jv_getz(req, "stream"); if (sv && sv->k == J_BOOL) stream = sv->b;
	/* llama-server's return_progress: while the prompt runs, chunks with no choices carry
	   "prompt_progress":{"total","cache","processed","time_ms"} (the engine's "prog" lines, at most one every 500 ms) */
	int progress = 0; jv *rp = jv_getz(req, "return_progress"); if (rp && rp->k == J_BOOL) progress = rp->b;
	long max_tokens = 512; jv *mt = jv_getz(req, "max_tokens"); if (!mt) mt = jv_getz(req, "max_completion_tokens");
	if (mt && mt->k == J_INT && mt->i > 0) max_tokens = (long)mt->i;

	char err[512]; size_t pn;
	char *prompt = jinja_render(TPL, a, vars, &pn, err, sizeof err);
	if (!prompt) { error(fd, 400, err); arena_free(a); return; }

    double queue_start_ms = http_clock_ms();
	int hdr = 0; double qwait = 0;
	int qr = queue_enter(fd, stream, &hdr, &qwait, deadline);
	if (qr) {
		free(prompt); arena_free(a);
		if (qr == -1) fprintf(stderr, "flashd: the client left the queue after %.1f s\n", qwait / 1e3);
		else if (qr == -4) { fprintf(stderr, "flashd: the request reached FLASHD_REQUEST_MAX_S (%d s) while queued (%.1f s): cancelled\n", REQ_MAX_S, qwait / 1e3); timed_out(fd, stream, hdr); }
		else if (qr == -2) { fprintf(stderr, "flashd: queue full (%d waiting): 503\n", QMAX); busy(fd, "the request queue is full", 30); }
		else { fprintf(stderr, "flashd: waited %.0f s in the queue: 503\n", qwait / 1e3); if (hdr) { chunk(fd, "data: {\"error\":{\"message\":\"waited too long in the request queue\",\"type\":\"queue_timeout\"}}\n\n", sizeof("data: {\"error\":{\"message\":\"waited too long in the request queue\",\"type\":\"queue_timeout\"}}\n\n") - 1); chunk(fd, "data: [DONE]\n\n", 14); wrs(fd, "0\r\n\r\n"); } else busy(fd, "waited too long in the request queue", 30); }
		return;
	}
	if (client_gone(fd)) { queue_leave(); free(prompt); arena_free(a); fprintf(stderr, "flashd: the client left while queued (%.1f s); not started\n", qwait / 1e3); return; }
    if (deadline_expired(deadline)) { queue_leave(); free(prompt); arena_free(a); timed_out(fd, stream, hdr); return; }
    double lock_acquired_ms = http_clock_ms(), encode_start_ms = lock_acquired_ms;
	uint32_t *ids = malloc(4 * (pn + 16));
	int np = tok_encode(V, prompt, pn, 1, ids, (int)pn + 16);
    tokenization_ms += http_clock_ms() - encode_start_ms;
    if (deadline_expired(deadline)) { queue_leave(); free(ids); free(prompt); arena_free(a); timed_out(fd, stream, hdr); return; }
	if (np < 1 || np + 1 >= POSITIONS) {
		queue_leave(); free(ids); free(prompt); arena_free(a);
		admitted_error(fd, hdr, 400, np < 1 ? "empty prompt" : "the prompt is longer than the engine's context"); return;
	}
	/* the engine returns n_new + 1 tokens (the one predicted at the last prompt position, then n_new more) */
	long n_new = max_tokens - 1 < POSITIONS - np ? max_tokens - 1 : POSITIONS - np;
	/* the prefix cache: the next turn of this conversation will start with this prompt up to the assistant turn it appends, so
	   render the conversation plus an assistant message and keep the tokens the two share (at least one prompt token must run) */
	int snap = 0;
	{
		jv *l2 = jv_new(a, J_LIST), *am = jv_new(a, J_DICT);
		for (int i = 0; i < msgs->l.n; i++) jv_push(a, l2, msgs->l.a[i]);
		jv_set(a, am, "role", 4, jv_str(a, "assistant", 9)); jv_set(a, am, "content", 7, jv_str(a, "x", 1)); jv_push(a, l2, am);
		jv_set(a, vars, "messages", 8, l2);
		size_t pn2; char *p2 = jinja_render(TPL, a, vars, &pn2, err, sizeof err);
		jv_set(a, vars, "messages", 8, msgs);
		if (p2) {
            encode_start_ms = http_clock_ms();
			uint32_t *ids2 = malloc(4 * (pn2 + 16)); int n2 = tok_encode(V, p2, pn2, 1, ids2, (int)pn2 + 16);
            tokenization_ms += http_clock_ms() - encode_start_ms;
			int l = 0; while (l < n2 && l < np && ids2[l] == ids[l]) l++;
			snap = l < np - 1 ? l : np - 1;
			if (snap < 16) snap = 0;
			free(ids2); free(p2);
		}
	}
	/* images: every <|image_pad|> placeholder becomes one token per embedding row (the engine swaps in the embeddings), and a snapshot stops
	   before the first image */
	int istart[8] = { 0 };
	if (nim) {
		int found = 0, first_img = -1; for (int i = 0; i < np; i++) if ((int)ids[i] == ID_IMGPAD) found++;
		if (found != nim || ID_IMGPAD < 0) { queue_leave(); free(ids); free(prompt); arena_free(a); admitted_error(fd, hdr, 400, "the chat template did not place one image token per image"); return; }
		uint32_t *ex = malloc(4 * ((size_t)np + img_tokens + 16)); int k = 0, im = 0;
		for (int i = 0; i < np; i++) {
			if ((int)ids[i] == ID_IMGPAD) { if (first_img < 0) first_img = i; istart[im] = k; for (unsigned q = 0; q < imgs[im].ntok; q++) ex[k++] = ids[i]; im++; }
			else ex[k++] = ids[i];
		}
		free(ids); ids = ex; np = k;
		if (snap > first_img) snap = first_img < 16 ? 0 : first_img;
		if (np + 1 >= POSITIONS) { queue_leave(); free(ids); free(prompt); arena_free(a); admitted_error(fd, hdr, 400, "the prompt is longer than the engine's context"); return; }
		n_new = max_tokens - 1 < POSITIONS - np ? max_tokens - 1 : POSITIONS - np;
	}
    if (deadline_expired(deadline)) { queue_leave(); free(ids); free(prompt); arena_free(a); timed_out(fd, stream, hdr); return; }
	FILE *out = NULL, *in = engine_open(&out);
	if (!in) {
        int expired = deadline_expired(deadline) || errno == ETIMEDOUT;
        queue_leave(); free(ids); free(prompt); arena_free(a);
        if (expired) timed_out(fd, stream, hdr); else admitted_error(fd, hdr, 500, "the engine is not running");
        return;
    }
    /* stdio can block indefinitely while flushing a large prompt to a core
       that stopped reading. Preserve partial sends under the absolute budget. */
    sbuf command = {0};
    sb_puts(&command, "timings 1\ncancel_on_eof 1\n");
	sb_printf(&command, "stop %u %u\n", ID_END, tok_eos(V) == ID_END ? 0xffffffffu : tok_eos(V));
	sb_printf(&command, "sample %llu %.6g %ld %.6g\n", seed, temp, topk <= 0 ? 32 : topk, topp);
	for (int i = 0; i < nim; i++) sb_printf(&command, "img %s %d\n", imgs[i].path, istart[i]);
	if (snap) sb_printf(&command, "snap %d\n", snap);
	sb_printf(&command, "chat %ld", n_new); for (int i = 0; i < np; i++) sb_printf(&command, " %u", ids[i]); sb_putc(&command, '\n');
    int dispatch_failed = wr_until(fileno(out), command.s, command.n, deadline);
    int dispatch_expired = deadline_expired(deadline) || (dispatch_failed && errno == ETIMEDOUT);
    sb_free(&command);
	free(ids);
    if (dispatch_failed || dispatch_expired) {
        shutdown(fileno(out), SHUT_WR); fclose(out); fclose(in);
        queue_leave(); free(prompt); arena_free(a);
        if (dispatch_expired) timed_out(fd, stream, hdr); else admitted_error(fd, hdr, 502, "the engine stopped reading the request");
        return;
    }

    double engine_submitted_ms = http_clock_ms();
	gen g = { .fd = fd, .stream = stream, .created = (long)time(NULL), .request_start_ms = request_start_ms };
	g.in_think = pn >= 8 && !memcmp(prompt + pn - 8, "<think>\n", 8);
	if (stream) {
		const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
		if (!hdr && wrs(fd, h)) g.gone = 1;
		sbuf b = {0};
		sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\"}\n\n", g.created, NAME);
		if (!g.gone && chunk(fd, b.s, b.n)) g.gone = 1;
		sb_free(&b);
	}
	char line[8192]; int quiet = 0, ngen = 0, ended = 0, resumed = 0, snapped = 0, done = 0, failed = 0, expired = 0; double run_ms = 0, first_ms = 0;
    int raw_token_events = 0, negotiating = 2, cancel_on_eof = 0;
    int cancel_requested = 0, cancel_drain_timed_out = 0;
    double cancel_start_ms = 0;
    double first_socket_observed_ms = 0, last_socket_observed_ms = 0, last_progress_ms = 0;
    sbuf engine_timing = {0};
    engine_reader reader={0};
	/* a long prompt takes minutes before the first token: an idle stream is cut by proxies (Cloudflare closes after 100 s), so an SSE comment goes out
	   every 10 s while the engine is silent */
	for (;;) {
        if (deadline_expired(deadline)) expired = 1;
        if (!g.gone && client_gone(fd)) g.gone=1;
        if (expired && (negotiating || !cancel_on_eof)) { cancel_requested = 1; break; }
        if ((g.gone || g.cap_hit || expired) && !cancel_requested && !negotiating) {
            cancel_requested=1;
            if (!cancel_on_eof || shutdown(fileno(out),SHUT_WR)) break;
            cancel_start_ms=http_clock_ms();
        }
        double cleanup_deadline = cancel_start_ms ? cancel_start_ms + CANCEL_DRAIN_MS : 0;
        if (cleanup_deadline && deadline > 0 && deadline + 2000. < cleanup_deadline) cleanup_deadline = deadline + 2000.;
        if (cleanup_deadline && http_clock_ms() >= cleanup_deadline) {
            cancel_drain_timed_out=1;break;
        }
        double read_deadline=http_clock_ms()+2000.;
        if (cleanup_deadline && cleanup_deadline < read_deadline) read_deadline = cleanup_deadline;
        if (!expired && deadline > 0 && deadline < read_deadline) read_deadline = deadline;
        int received=engine_line(fileno(in),&reader,line,sizeof line,read_deadline);
		if (received<=0) {
			if (received==-2) { if (client_gone(fd)) g.gone = 1; if (stream && !g.gone && !g.cap_hit && !expired && ++quiet >= 5) { quiet = 0; if (chunk(fd, ": keepalive\n\n", 13)) g.gone = 1; } continue; }
			break;
		}
        /* These two ordered acknowledgements precede chat. Unknown-command
           compatibility applies only here, never to a later engine failure. */
        if(negotiating) {
            int which=negotiating--;
            if(which==2 && !strcmp(line,"ok: timings 1\n"))continue;
            if(which==1 && !strcmp(line,"ok: cancel_on_eof 1\n")){cancel_on_eof=1;continue;}
            if(!strcmp(line,"error: gen|chat <n_new> <id>... | stop [<id> [<id>]] | info | cubin <path> | quit\n"))continue;
            failed=1;break;
        }
        if(!strncmp(line,"timings ",8)) {
            size_t ln=strlen(line);const char *te=NULL;
            int terminated=ln>8 && line[ln-1]=='\n';
            /* The engine's line terminator is protocol framing. Preserve the
               JSON bytes without leaking an unprefixed line into SSE data. */
            if(terminated){--ln;if(ln>8 && line[ln-1]=='\r')--ln;}
            jv *v=terminated?json_parse(a,line+8,ln-8,&te):NULL;
            sb_free(&engine_timing);engine_timing=(sbuf){0};
            if(v && v->k==J_DICT)sb_put(&engine_timing,line+8,ln-8);
            continue;
        }
        unsigned id; int e, npm, ran, torun, cached = 0; double r, f;
		if (sscanf(line, "prog %d %d %d", &ran, &torun, &cached) >= 2) {
            double now = http_clock_ms();
            if (progress && stream && !g.gone && !g.cap_hit && !expired && (ran >= torun || now - last_progress_ms >= 500.)) {
                last_progress_ms = now; quiet = 0;
                sbuf b = {0};
                sb_printf(&b, "data: {\"choices\":[],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\",\"prompt_progress\":{\"total\":%d,\"cache\":%d,\"processed\":%d,\"time_ms\":%.1f}}\n\n", g.created, NAME, torun, cached, ran, now - engine_submitted_ms);
                if (chunk(fd, b.s, b.n)) g.gone = 1;
                sb_free(&b);
            }
        }
		else if (sscanf(line, "tok %u", &id) == 1) {
            last_socket_observed_ms = http_clock_ms();
            if (!raw_token_events++) first_socket_observed_ms = last_socket_observed_ms;
            if(!g.gone && !g.cap_hit && !expired)on_token(&g, id);
        }
		else if (sscanf(line, "done %d %d %d %lf %lf %d %d", &ngen, &e, &npm, &r, &f, &resumed, &snapped) >= 5) { ended = e; run_ms = r; first_ms = f; done = 1; break; }
		else if (!strncmp(line, "error", 5)) { fprintf(stderr, "flashd: engine: %s", line); failed = 1; break; }
	}
    if (deadline_expired(deadline)) expired = 1;
	if (!done && !g.gone && !g.cap_hit && !expired) failed = 1;   /* the engine's stream ended without its "done" line: the response is not complete */
    if (!done) ngen = raw_token_events;
    if (ngen < 0) ngen = 0;
	fclose(in); fclose(out);
	queue_leave();
	free(prompt);
    /* CLI timings is exactly this nested engine record. Flat fields
       retain old names and use the canonical values when available. */
    const char *te=NULL;jv *ej=engine_timing.s?json_parse(a,engine_timing.s,engine_timing.n,&te):NULL;
    char fallback[4096];
    flash_timing legacy={.prompt=np,.resumed=resumed,.completed=done?np-1+ngen:0,
        .counts_known=done,.counts_final=done,.ended=ended,.aborted=g.gone||g.cap_hit||expired,
        .first_valid=done&&ngen>0,.first_ms=first_ms,.run_ms=done?run_ms:NAN,.build_ms=NAN};
    if(!ej){flash_timing_json(fallback,sizeof fallback,&legacy);ej=json_parse(a,fallback,strlen(fallback),&te);}
    jv *ea=jv_getz(ej,"aborted"),*es=jv_getz(ej,"stop_token_observed");
    int engine_aborted=ea && ea->k==J_BOOL && ea->b;
    if(engine_timing.s && es && es->k==J_BOOL)ended=es->b;
    jv *eg=jv_getz(ej,"generated_tokens");
    if(engine_timing.s && eg && eg->k==J_INT && eg->i>=0 && eg->i<=INT_MAX)ngen=(int)eg->i;
    sbuf t={0};sb_puts(&t,"{");
    static const char *aliases[]={"prompt_n","cache_n","prompt_ms","prompt_per_second","predicted_n","predicted_ms","predicted_per_second","decode_intervals","decode_ms"};
    for(size_t k=0;k<sizeof aliases/sizeof aliases[0];k++) {
        if(k)sb_puts(&t,",");
        sb_json_str(&t,aliases[k],strlen(aliases[k]));sb_puts(&t,":");
        jv *v=jv_getz(ej,aliases[k]);
        if(v && v->k==J_NUM)sb_printf(&t,"%.17g",v->d);
        else if(v && v->k==J_INT)sb_printf(&t,"%lld",(long long)v->i);
        else sb_puts(&t,"null");
    }
    sb_printf(&t,",\"schema_version\":1,\"engine\":%s,\"http\":{\"request_id\":\"%ld-%llu\",\"clock\":\"CLOCK_MONOTONIC\",\"start_event\":\"request_worker_entry\",\"complete\":%s,\"aborted\":%s,\"engine_timing_available\":%s,\"raw_token_events\":%d,\"request_read_ms\":%.17g,\"prequeue_ms\":%.17g,\"queue_ms\":%.17g,\"tokenization_ms\":%.17g,\"setup_ms\":%.17g,\"engine_submitted_ms\":%.17g,\"completion_record_created_ms\":%.17g",
        engine_timing.s?engine_timing.s:"null",(long)getpid(),request_id,done&&!failed&&!g.gone&&!g.cap_hit&&!engine_aborted&&!expired?"true":"false",
        g.gone||g.cap_hit||engine_aborted||expired?"true":"false",engine_timing.s?"true":"false",raw_token_events,
        chat_start_ms-request_start_ms,queue_start_ms-chat_start_ms,
        qwait,tokenization_ms,engine_submitted_ms-lock_acquired_ms,
        engine_submitted_ms-request_start_ms,http_clock_ms()-request_start_ms);
    http_optional_ms(&t,"first_token_observed_ms",first_socket_observed_ms,request_start_ms);
    http_optional_ms(&t,"last_token_observed_ms",last_socket_observed_ms,request_start_ms);
    http_optional_ms(&t,"first_text_send_ms",g.first_text_send_ms,request_start_ms);
    http_optional_ms(&t,"ttft_ms",g.first_text_send_ms,request_start_ms);
    http_optional_ms(&t,"first_observation_to_text_send_ms",g.first_text_send_ms,first_socket_observed_ms);
    sb_printf(&t,",\"cancel_requested\":%s,\"cancel_drain_timed_out\":%s,\"cancel_drain_ms\":",
        cancel_requested?"true":"false",cancel_drain_timed_out?"true":"false");
    if(cancel_start_ms)sb_printf(&t,"%.17g",http_clock_ms()-cancel_start_ms);else sb_puts(&t,"null");
    sb_puts(&t,",\"socket_decode_tokens_per_second\":");
    if(done && !failed && !g.gone && !g.cap_hit && !engine_aborted && !expired && raw_token_events>1 && last_socket_observed_ms>first_socket_observed_ms)
        sb_printf(&t,"%.17g",1000.*(raw_token_events-1)/(last_socket_observed_ms-first_socket_observed_ms));
    else sb_puts(&t,"null");
    sb_puts(&t,",\"response_write_complete_ms\":null}}");
    /* Disconnected clients cannot receive a final record; log the same object. */
    fprintf(stderr,"flashd_timings: %s\n",t.s);
    int response_write_attempted=0;
    if (expired) {
        fprintf(stderr, "flashd: the request reached FLASHD_REQUEST_MAX_S (%d s); engine cancellation requested\n", REQ_MAX_S);
        response_write_attempted = !client_gone(fd);
        if (response_write_attempted) timed_out_record(fd, stream, stream || hdr, t.s);
        http_write_log(request_id, request_start_ms, response_write_attempted, 1, g.gone);
        sb_free(&engine_timing); sb_free(&t); sb_free(&g.content); sb_free(&g.reasoning); arena_free(a);
        return;
    }
    if (failed) {
        fprintf(stderr,"flashd: the engine failed during the response\n");
        sbuf b={0};
        if(stream) {
            if(!g.gone) {
                response_write_attempted=1;
                sb_printf(&b,"data: {\"error\":{\"message\":\"the engine failed during the response\",\"type\":\"engine_error\"},\"timings\":%s}\n\n",t.s);
                if(chunk(fd,b.s,b.n)||chunk(fd,"data: [DONE]\n\n",14)||wrs(fd,"0\r\n\r\n"))g.gone=1;
            }
        } else {
            response_write_attempted=1;
            sb_printf(&b,"{\"error\":{\"message\":\"the engine failed during the response\",\"code\":502},\"timings\":%s}",t.s);
            if(reply(fd,502,"application/json",b.s,b.n))g.gone=1;
        }
        http_write_log(request_id,request_start_ms,response_write_attempted,1,g.gone);
        sb_free(&b);sb_free(&engine_timing);sb_free(&t);sb_free(&g.content);sb_free(&g.reasoning);arena_free(a);return;
    }
    if(g.gone)fprintf(stderr,"flashd: client left; engine cancellation requested\n");
    char pp_text[48]="n/a",decode_text[48]="n/a";
    jv *pp=jv_getz(ej,"prompt_positions_per_second"),*dec=jv_getz(ej,"decode_tokens_per_second");
    if(pp && (pp->k==J_NUM||pp->k==J_INT))snprintf(pp_text,sizeof pp_text,"%.1f",pp->k==J_NUM?pp->d:(double)pp->i);
    if(dec && (dec->k==J_NUM||dec->k==J_INT))snprintf(decode_text,sizeof decode_text,"%.1f",dec->k==J_NUM?dec->d:(double)dec->i);
	const char *fin = g.gone || g.cap_hit || engine_aborted ? "cancelled" : ended ? "stop" : "length";
	fprintf(stderr, "flashd: %d prompt (%d from the cache, snapshot at %d) + %d generated tokens, %.1f ms (first %.1f), %s, pp %s decode %s tok/s, queued %.1f s\n", np, resumed, snapped, ngen, run_ms, first_ms, fin, pp_text, decode_text, qwait / 1e3);
	if (stream) {
		if (!g.gone) {
			response_write_attempted=1;
			sbuf b = {0};
			sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":%d,\"prompt_tokens\":%d,\"total_tokens\":%d},\"timings\":%s}\n\n", fin, g.created, NAME, ngen, np, ngen + np, t.s);
			if(chunk(fd,b.s,b.n)||chunk(fd,"data: [DONE]\n\n",14)||wrs(fd,"0\r\n\r\n"))g.gone=1;
            sb_free(&b);
		}
	} else {
		response_write_attempted=1;
		sbuf b = {0};
		sb_puts(&b, "{\"choices\":[{\"index\":0,\"finish_reason\":\""); sb_puts(&b, fin); sb_puts(&b, "\",\"message\":{\"role\":\"assistant\",\"content\":");
		sb_json_str(&b, g.content.s ? g.content.s : "", g.content.n);
		if (g.reasoning.n) { sb_puts(&b, ",\"reasoning_content\":"); sb_json_str(&b, g.reasoning.s, g.reasoning.n); }
		sb_printf(&b, "}}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":%d,\"prompt_tokens\":%d,\"total_tokens\":%d},\"timings\":%s}", g.created, NAME, ngen, np, ngen + np, t.s);
		if(reply(fd,200,"application/json",b.s,b.n))g.gone=1;
            sb_free(&b);
	}
    http_write_log(request_id,request_start_ms,response_write_attempted,0,g.gone);
	sb_free(&engine_timing); sb_free(&t); sb_free(&g.content); sb_free(&g.reasoning); arena_free(a);
}

/* ---- HTTP ---- */
static void *serve(void *arg) {
	double request_start_ms = http_clock_ms();
    request_deadline = REQ_MAX_S > 0 ? request_start_ms + REQ_MAX_S * 1e3 : 0;
	int fd = (int)(long)arg; struct timeval tv = { 30, 0 };
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);   /* a client that stops reading must not hold the engine lock: a send that blocks 30 s fails and the response is dropped */
	char *req = malloc(1 << 16); size_t n = 0; char *hend = NULL;
	while (!hend && n < (1 << 16) - 1) {
		ssize_t k = rd(fd, req + n, (1 << 16) - 1 - n);
        if (k <= 0) { if (deadline_expired(request_deadline)) timed_out(fd, 0, 0); free(req); close(fd); return NULL; }
		n += (size_t)k; req[n] = 0; hend = strstr(req, "\r\n\r\n");
	}
	if (!hend) { error(fd, 400, "bad request"); free(req); close(fd); return NULL; }
	size_t hl = (size_t)(hend - req) + 4, clen = 0;
	for (char *p = req; p < hend; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : hend)
		if (!strncasecmp(p, "content-length:", 15)) clen = (size_t)strtoull(p + 15, NULL, 10);
	if (clen > (128u << 20)) { error(fd, 413, "request too large"); free(req); close(fd); return NULL; }
	char *body = malloc(clen + 1); size_t have = n - hl > clen ? clen : n - hl; memcpy(body, req + hl, have);
	while (have < clen) { ssize_t k = rd(fd, body + have, clen - have); if (k <= 0) break; have += (size_t)k; }
    if (deadline_expired(request_deadline)) { timed_out(fd, 0, 0); free(body); free(req); close(fd); return NULL; }
	body[have] = 0;
	char method[16] = "", path[256] = ""; sscanf(req, "%15s %255s", method, path);
	char *q = strchr(path, '?'); if (q) *q = 0;
	if (!strcmp(method, "GET") && !strcmp(path, "/health")) reply(fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
	else if (!strcmp(method, "GET") && !strcmp(path, "/queue")) {
		char q[96]; pthread_mutex_lock(&QM); int k = snprintf(q, sizeof q, "{\"running\":%d,\"waiting\":%d,\"max\":%d}", QBUSY, QLEN, QMAX); pthread_mutex_unlock(&QM);
		reply(fd, 200, "application/json", q, (size_t)k);
	}
	else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
		sbuf b = {0}; sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":"); sb_json_str(&b, NAME, strlen(NAME)); sb_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"flashd\"}]}");
		reply(fd, 200, "application/json", b.s, b.n); sb_free(&b);
	} else if (!strcmp(method, "GET") && !strcmp(path, "/props")) {
		sbuf b = {0}; const char *t = tok_template(V);
		sb_printf(&b, "{\"default_generation_settings\":{\"n_ctx\":%d,\"temperature\":%g,\"top_k\":%d,\"top_p\":%g,\"presence_penalty\":0.0,\"repeat_penalty\":1.0,\"max_reasoning_tokens\":%ld,\"max_answer_tokens\":%ld},\"modalities\":{\"vision\":%s},\"model_alias\":", POSITIONS, DEF_TEMP, DEF_TOPK, DEF_TOPP, REASONING_MAX, ANSWER_MAX, VIS_SOCK ? "true" : "false"); sb_json_str(&b, NAME, strlen(NAME));
		sb_puts(&b, ",\"chat_template\":"); sb_json_str(&b, t ? t : "", t ? strlen(t) : 0); sb_putc(&b, '}');
		reply(fd, 200, "application/json", b.s, b.n); sb_free(&b);
	} else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) chat(fd, body, have, request_start_ms);
	else error(fd, 404, "not found");
	free(body); free(req); close(fd);
	return NULL;
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: flashd <shard-1.gguf> <flash-socket> <host:port>\n"); return 2; }
	signal(SIGPIPE, SIG_IGN);
	SOCK = argv[2];
	if (getenv("FLASHD_NAME")) NAME = getenv("FLASHD_NAME");
	if (getenv("FLASHD_THINKING")) THINK_DEFAULT = atoi(getenv("FLASHD_THINKING"));
	if (getenv("FLASHD_QUEUE_MAX")) QMAX = atoi(getenv("FLASHD_QUEUE_MAX"));
	if (getenv("FLASHD_CONCURRENT")) QCONC = atoi(getenv("FLASHD_CONCURRENT")) > 0 ? atoi(getenv("FLASHD_CONCURRENT")) : 1;
	if (getenv("FLASHD_QUEUE_WAIT_S")) QWAIT_S = atoi(getenv("FLASHD_QUEUE_WAIT_S"));
	VIS_SOCK = getenv("FLASHVIS_SOCK");
	REQ_MAX_S = request_max_seconds(getenv("FLASHD_REQUEST_MAX_S"), VIS_SOCK && *VIS_SOCK);
	V = tok_load(argv[1]); if (!V) return 1;
	if (getenv("FLASHD_IMGDIR")) snprintf(IMGDIR, sizeof IMGDIR, "%s", getenv("FLASHD_IMGDIR"));
	ID_IMGPAD = tok_find(V, "<|image_pad|>");
	{ double x; if (!tok_meta_num(V, "general.sampling.temp", &x)) DEF_TEMP = x; if (!tok_meta_num(V, "general.sampling.top_p", &x)) DEF_TOPP = x; if (!tok_meta_num(V, "general.sampling.top_k", &x)) DEF_TOPK = (int)x;
	  if (getenv("FLASHD_GREEDY")) DEF_TEMP = 0.0; }
	const char *ts = tok_template(V); char err[512];
	if (!ts) { fprintf(stderr, "%s has no chat template\n", argv[1]); return 1; }
	TPL = jinja_compile(ts, strlen(ts), err, sizeof err);
	if (!TPL) { fprintf(stderr, "chat template: %s\n", err); return 1; }
	int e = tok_find(V, "<|im_end|>"), te = tok_find(V, "</think>");
	if (e < 0) { fprintf(stderr, "no <|im_end|> token\n"); return 1; }
	ID_END = (uint32_t)e; ID_THINK_END = te < 0 ? 0xffffffffu : (uint32_t)te;
	FILE *out = NULL, *in = engine_open(&out);
	if (!in) { fprintf(stderr, "cannot reach the engine at %s\n", SOCK); return 1; }
	fprintf(out, "info\n"); fflush(out);
	char line[128]; if (fgets(line, sizeof line, in) && sscanf(line, "info %d %u", &POSITIONS, &IMG_ROWS) >= 1) {} else { fprintf(stderr, "engine did not answer info\n"); return 1; }
	fclose(in); fclose(out);

	char host[128]; int port;
	if (sscanf(argv[3], "%127[^:]:%d", host, &port) != 2) { fprintf(stderr, "listen address must be host:port\n"); return 2; }
	int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) }; inet_pton(AF_INET, host, &sa.sin_addr);
	if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 64)) { perror(argv[3]); return 1; }
	fprintf(stderr, "flashd: %s, %d positions, sampling defaults from the GGUF: temp %g top_k %d top_p %g%s, listening on %s\n", NAME, POSITIONS, DEF_TEMP, DEF_TOPK, DEF_TOPP, DEF_TEMP > 0 ? "" : " (greedy)", argv[3]);
	for (;;) {
		int c = accept(ls, NULL, NULL); if (c < 0) { if (errno != EINTR) perror("accept"); continue; }
		pthread_t th; if (pthread_create(&th, NULL, serve, (void *)(long)c)) { close(c); continue; }
		pthread_detach(th);
	}
}
