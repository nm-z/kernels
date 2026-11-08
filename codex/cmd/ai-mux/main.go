// ai-mux serves AI on one host: it starts and stops every model endpoint and serves the Responses API.
//
//	ai-mux serve [-f ai-mux.yaml]      the daemon
//	ai-mux ps | up NAME | down NAME | events [-since TS] [-f ai-mux.yaml]
//
// The CLI finds the daemon through AI_MUX_URL, else the `listen` of the compose file.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/nm-z/kernels/codex/mux"
)

func usage() {
	fmt.Fprint(os.Stderr, `usage: ai-mux serve [-f FILE]
       ai-mux ps [-f FILE]
       ai-mux events [-since TS] [-f FILE]
       ai-mux up NAME | down NAME [-f FILE]

FILE defaults to ~/.config/ai-mux/ai-mux.yaml; the CLI uses AI_MUX_URL (http://host:port) when set.
`)
	os.Exit(2)
}

func main() {
	args := os.Args[1:]
	file, since := "", ""
	var rest []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-f" || a == "--f" || a == "-file" || a == "--file":
			if i+1 >= len(args) {
				usage()
			}
			file = args[i+1]
			i++
		case strings.HasPrefix(a, "-f=") || strings.HasPrefix(a, "--f="):
			file = a[strings.Index(a, "=")+1:]
		case a == "-since" || a == "--since":
			if i+1 >= len(args) {
				usage()
			}
			since = args[i+1]
			i++
		case strings.HasPrefix(a, "-since=") || strings.HasPrefix(a, "--since="):
			since = a[strings.Index(a, "=")+1:]
		case a == "-h" || a == "--help" || a == "help":
			usage()
		default:
			rest = append(rest, a)
		}
	}
	if len(rest) == 0 {
		usage()
	}
	if file == "" {
		home, _ := os.UserHomeDir()
		file = home + "/.config/ai-mux/ai-mux.yaml"
	}
	switch rest[0] {
	case "serve":
		serve(file)
	case "ps":
		ps(baseURL(file))
	case "up", "down":
		if len(rest) != 2 {
			usage()
		}
		toggle(baseURL(file), rest[1], rest[0])
	case "events":
		events(baseURL(file), since)
	default:
		usage()
	}
}

func serve(file string) {
	cfg, err := mux.LoadConfig(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	srv, err := mux.New(mux.Options{Config: cfg})
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	if err := srv.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	s := <-sig
	fmt.Fprintf(os.Stderr, "ai-mux: %s: stopping every service\n", s)
	srv.Shutdown()
}

// ---- the CLI --------------------------------------------------------------------------------------------

func baseURL(file string) string {
	if u := os.Getenv("AI_MUX_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	listen, err := mux.ReadListen(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux: cannot find the daemon:", err, "(set AI_MUX_URL or -f)")
		os.Exit(1)
	}
	host, port, err := net.SplitHostPort(listen)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux: listen:", err)
		os.Exit(1)
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port)
}

func call(method, url string) []byte {
	req, _ := http.NewRequest(method, url, nil)
	cl := &http.Client{Timeout: 2 * time.Minute}
	resp, err := cl.Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		fmt.Fprintf(os.Stderr, "ai-mux: %s %s: %s\n", method, url, strings.TrimSpace(string(b)))
		os.Exit(1)
	}
	return b
}

func ps(base string) {
	var out struct {
		Data []mux.ServiceInfo `json:"data"`
	}
	if err := json.Unmarshal(call("GET", base+"/v1/services"), &out); err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tKIND\tSTATE\tPID\tCTX\tSLOTS\tFOR\tLAST RUN / PROGRESS\tDEVICE")
	for _, e := range out.Data {
		since := "-"
		if t, err := time.Parse(time.RFC3339, e.Since); err == nil {
			since = age(time.Since(t))
		}
		ctx := "-"
		if e.Ctx > 0 {
			ctx = fmt.Sprint(e.Ctx)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%d\t%s\t%s\t%s\n", e.Name, e.Kind, e.State, e.PID, ctx, e.Slots, since, lastRun(e), e.Device)
	}
	tw.Flush()
	for _, e := range out.Data {
		if e.Reason != "" {
			fmt.Printf("%s: %s: %s\n", e.Name, e.State, e.Reason)
		}
	}
}

// lastRun is the one-cell summary of what a service last did, by kind.
func lastRun(e mux.ServiceInfo) string {
	if p := e.Progress; p != nil && e.Kind == mux.KindJob {
		s := fmt.Sprintf("%.0f%%", p.Pct)
		if p.Title != "" {
			s += " " + p.Title
		}
		if p.Detail != "" {
			s += " (" + p.Detail + ")"
		}
		return s
	}
	l := e.Last
	if l == nil {
		return "-"
	}
	f := func(label string, v *float64) string {
		if v == nil {
			return ""
		}
		return fmt.Sprintf("%s %.1f ", label, *v)
	}
	switch e.Kind {
	case mux.KindEngine:
		return strings.TrimSpace(f("prefill", l.PrefillTPS) + "tok/s " + f("decode", l.DecodeTPS) + "tok/s " + f("idle", l.IdlePct) + "%")
	case mux.KindShim:
		return strings.TrimSpace(f("ttft", l.TTFTMS) + "ms " + strings.Join(l.Route, ","))
	case mux.KindEmbed:
		return fmt.Sprintf("%d texts", l.Texts)
	}
	return "-"
}

func age(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
	return fmt.Sprintf("%dd", int(d.Hours()/24))
}

func toggle(base, name, action string) {
	var e mux.ServiceInfo
	if err := json.Unmarshal(call("POST", base+"/v1/services/"+name+"/"+action), &e); err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	fmt.Printf("%s: %s (desired %s)\n", e.Name, e.State, action)
}

func events(base, since string) {
	q := ""
	if since != "" {
		q = "?since=" + url.QueryEscape(since)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"/v1/events"+q, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-mux:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "ai-mux: %s\n", strings.TrimSpace(string(b)))
		os.Exit(1)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		fmt.Println(sc.Text())
	}
}
