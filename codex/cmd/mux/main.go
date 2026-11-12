// mux serves AI on one host: it starts and stops every service and serves the Responses API.
//
//	mux serve [-f mux.yaml]      the daemon
//	mux ps [-local] [-json] | up NAME | down NAME | events [-since TS] [-f mux.yaml]
//
// The CLI finds the daemon through MUX_URL, else the `listen` of the compose file. `mux ps` lists the services of
// every machine the host file's `fleet:` names (this one through its daemon, the others through ssh), one table
// with a HOST column; -local lists this machine's alone.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/nm-z/kernels/codex/mux"
)

func usage() {
	fmt.Fprint(os.Stderr, `usage: mux serve [-f FILE]
       mux ps [-local] [-json] [-f FILE]
       mux events [-since TS] [-f FILE]
       mux up NAME | down NAME [-f FILE]

FILE defaults to ~/.config/mux/mux.yaml; the CLI uses MUX_URL (http://host:port) when set.
ps lists every machine in FILE's fleet: (ssh names), HOST first; -local lists this machine alone.
`)
	os.Exit(2)
}

func main() {
	args := os.Args[1:]
	file, since := "", ""
	local, asJSON := false, false
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
		case a == "-local" || a == "--local":
			local = true
		case a == "-json" || a == "--json":
			asJSON = true
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
		file = home + "/.config/mux/mux.yaml"
	}
	switch rest[0] {
	case "serve":
		serve(file)
	case "ps":
		ps(file, local, asJSON)
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
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	srv, err := mux.New(mux.Options{Config: cfg})
	if err != nil {
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	if err := srv.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	s := <-sig
	fmt.Fprintf(os.Stderr, "mux: %s: stopping every service\n", s)
	srv.Shutdown()
}

// ---- the CLI --------------------------------------------------------------------------------------------

func baseURL(file string) string {
	if u := os.Getenv("MUX_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	listen, err := mux.ReadListen(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mux: cannot find the daemon:", err, "(set MUX_URL or -f)")
		os.Exit(1)
	}
	host, port, err := net.SplitHostPort(listen)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mux: listen:", err)
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
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		fmt.Fprintf(os.Stderr, "mux: %s %s: %s\n", method, url, strings.TrimSpace(string(b)))
		os.Exit(1)
	}
	return b
}

// hostServices is one machine's answer to ps: its services, or why there are none.
type hostServices struct {
	Host  string            `json:"host"`
	Data  []mux.ServiceInfo `json:"data"`
	Error string            `json:"error,omitempty"` // "unreachable: …", "no mux: …"
}

// ps lists the services of every machine in the fleet (or this one alone with -local), asked at once.
func ps(file string, local, asJSON bool) {
	self := shortHost()
	if local {
		h := localServices(file, self)
		if asJSON {
			b, _ := json.Marshal(h)
			fmt.Println(string(b))
			return
		}
		if h.Error != "" {
			fmt.Fprintln(os.Stderr, "mux:", h.Error)
			os.Exit(1)
		}
		printServices([]hostServices{h})
		return
	}
	fleet, err := mux.ReadFleet(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	if !contains(fleet, self) {
		fleet = append([]string{self}, fleet...)
	}
	all := make([]hostServices, len(fleet))
	var wg sync.WaitGroup
	for i, h := range fleet {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if h == self {
				all[i] = localServices(file, self)
			} else {
				all[i] = remoteServices(h)
			}
		}()
	}
	wg.Wait()
	if asJSON {
		b, _ := json.Marshal(all)
		fmt.Println(string(b))
		return
	}
	printServices(all)
}

func shortHost() string {
	h, err := os.Hostname()
	if err != nil || h == "" {
		return "localhost"
	}
	return strings.SplitN(h, ".", 2)[0]
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// localServices asks this machine's daemon (MUX_URL, else the host file's listen).
func localServices(file, self string) hostServices {
	out := hostServices{Host: self}
	base := os.Getenv("MUX_URL")
	if base == "" {
		listen, err := mux.ReadListen(file)
		if err != nil {
			out.Error = "no mux: " + err.Error()
			return out
		}
		host, port, err := net.SplitHostPort(listen)
		if err != nil {
			out.Error = "no mux: listen: " + err.Error()
			return out
		}
		if host == "" || host == "0.0.0.0" || host == "::" {
			host = "127.0.0.1"
		}
		base = "http://" + net.JoinHostPort(host, port)
	}
	cl := &http.Client{Timeout: 10 * time.Second}
	resp, err := cl.Get(strings.TrimRight(base, "/") + "/v1/services")
	if err != nil {
		out.Error = "no mux: " + err.Error()
		return out
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		out.Error = fmt.Sprintf("no mux: GET /v1/services: %s", strings.TrimSpace(string(b)))
		return out
	}
	var v struct {
		Data []mux.ServiceInfo `json:"data"`
	}
	if err := json.Unmarshal(b, &v); err != nil {
		out.Error = "no mux: " + err.Error()
		return out
	}
	out.Data = v.Data
	return out
}

// remoteCommand runs that machine's mux for its own services (its ~/.local/bin/mux, else the one on its PATH).
const remoteCommand = `m="$HOME/.local/bin/mux"; [ -x "$m" ] || m=mux; exec "$m" ps -local -json`

// probeTimeout bounds the first look at a machine: a LAN answers in under a millisecond, while a machine that is
// off answers "no route to host" only when the kernel's ARP gives up, about 3 s later.
const probeTimeout = 250 * time.Millisecond

// sshTarget is where ssh would connect for an ssh name (its config's hostname and port, from `ssh -G`).
func sshTarget(host string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	b, err := exec.CommandContext(ctx, "ssh", "-G", host).Output()
	if err != nil {
		return "", err
	}
	name, port := host, "22"
	for _, l := range strings.Split(string(b), "\n") {
		if k, v, ok := strings.Cut(l, " "); ok {
			switch k {
			case "hostname":
				name = v
			case "port":
				port = v
			}
		}
	}
	return net.JoinHostPort(name, port), nil
}

// remoteServices asks another machine over ssh, once its ssh port answers within probeTimeout; ssh's own failure
// (exit 255) is "unreachable" too.
func remoteServices(host string) hostServices {
	out := hostServices{Host: host}
	if addr, err := sshTarget(host); err == nil {
		c, err := net.DialTimeout("tcp", addr, probeTimeout)
		if err != nil {
			out.Error = "unreachable: " + err.Error()
			return out
		}
		c.Close()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host, remoteCommand)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	b, err := cmd.Output()
	reason := lastLine(stderr.String())
	if err != nil {
		var ee *exec.ExitError
		switch {
		case ctx.Err() != nil:
			out.Error = "unreachable: no answer in 15 s"
		case errors.As(err, &ee) && ee.ExitCode() == 255:
			out.Error = "unreachable: " + reason
		case errors.As(err, &ee) && ee.ExitCode() == 127:
			out.Error = "no mux: not installed"
		default:
			out.Error = "no mux: " + firstNonEmpty(reason, err.Error())
		}
		return out
	}
	var v hostServices
	if err := json.Unmarshal(b, &v); err != nil {
		out.Error = "no mux: its mux predates fleet ps (update it): " + firstNonEmpty(lastLine(string(b)), err.Error())
		return out
	}
	out.Data, out.Error = v.Data, v.Error
	return out
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// printServices is the table, HOST first; a machine without services is a row saying why, its reason below.
func printServices(all []hostServices) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "HOST\tNAME\tKIND\tSTATE\tPID\tCTX\tSLOTS\tFOR\tLAST RUN / PROGRESS\tDEVICE")
	for _, h := range all {
		if h.Error != "" {
			state, _, _ := strings.Cut(h.Error, ":")
			fmt.Fprintf(tw, "%s\t-\t-\t%s\t-\t-\t-\t-\t-\t-\n", h.Host, state)
			continue
		}
		for _, e := range h.Data {
			since := "-"
			if t, err := time.Parse(time.RFC3339, e.Since); err == nil {
				since = age(time.Since(t))
			}
			ctx := "-"
			if e.Ctx > 0 {
				ctx = fmt.Sprint(e.Ctx)
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\t%s\t%s\n", h.Host, e.Name, e.Kind, e.State, e.PID, ctx, e.Slots, since, lastRun(e), e.Device)
		}
	}
	tw.Flush()
	for _, h := range all {
		if h.Error != "" {
			fmt.Printf("%s: %s\n", h.Host, h.Error)
		}
		for _, e := range h.Data {
			if e.Reason != "" {
				fmt.Printf("%s/%s: %s: %s\n", h.Host, e.Name, e.State, e.Reason)
			}
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
		fmt.Fprintln(os.Stderr, "mux:", err)
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
		fmt.Fprintln(os.Stderr, "mux:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "mux: %s\n", strings.TrimSpace(string(b)))
		os.Exit(1)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		fmt.Println(sc.Text())
	}
}
