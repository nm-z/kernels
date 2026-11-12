package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/nm-z/kernels/codex/mux"
)

func TestHumanCtx(t *testing.T) {
	for n, want := range map[int]string{0: "-", 131072: "128k", 128000: "128k", 400000: "400k", 4096: "4k", 1500: "1500"} {
		if got := humanCtx(n); got != want {
			t.Errorf("humanCtx(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestShownState(t *testing.T) {
	for s, want := range map[string]string{"up": "up", "busy": "busy", "down": "down", "starting": "down", "failed": "down", "done": "down"} {
		if got := shownState(s); got != want {
			t.Errorf("shownState(%q) = %q, want %q", s, got, want)
		}
	}
}

func TestDeviceOn(t *testing.T) {
	if got := deviceOn("archy", "archy 2x Tesla M60 (sm_52)"); got != "2x Tesla M60 (sm_52)" {
		t.Errorf("got %q", got)
	}
	if got := deviceOn("archy", "archyx M60"); got != "archyx M60" {
		t.Errorf("only the whole host name is dropped: got %q", got)
	}
}

// The table lists services only: no row for a machine without any, no LAST RUN column.
func TestPrintServicesRows(t *testing.T) {
	r, w, _ := os.Pipe()
	stdout := os.Stdout
	os.Stdout = w
	printServices([]hostServices{
		{Host: "archy", Data: []mux.ServiceInfo{{Name: "nemotron-3-embed-8b", Kind: "embed", State: "starting", Ctx: 131072, Device: "archy 2x Tesla M60"}}},
		{Host: "benji", Error: "unreachable: no route to host"},
	})
	w.Close()
	os.Stdout = stdout
	b, _ := io.ReadAll(r)
	out := string(b)
	if strings.Contains(out, "benji") || strings.Contains(out, "unreachable") || strings.Contains(out, "LAST RUN") {
		t.Errorf("table:\n%s", out)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 || !strings.Contains(lines[1], "down") || !strings.Contains(lines[1], "128k") ||
		!strings.HasSuffix(lines[1], " 2x Tesla M60") {
		t.Errorf("want the header and one archy row (down, 128k, device without the host):\n%s", out)
	}
}
