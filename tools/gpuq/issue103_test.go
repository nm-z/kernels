package gpuq

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRerunReportsDatabaseSetupFailure(t *testing.T) {
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate issue103_test.go")
	}
	toolDir := filepath.Dir(source)
	database := filepath.Join(t.TempDir(), "history-directory")
	if err := os.Mkdir(database, 0700); err != nil {
		t.Fatal(err)
	}

	command := exec.Command("ruby", filepath.Join(toolDir, "gpuq.rb"), "--db", database, "rerun", "unused")
	command.Dir = toolDir
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	exitError, ok := err.(*exec.ExitError)
	if !ok || exitError.ExitCode() != 2 {
		t.Fatalf("rerun exit = %v, want exit status 2", err)
	}
	if stdout.Len() != 0 {
		t.Fatalf("rerun wrote unexpected stdout: %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "gpuq rerun failed:") {
		t.Fatalf("rerun did not report the setup failure on stderr: %q", stderr.String())
	}
}
