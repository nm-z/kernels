package mux

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// EndpointConfig is one entry of ai-mux.yaml's endpoints.
type EndpointConfig struct {
	Name     string            `yaml:"-"`
	Up       bool              `yaml:"up"`
	GGUF     string            `yaml:"gguf"`
	Ctx      int               `yaml:"ctx"`
	Efforts  []string          `yaml:"efforts"`
	Vision   bool              `yaml:"vision"`
	Device   string            `yaml:"device"`
	Command  []string          `yaml:"command"`
	Cwd      string            `yaml:"cwd"`
	Env      map[string]string `yaml:"env"`
	PostStop []string          `yaml:"post_stop"`
	Restart  string            `yaml:"restart"`
}

// Config is ai-mux.yaml.
type Config struct {
	Listen    string
	Socket    string
	Endpoints []*EndpointConfig // in file order
	Path      string            // where it was read from ("" for a built one)
}

var defaultEfforts = []string{"none", "low", "medium", "high", "xhigh"}

// LoadConfig reads and validates a compose file.
func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c, err := ParseConfig(b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	c.Path = path
	return c, nil
}

// ParseConfig parses and validates a compose file's text.
func ParseConfig(b []byte) (*Config, error) {
	var raw struct {
		Listen    string    `yaml:"listen"`
		Socket    string    `yaml:"socket"`
		Endpoints yaml.Node `yaml:"endpoints"`
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	if err := dec.Decode(&raw); err != nil {
		return nil, err
	}
	home, _ := os.UserHomeDir()
	c := &Config{Listen: raw.Listen, Socket: expandHome(raw.Socket, home)}
	if c.Listen == "" {
		c.Listen = "127.0.0.1:18100"
	}
	if c.Socket == "" {
		if d := os.Getenv("XDG_RUNTIME_DIR"); d != "" {
			c.Socket = filepath.Join(d, "ai-mux.sock")
		} else {
			return nil, fmt.Errorf("socket: not set and there is no XDG_RUNTIME_DIR")
		}
	}
	if raw.Endpoints.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("endpoints: want a mapping of name to endpoint")
	}
	seen := map[string]bool{}
	for i := 0; i+1 < len(raw.Endpoints.Content); i += 2 {
		name := raw.Endpoints.Content[i].Value
		e := &EndpointConfig{Name: name}
		if err := raw.Endpoints.Content[i+1].Decode(e); err != nil {
			return nil, fmt.Errorf("endpoints.%s: %w", name, err)
		}
		if seen[name] {
			return nil, fmt.Errorf("endpoints.%s: defined twice", name)
		}
		seen[name] = true
		if strings.ContainsAny(name, "/ \t\n") || name == "" {
			return nil, fmt.Errorf("endpoints.%s: the name must not contain slashes or spaces", name)
		}
		e.GGUF = expandHome(e.GGUF, home)
		e.Cwd = expandHome(e.Cwd, home)
		if e.GGUF == "" {
			return nil, fmt.Errorf("endpoints.%s.gguf: required (tokenizer and chat template)", name)
		}
		if e.Ctx < 1 {
			return nil, fmt.Errorf("endpoints.%s.ctx: required", name)
		}
		if len(e.Command) == 0 {
			return nil, fmt.Errorf("endpoints.%s.command: required", name)
		}
		if len(e.Efforts) == 0 {
			e.Efforts = append([]string(nil), defaultEfforts...)
		}
		switch e.Restart {
		case "":
			e.Restart = "on-failure"
		case "on-failure", "no":
		default:
			return nil, fmt.Errorf("endpoints.%s.restart: %q is neither on-failure nor no", name, e.Restart)
		}
		c.Endpoints = append(c.Endpoints, e)
	}
	return c, nil
}

// SetUpInFile rewrites `up:` of one endpoint in the compose file, keeping the rest of it (comments,
// order) as yaml.v3 allows. The write is atomic.
func SetUpInFile(path, name string, up bool) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(b, &doc); err != nil {
		return err
	}
	if doc.Kind != yaml.DocumentNode || len(doc.Content) != 1 || doc.Content[0].Kind != yaml.MappingNode {
		return fmt.Errorf("%s: not a mapping", path)
	}
	ep := mapGet(mapGet(doc.Content[0], "endpoints"), name)
	if ep == nil || ep.Kind != yaml.MappingNode {
		return fmt.Errorf("%s: no endpoint %q", path, name)
	}
	val := "false"
	if up {
		val = "true"
	}
	if n := mapGet(ep, "up"); n != nil {
		n.Kind, n.Tag, n.Value, n.Style = yaml.ScalarNode, "!!bool", val, 0
	} else {
		k := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "up"}
		v := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: val}
		ep.Content = append([]*yaml.Node{k, v}, ep.Content...)
	}
	var out bytes.Buffer
	enc := yaml.NewEncoder(&out)
	enc.SetIndent(2)
	if err := enc.Encode(&doc); err != nil {
		return err
	}
	enc.Close()
	tmp := path + ".tmp"
	mode := os.FileMode(0o644)
	if st, err := os.Stat(path); err == nil {
		mode = st.Mode().Perm()
	}
	if err := os.WriteFile(tmp, out.Bytes(), mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func mapGet(m *yaml.Node, key string) *yaml.Node {
	if m == nil || m.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

// ReadListen reads only `listen:` of a compose file (what the CLI needs to find the daemon).
func ReadListen(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var raw struct {
		Listen string `yaml:"listen"`
	}
	if err := yaml.Unmarshal(b, &raw); err != nil {
		return "", fmt.Errorf("%s: %w", path, err)
	}
	if raw.Listen == "" {
		raw.Listen = "127.0.0.1:18100"
	}
	return raw.Listen, nil
}
