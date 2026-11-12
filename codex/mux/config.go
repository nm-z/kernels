package mux

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Service kinds (PROTOCOL.md).
const (
	KindEngine = "engine"
	KindShim   = "shim"
	KindEmbed  = "embed"
	KindJob    = "job"
)

// ServiceConfig is one service: its service.yaml, plus its desired state from the host file.
type ServiceConfig struct {
	Name     string            `yaml:"-"`
	Dir      string            `yaml:"-"` // the service folder (services_dir/NAME)
	Up       bool              `yaml:"-"` // desired state, from the host file
	Kind     string            `yaml:"kind"`
	Display  string            `yaml:"display"` // the name web chat shows; the llama.cpp face lists NAME.gguf
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

// Config is a host's mux.yaml with the services it names loaded from services_dir.
type Config struct {
	Listen      string
	Socket      string
	ServicesDir string
	Services    []*ServiceConfig // in host-file order
	// CompatListen is a second HTTP address that serves only the llama.cpp face (/v1/models as llama-server
	// lists it, /props, /v1/chat/completions) for web chat; CompatService is the service it fronts, and the
	// model a chat completion without one runs on.
	CompatListen  string
	CompatService string
	Path          string // where the host file was read from ("" for a built one)
}

var defaultEfforts = []string{"none", "low", "medium", "high", "xhigh"}

// shimEfforts are the efforts a shim takes when its file names none: every effort a client sends
// (Codex sends minimal and max). A shim runs its provider's strongest options whatever the effort.
var shimEfforts = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// hostFile is mux.yaml as written.
type hostFile struct {
	Listen      string    `yaml:"listen"`
	Socket      string    `yaml:"socket"`
	ServicesDir string    `yaml:"services_dir"`
	Services    yaml.Node `yaml:"services"`

	CompatListen  string `yaml:"compat_listen"`
	CompatService string `yaml:"compat_service"`
}

// HostService is one entry of the host file's services.
type HostService struct {
	Name string
	Up   bool
}

// Host is the host file alone (what the CLI and the tests of the file need).
type Host struct {
	Listen        string
	Socket        string
	ServicesDir   string
	Services      []HostService
	CompatListen  string
	CompatService string
}

// ParseHost reads the text of a host file. Relative services_dir is resolved against base (the
// directory of the file).
func ParseHost(b []byte, base string) (*Host, error) {
	var raw hostFile
	if err := yaml.NewDecoder(bytes.NewReader(b)).Decode(&raw); err != nil {
		return nil, err
	}
	home, _ := os.UserHomeDir()
	h := &Host{Listen: raw.Listen, Socket: expandHome(raw.Socket, home), ServicesDir: expandHome(raw.ServicesDir, home),
		CompatListen: raw.CompatListen, CompatService: raw.CompatService}
	if h.Listen == "" {
		h.Listen = "127.0.0.1:18100"
	}
	if h.Socket == "" {
		d := os.Getenv("XDG_RUNTIME_DIR")
		if d == "" {
			return nil, fmt.Errorf("socket: not set and there is no XDG_RUNTIME_DIR")
		}
		h.Socket = filepath.Join(d, "mux.sock")
	}
	if h.ServicesDir == "" {
		return nil, fmt.Errorf("services_dir: required (the folder holding NAME/service.yaml)")
	}
	if !filepath.IsAbs(h.ServicesDir) && base != "" {
		h.ServicesDir = filepath.Join(base, h.ServicesDir)
	}
	if raw.Services.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("services: want a mapping of name to {up: bool}")
	}
	seen := map[string]bool{}
	for i := 0; i+1 < len(raw.Services.Content); i += 2 {
		name := raw.Services.Content[i].Value
		if name == "" || strings.ContainsAny(name, "/ \t\n") || name == "." || name == ".." {
			return nil, fmt.Errorf("services.%s: the name must not contain slashes or spaces", name)
		}
		if seen[name] {
			return nil, fmt.Errorf("services.%s: defined twice", name)
		}
		seen[name] = true
		var v struct {
			Up bool `yaml:"up"`
		}
		if n := raw.Services.Content[i+1]; !(n.Kind == yaml.ScalarNode && n.Tag == "!!null") {
			if err := n.Decode(&v); err != nil {
				return nil, fmt.Errorf("services.%s: %w", name, err)
			}
		}
		h.Services = append(h.Services, HostService{Name: name, Up: v.Up})
	}
	return h, nil
}

// LoadConfig reads the host file and every service it names.
func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	c, err := loadConfig(b, filepath.Dir(abs))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	c.Path = path
	return c, nil
}

// ParseConfig is LoadConfig for host-file text; relative services_dir is relative to the current directory.
func ParseConfig(b []byte) (*Config, error) { return loadConfig(b, "") }

func loadConfig(b []byte, base string) (*Config, error) {
	h, err := ParseHost(b, base)
	if err != nil {
		return nil, err
	}
	c := &Config{Listen: h.Listen, Socket: h.Socket, ServicesDir: h.ServicesDir, CompatListen: h.CompatListen, CompatService: h.CompatService}
	for _, hs := range h.Services {
		sc, err := LoadService(h.ServicesDir, hs.Name)
		if err != nil {
			return nil, err
		}
		sc.Up = hs.Up
		c.Services = append(c.Services, sc)
	}
	if c.CompatListen != "" && c.CompatService == "" {
		return nil, fmt.Errorf("compat_listen: needs compat_service (the service it fronts)")
	}
	if c.CompatService != "" {
		var sc *ServiceConfig
		for _, s := range c.Services {
			if s.Name == c.CompatService {
				sc = s
			}
		}
		if sc == nil {
			return nil, fmt.Errorf("compat_service: %q is not one of services", c.CompatService)
		}
		if sc.Kind != KindEngine && sc.Kind != KindShim {
			return nil, fmt.Errorf("compat_service: %q is a %s; chat needs an engine or a shim", c.CompatService, sc.Kind)
		}
	}
	return c, nil
}

// LoadService reads services_dir/NAME/service.yaml.
func LoadService(servicesDir, name string) (*ServiceConfig, error) {
	dir := filepath.Join(servicesDir, name)
	file := filepath.Join(dir, "service.yaml")
	b, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("service %s: %w", name, err)
	}
	s, err := ParseService(name, dir, b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", file, err)
	}
	return s, nil
}

// ParseService reads and validates the text of a service.yaml.
func ParseService(name, dir string, b []byte) (*ServiceConfig, error) {
	s := &ServiceConfig{Name: name, Dir: dir}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	if err := dec.Decode(s); err != nil {
		return nil, err
	}
	home, _ := os.UserHomeDir()
	s.GGUF = expandHome(s.GGUF, home)
	s.Cwd = expandHome(s.Cwd, home)
	switch s.Kind {
	case KindEngine, KindShim, KindEmbed, KindJob:
	case "":
		return nil, fmt.Errorf("kind: required (engine, shim, embed or job)")
	default:
		return nil, fmt.Errorf("kind: %q is none of engine, shim, embed, job", s.Kind)
	}
	if len(s.Command) == 0 {
		return nil, fmt.Errorf("command: required")
	}
	if s.Kind == KindEngine && s.GGUF == "" {
		return nil, fmt.Errorf("gguf: required for an engine (tokenizer and chat template)")
	}
	if (s.Kind == KindEngine || s.Kind == KindShim) && s.Ctx < 1 {
		return nil, fmt.Errorf("ctx: required for an engine or shim (the context window broadcast to clients)")
	}
	if s.Kind == KindEngine && len(s.Efforts) == 0 {
		s.Efforts = append([]string(nil), defaultEfforts...)
	}
	if s.Kind == KindShim && len(s.Efforts) == 0 {
		s.Efforts = append([]string(nil), shimEfforts...)
	}
	switch s.Restart {
	case "":
		s.Restart = "on-failure"
		if s.Kind == KindJob {
			s.Restart = "no"
		}
	case "on-failure", "no":
	default:
		return nil, fmt.Errorf("restart: %q is neither on-failure nor no", s.Restart)
	}
	if s.Cwd == "" {
		s.Cwd = dir
	} else if !filepath.IsAbs(s.Cwd) && dir != "" {
		s.Cwd = filepath.Join(dir, s.Cwd)
	}
	return s, nil
}

// SetUpInFile rewrites `up:` of one service in the host file, keeping the rest of it (comments, order)
// as yaml.v3 allows. The write is atomic.
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
	svcs := mapGet(doc.Content[0], "services")
	var idx = -1
	if svcs != nil && svcs.Kind == yaml.MappingNode {
		for i := 0; i+1 < len(svcs.Content); i += 2 {
			if svcs.Content[i].Value == name {
				idx = i + 1
			}
		}
	}
	if idx < 0 {
		return fmt.Errorf("%s: no service %q", path, name)
	}
	val := "false"
	if up {
		val = "true"
	}
	entry := svcs.Content[idx]
	if entry.Kind != yaml.MappingNode { // `NAME:` with nothing after it
		*entry = yaml.Node{Kind: yaml.MappingNode, Tag: "!!map", Style: yaml.FlowStyle}
	}
	if n := mapGet(entry, "up"); n != nil {
		n.Kind, n.Tag, n.Value, n.Style = yaml.ScalarNode, "!!bool", val, 0
	} else {
		k := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "up"}
		v := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: val}
		entry.Content = append([]*yaml.Node{k, v}, entry.Content...)
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

// ReadListen reads only `listen:` of a host file (what the CLI needs to find the daemon).
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

// ReadFleet reads only `fleet:` of a host file: the ssh names of every machine running a mux (this one among them),
// which `mux ps` lists together. None means this machine alone.
func ReadFleet(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Fleet []string `yaml:"fleet"`
	}
	if err := yaml.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	seen := map[string]bool{}
	var out []string
	for _, h := range raw.Fleet {
		if h = strings.TrimSpace(h); h == "" || strings.HasPrefix(h, "-") || strings.ContainsAny(h, " \t\n") {
			return nil, fmt.Errorf("%s: fleet: %q is not a host name", path, h)
		}
		if !seen[h] {
			seen[h] = true
			out = append(out, h)
		}
	}
	return out, nil
}

// Label is the name web chat shows for the service: display, else the service name.
func (c *ServiceConfig) Label() string {
	if c.Display != "" {
		return c.Display
	}
	return c.Name
}

// LlamaID is the model id the llama.cpp face lists: llama-server names models after their GGUF file.
func (c *ServiceConfig) LlamaID() string { return strings.TrimSuffix(c.Label(), ".gguf") + ".gguf" }
