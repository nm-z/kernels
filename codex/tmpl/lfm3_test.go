package tmpl

import (
	"os"
	"slices"
	"testing"
)

// TestLFM3PreTokenizer pins the ids of the lfm3 pre-tokenizer (letters only, digit runs of up to 3, merges ignored for
// a word that is a token) to what the flashd tokenizer, itself checked against llama.cpp's reference, gives on
// the same GGUF (flash-sass/kern/tok_test). It skips without the LFM3 GGUF (MUX_TEST_LFM3_GGUF or
// ~/models/lfm3-1.7b-nate/tokenizer.gguf).
func TestLFM3PreTokenizer(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := os.Getenv("MUX_TEST_LFM3_GGUF")
	if path == "" {
		path = home + "/models/lfm3-1.7b-nate/tokenizer.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skip("no LFM3 GGUF found (set MUX_TEST_LFM3_GGUF)")
	}
	tk, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	// (tok_test prints the BOS first, 124894; Encode leaves it to the caller)
	for _, c := range []struct {
		text string
		ids  []uint32
	}{
		{"Hello world", []uint32{35808, 1530}},
		{"12345 3.14159 x2 v1.2.3", []uint32{9792, 2136, 229, 27, 22, 13387, 5098, 2552, 26, 346, 25, 22, 26, 22, 27}},
		{"Café naïve — “quotes” 日本語のテキスト", []uint32{43, 1804, 359, 1615, 6337, 623, 1679, 870, 361, 8841, 737, 37054, 12673, 583, 3873, 89301}},
		{"I'm can't we'll 'quoted'", []uint32{49, 5716, 510, 1400, 522, 5515, 951, 361, 9691, 15}},
	} {
		got, err := tk.Encode(c.text)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) > 0 && got[0] == 124894 {
			got = got[1:]
		}
		if !slices.Equal(got, c.ids) {
			t.Errorf("%q: got %v, want %v", c.text, got, c.ids)
		}
	}
}
