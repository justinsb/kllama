package llamacpp

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestTokenizer(t *testing.T) {
	ctx := context.Background()

	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		t.Fatalf("MODEL_PATH is not set")
	}

	tokenizer, err := NewTokenizer(modelPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}

	promptString := `<|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 1 Jan 2025
<|eot_id|>
What is today?
`

	tokens, err := tokenizer.Tokenize(ctx, promptString)
	if err != nil {
		t.Fatalf("failed to tokenize: %v", err)
	}
	t.Logf("tokens are: %v", tokens)

	detokenized, err := tokenizer.Detokenize(ctx, tokens)
	if err != nil {
		t.Fatalf("failed to detokenize: %v", err)
	}
	t.Logf("detokenized is: %v", detokenized)

	detokenized = strings.TrimPrefix(detokenized, "<|begin_of_text|>")
	if detokenized != promptString {
		t.Fatalf("detokenized %q is not equal to prompt string %q", detokenized, promptString)
	}
}
