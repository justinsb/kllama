package llamacpp

import (
	"context"
	"fmt"
	"strings"

	"github.com/justinsb/kllama/third_party/llamacpp"
)

type Tokenizer struct {
	model *llamacpp.LlamaModel
	vocab *llamacpp.LlamaVocab
}

func NewTokenizer(modelPath string) (*Tokenizer, error) {
	modelParams := llamacpp.NewLlamaModelParams()
	model, err := llamacpp.NewLlamaModel(modelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// template, ok := t.model.GetChatTemplate()
	// if !ok {
	// 	return nil, fmt.Errorf("failed to get chat template")
	// }
	// fmt.Println(template)

	vocab, err := model.GetVocab()
	if err != nil {
		model.Free()
		return nil, fmt.Errorf("failed to get vocab: %w", err)
	}

	return &Tokenizer{model: model, vocab: vocab}, nil
}

func (t *Tokenizer) Free() {
	t.model.Free()
	t.vocab.Free()
}

func (t *Tokenizer) Tokenize(ctx context.Context, prompt string) ([]llamacpp.Token, error) {

	tokens, err := t.vocab.TokenizePrompt(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize prompt: %w", err)
	}

	return tokens, nil
}

func (t *Tokenizer) Detokenize(ctx context.Context, tokens []llamacpp.Token) (string, error) {
	var sb strings.Builder

	for _, token := range tokens {
		piece, err := t.vocab.ConvertTokenToString(token)
		if err != nil {
			return "", fmt.Errorf("failed to convert token to string: %w", err)
		}
		sb.WriteString(piece)
	}

	return sb.String(), nil
}
