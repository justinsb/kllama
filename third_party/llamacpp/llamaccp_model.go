package llamacpp

// #cgo CFLAGS: -O3 -DNDEBUG -I llama.cpp/include -I llama.cpp/ggml/include
// #cgo LDFLAGS: -L llama.cpp/build/src  -L llama.cpp/build/ggml/src  -L llama.cpp/build/common -L llama.cpp/build/ggml/src/ggml-metal -L llama.cpp/build/ggml/src/ggml-rpc
// #cgo LDFLAGS: -l llama -l ggml -l ggml-base -l ggml-cpu -l ggml-metal -l ggml-rpc -l common -l m  -l stdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit
// #cgo LDFLAGS: -L llama.cpp/build/ggml/src/ggml-blas -l ggml-blas
// #include <stdlib.h>
// #include "llama.h"
// #include "ggml.h"
// #include "gguf.h"
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"k8s.io/klog/v2"
)

type Token C.llama_token

type LlamaModel struct {
	model *C.struct_llama_model
}

func NewLlamaModelParams() C.struct_llama_model_params {
	return C.llama_model_default_params()
}

func NewLlamaModel(modelPath string, modelParams C.struct_llama_model_params) (*LlamaModel, error) {
	modelPath_cstr := C.CString(modelPath)
	defer C.free(unsafe.Pointer(modelPath_cstr))

	model := C.llama_model_load_from_file(modelPath_cstr, modelParams)
	if model == nil {
		return nil, errors.New("failed to load model")
	}
	return &LlamaModel{model: model}, nil
}

func (m *LlamaModel) Free() {
	if m.model == nil {
		return
	}
	C.llama_model_free(m.model)
	m.model = nil
}

func (m *LlamaModel) GetChatTemplate() (string, bool) {
	if m.model == nil {
		return "", false
	}
	// TODO: pass name (second arg)
	template_cstr := C.llama_model_chat_template(m.model, nil)
	if template_cstr == nil {
		return "", false
	}
	return C.GoString(template_cstr), true
}

func (m *LlamaModel) GetVocab() (*LlamaVocab, error) {
	if m.model == nil {
		return nil, fmt.Errorf("model is nil")
	}
	vocab := C.llama_model_get_vocab(m.model)
	if vocab == nil {
		return nil, fmt.Errorf("llama_model_get_vocab failed")
	}
	return &LlamaVocab{vocab: vocab}, nil
}

type LlamaVocab struct {
	vocab *C.struct_llama_vocab
}

func (v *LlamaVocab) Free() {
	// Owned by the model
	// if v.vocab == nil {
	// 	return
	// }
	// C.llama_vocab_free(v.vocab)
	// v.vocab = nil
}

func (v *LlamaVocab) NTokens() int {
	if v.vocab == nil {
		return -1
	}
	return int(C.llama_vocab_n_tokens(v.vocab))
}

// vocab.IsEog(new_token_id)
func (v *LlamaVocab) IsEog(token Token) bool {
	if v.vocab == nil {
		return false
	}
	ret := C.llama_vocab_is_eog(v.vocab, C.int(token))
	return ret != C.bool(false)
}

// // Function to tokenize the prompt
// static int tokenize_prompt(const llama_context * ctx, const llama_vocab * vocab, const std::string & prompt,
// 	std::vector<llama_token> & prompt_tokens) {
// const bool is_first = llama_get_kv_cache_used_cells(ctx) == 0;

// const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
// prompt_tokens.resize(n_prompt_tokens);
// if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first,
// true) < 0) {
// printf("failed to tokenize the prompt\n");
// return -1;
// }

// return n_prompt_tokens;
// }

func (v *LlamaVocab) TokenizePrompt(prompt string) ([]Token, error) {
	if v.vocab == nil {
		return nil, fmt.Errorf("vocab is nil")
	}

	prompt_cstr := C.CString(prompt)
	defer C.free(unsafe.Pointer(prompt_cstr))
	prompt_len := C.int(len(prompt))
	n_prompt_tokens := -C.llama_tokenize(v.vocab, prompt_cstr, prompt_len, nil, 0, true, true)
	prompt_tokens := make([]Token, n_prompt_tokens)

	prompt_tokens_c := (*C.int)(unsafe.SliceData(prompt_tokens))
	n := C.llama_tokenize(v.vocab, prompt_cstr, prompt_len, prompt_tokens_c, n_prompt_tokens, true, true)
	if n < 0 {
		return nil, fmt.Errorf("failed to tokenize the prompt")
	}
	return prompt_tokens, nil
}

func (v *LlamaVocab) ConvertTokenToString(token Token) (string, error) {
	if v.vocab == nil {
		return "", fmt.Errorf("vocab is nil")
	}

	buf_len := C.int(256)
	buf := make([]byte, buf_len)
	n := C.llama_token_to_piece(v.vocab, C.int(token), (*C.char)(unsafe.Pointer(&buf[0])), buf_len, 0, true)
	if n < 0 {
		return "", fmt.Errorf("failed to convert token to piece")
	}

	return string(buf[:n]), nil
}

// llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
// llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
// llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
// llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

type LlamaSamplerOptions struct {
	MinP *MinPOptions
	Temp float32
	Seed int
}

type MinPOptions struct {
	P       float32
	MinKeep int32
}

func NewLlamaSampler(options LlamaSamplerOptions) *LlamaSampler {
	sampler := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	if minp := options.MinP; minp != nil {
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_min_p(C.float(minp.P), C.size_t(minp.MinKeep)))
	}
	if options.Temp != 0 {
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_temp(C.float(options.Temp)))
	}
	if options.Seed != 0 {
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_dist(C.uint32_t(options.Seed)))
	}
	return &LlamaSampler{sampler: sampler}
}

type LlamaSampler struct {
	sampler *C.struct_llama_sampler
}

func (s *LlamaSampler) Free() {
	if s.sampler == nil {
		return
	}
	C.llama_sampler_free(s.sampler)
}

//   llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// // Return batch for single sequence of tokens
// // The sequence ID will be fixed to 0
// // The position of the tokens will be tracked automatically by llama_decode
// //
// // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
// //
// LLAMA_API struct llama_batch llama_batch_get_one(
// 	llama_token * tokens,
// 		int32_t   n_tokens);

func BatchGetOne(tokens []Token) (*LlamaBatch, error) {
	batch := C.llama_batch_get_one((*C.llama_token)(unsafe.Pointer(&tokens[0])), C.int32_t(len(tokens)))
	// if batch == nil {
	// 	return nil, fmt.Errorf("failed to create llama batch")
	// }
	klog.Infof("TODO: move to new batch API")

	return &LlamaBatch{c: batch}, nil
}

type LlamaBatch struct {
	c C.struct_llama_batch
}

func (b *LlamaBatch) Free() {
	// if b.p == nil {
	// 	return
	// }
	// C.llama_batch_free(b.c)
	// b.p = nil
	klog.Infof("TODO: free LlamaBatch")
}
