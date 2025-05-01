package ggml

// #cgo CFLAGS: -O3 -DNDEBUG -I llama.cpp/include -I llama.cpp/ggml/include
// #cgo LDFLAGS: -L llama.cpp/build/src  -L llama.cpp/build/ggml/src  -L llama.cpp/build/common -l llama -l ggml -l ggml-base -l ggml-cpu -l common -l m  -l stdc++ -framework Accelerate
// #include <stdlib.h>
// #include "llama.h"
// #include "ggml.h"
// #include "gguf.h"
import "C"

import (
	"errors"
)

func NewGgmlInitParams(memorySize int) C.struct_ggml_init_params {
	return C.struct_ggml_init_params{
		mem_size:   C.size_t(memorySize),
		mem_buffer: nil,
		no_alloc:   false,
	}
}

type GgmlContext struct {
	p *C.struct_ggml_context
}

func NewGgmlContext(params C.struct_ggml_init_params) (*GgmlContext, error) {
	p := C.ggml_init(params)
	if p == nil {
		return nil, errors.New("failed to initialize GGML context")
	}
	return &GgmlContext{p: p}, nil
}

// GgmlRMSNorm
func (ctx *GgmlContext) GgmlRMSNorm(t *GgmlTensor, eps float32) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_rms_norm(ctx.p, t.p, C.float(eps))}
}

func (ctx *GgmlContext) Free() {
	if ctx.p == nil {
		return
	}
	C.ggml_free(ctx.p)
	ctx.p = nil
}
