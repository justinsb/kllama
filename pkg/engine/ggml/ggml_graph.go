package ggml

// #include <stdlib.h>
// #include "llama.h"
// #include "ggml.h"
// #include "gguf.h"
import "C"
import (
	"errors"
	"fmt"
)

type Graph struct {
	p *C.struct_ggml_cgraph
}

func (ctx *GgmlContext) NewGgmlCGraph() (*Graph, error) {
	p := C.ggml_new_graph(ctx.p)
	if p == nil {
		return nil, errors.New("failed to create GGML graph")
	}
	return &Graph{p: p}, nil
}

func (g *Graph) Free() {
	// if g.p == nil {
	// 	return
	// }
	// C.ggml_free_graph(g.p)
}

func (g *Graph) BuildForwardExpand(f *GgmlTensor) {
	C.ggml_build_forward_expand(g.p, f.p)
}

func (g *Graph) ComputeWithCtx(ctx *GgmlContext, nThreads int) error {
	status := C.ggml_graph_compute_with_ctx(ctx.p, g.p, C.int(nThreads))
	if status != 0 {
		return fmt.Errorf("failed to compute graph (status %d)", status)
	}
	return nil
}
