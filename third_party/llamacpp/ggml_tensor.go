package llamacpp

// #cgo CFLAGS: -O3 -DNDEBUG -I llama.cpp/include -I llama.cpp/ggml/include

// #include <stdlib.h>
// #include "llama.h"
// #include "ggml.h"
// #include "gguf.h"
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type GgmlTensor struct {
	p *C.struct_ggml_tensor
}

func (t *GgmlTensor) String() string {
	return fmt.Sprintf("GgmlTensor{dimensions: %v}", t.Shape())
}

func (t *GgmlTensor) Shape() []int {
	ndims := t.GetNDims()
	dims := make([]int, ndims)
	for i := 0; i < ndims; i++ {
		dims[i] = int(t.p.ne[i])
	}
	return dims
}

type GgmlType C.enum_ggml_type

const (
	GGML_TYPE_F32 GgmlType = C.GGML_TYPE_F32
	GGML_TYPE_F16 GgmlType = C.GGML_TYPE_F16
	GGML_TYPE_I32 GgmlType = C.GGML_TYPE_I32
)

func (ctx *GgmlContext) NewGgmlTensor1D(t GgmlType, numElements int32) (*GgmlTensor, error) {
	p := C.ggml_new_tensor_1d(ctx.p, C.enum_ggml_type(t), C.int64_t(numElements))
	if p == nil {
		return nil, errors.New("failed to create GGML tensor")
	}
	return &GgmlTensor{p: p}, nil
}

func (ctx *GgmlContext) NewGgmlTensor2D(t GgmlType, ne0 int32, ne1 int32) (*GgmlTensor, error) {
	p := C.ggml_new_tensor_2d(ctx.p, C.enum_ggml_type(t), C.int64_t(ne0), C.int64_t(ne1))
	if p == nil {
		return nil, errors.New("failed to create GGML tensor")
	}
	return &GgmlTensor{p: p}, nil
}

func (t *GgmlTensor) GetNDims() int {
	return int(C.ggml_n_dims(t.p))
}

func (t *GgmlTensor) GetNelements() int64 {
	return int64(C.ggml_nelements(t.p))
}

func (t *GgmlTensor) GetNrows() int64 {
	return int64(C.ggml_nrows(t.p))
}

func (t *GgmlTensor) GetTensorType() int {
	return int(t.p._type)
}

func (t *GgmlTensor) IsContiguous() bool {
	return bool(C.ggml_is_contiguous(t.p))
}

func (t *GgmlTensor) GetValues_1D_F32() ([]float32, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("tensor is not contiguous")
	}
	values := make([]float32, t.GetNelements())
	data := unsafe.Pointer(C.ggml_get_data_f32(t.p))
	for i := range values {
		values[i] = *(*float32)(data)
		data = unsafe.Pointer(uintptr(data) + 4)
	}
	return values, nil
}

func (t *GgmlTensor) GetValues() ([]float32, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("tensor is not contiguous")
	}
	data := unsafe.Pointer(C.ggml_get_data_f32(t.p))
	n := t.GetNelements()
	return unsafe.Slice((*float32)(data), n), nil
}

func (t *GgmlTensor) GetRowStride() int64 {
	return int64(t.p.ne[1])
}

func (t *GgmlTensor) SetValues(values []float32) error {
	if !t.IsContiguous() {
		return fmt.Errorf("tensor is not contiguous")
	}
	n := t.GetNelements()
	if n != int64(len(values)) {
		return fmt.Errorf("tensor has %d elements, but %d values were provided", n, len(values))
	}
	data := unsafe.Pointer(C.ggml_get_data_f32(t.p))
	for i := range values {
		*(*float32)(data) = values[i]
		data = unsafe.Pointer(uintptr(data) + 4)
	}

	// {
	// 	out := make([]float32, t.GetNelements())
	// 	data := unsafe.Pointer(C.ggml_get_data_f32(t.p))
	// 	for i := range values {
	// 		out[i] = *(*float32)(data)
	// 		data = unsafe.Pointer(uintptr(data) + 4)
	// 		if out[i] != values[i] {
	// 			return fmt.Errorf("expected %v, got %v", values, out)
	// 		}
	// 	}
	// }
	// data := (*[1 << 31]float32)(unsafe.Pointer(C.ggml_get_data_f32(t.p)))[:n:n]
	// for i := range values {
	// 	data[i] = values[i]
	// }
	return nil
}
