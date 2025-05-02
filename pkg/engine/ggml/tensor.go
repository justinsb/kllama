package ggml

import (
	"fmt"

	api "github.com/justinsb/kllama/api/v1alpha1"
)

type tensor struct {
	id         TensorID
	definition *api.Tensor

	dependencies []TensorID
	ggmlTensor   *GgmlTensor
}

func (t *tensor) CopyDataTo(result *api.Tensor) error {
	if t.ggmlTensor == nil {
		return fmt.Errorf("copy on tensor %d with no GGML tensor", t.definition.GetId())
	}
	nDimensions := t.ggmlTensor.GetNDims()
	if nDimensions == 1 {
		if !t.ggmlTensor.IsContiguous() {
			return fmt.Errorf("tensor %d is not contiguous", t.definition.GetId())
		}
		values, err := t.ggmlTensor.GetValues_1D_F32()
		if err != nil {
			return fmt.Errorf("getting values: %v", err)
		}
		result.InlineData = &api.InlineData{Values: values}
		return nil
	} else if nDimensions == 2 {
		if !t.ggmlTensor.IsContiguous() {
			return fmt.Errorf("tensor %d is not contiguous", t.definition.GetId())
		}
		values, err := t.ggmlTensor.GetValues()
		if err != nil {
			return fmt.Errorf("getting values: %v", err)
		}

		// int64_t ne[GGML_MAX_DIMS]; // number of elements
		// size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
		//                            // nb[0] = ggml_type_size(type)
		//                            // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
		//                            // nb[i] = nb[i-1] * ne[i-1]

		result.InlineData = &api.InlineData{Values: values}
		return nil
	} else if nDimensions == 3 {
		if !t.ggmlTensor.IsContiguous() {
			return fmt.Errorf("tensor %d is not contiguous", t.definition.GetId())
		}
		values, err := t.ggmlTensor.GetValues()
		if err != nil {
			return fmt.Errorf("getting values: %v", err)
		}

		// int64_t ne[GGML_MAX_DIMS]; // number of elements
		// size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
		//                            // nb[0] = ggml_type_size(type)
		//                            // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
		//                            // nb[i] = nb[i-1] * ne[i-1]

		result.InlineData = &api.InlineData{Values: values}
		return nil
	} else {
		return fmt.Errorf("tensor %d has %d dimensions, expected 1, 2 or 3", t.definition.GetId(), nDimensions)
	}
}

func (t *tensor) Dependencies() []TensorID {
	return t.dependencies
}

func (t *tensor) TensorID() TensorID {
	return t.id
}
