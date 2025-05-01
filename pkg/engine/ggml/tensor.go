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
	} else {
		return fmt.Errorf("tensor %d has %d dimensions, expected 1", t.definition.GetId(), nDimensions)
	}
}

func (t *tensor) Dependencies() []TensorID {
	return t.dependencies
}

func (t *tensor) TensorID() TensorID {
	return t.id
}
