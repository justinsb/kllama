package engine

import (
	"io"

	api "github.com/justinsb/kllama/api/v1alpha1"
)

type Scope interface {
	io.Closer

	RegisterTensors(tensors []*api.Tensor) error
	GetTensor(id int32) (Tensor, bool)
	Evaluate(wantTensors []int32) error
}

type Tensor interface {
	CopyDataTo(result *api.Tensor) error
}
