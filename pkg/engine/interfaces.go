package engine

import (
	"io"

	api "github.com/justinsb/kllama/api/v1alpha1"
)

type TensorID int32

type Scope interface {
	io.Closer

	RegisterTensors(tensors []*api.Tensor) error
	AllTensors() map[TensorID]Tensor
	Evaluate(wantTensors []TensorID) error
}

type Tensor interface {
	TensorID() TensorID
	CopyDataTo(result *api.Tensor) error
	Dependencies() []TensorID
}
