package fallback

import (
	"fmt"
	"slices"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
)

type tensor struct {
	id         TensorID
	definition *api.Tensor
	inlineData *api.InlineData

	dimensions   []int32
	dependencies []TensorID
}

func newTensor(definition *api.Tensor) *tensor {
	id := TensorID(definition.GetId())
	t := &tensor{
		id:         id,
		definition: definition,
	}
	if inlineData := definition.GetInlineData(); inlineData != nil {
		t.inlineData = inlineData
		t.dimensions = inlineData.GetDimensions()
	}

	if computation := definition.GetComputation(); computation != nil {
		dependencies := engine.GetDependencies(computation)
		t.dependencies = append(t.dependencies, dependencies...)
	}

	return t
}

func (t *tensor) NDimensions() int {
	return len(t.dimensions)
}

func sameSize(t1 *tensor, t2 *tensor) bool {
	return slices.Equal(t1.dimensions, t2.dimensions)
}

func (t *tensor) CopyDataTo(result *api.Tensor) error {
	if t.inlineData == nil {
		return fmt.Errorf("tensor %d has no inline data", t.definition.GetId())
	}
	// TODO: copy the inline data
	result.InlineData = t.inlineData
	return nil
}

func (t *tensor) Dependencies() []TensorID {
	return t.dependencies
}

func (t *tensor) TensorID() TensorID {
	return t.id
}
