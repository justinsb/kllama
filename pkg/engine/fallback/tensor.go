package fallback

import (
	"fmt"

	api "github.com/justinsb/kllama/api/v1alpha1"
)

type tensor struct {
	id         TensorID
	definition *api.Tensor
	inlineData *api.InlineData

	dependencies []TensorID
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
