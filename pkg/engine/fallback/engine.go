package fallback

import (
	"fmt"
	"math"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
)

type TensorID = engine.TensorID

type CalculationScope struct {
	tensors map[TensorID]*tensor
}

func NewCalculationScope() (*CalculationScope, error) {
	return &CalculationScope{
		tensors: make(map[TensorID]*tensor),
	}, nil
}

func (c *CalculationScope) Close() error {
	return nil
}

func (c *CalculationScope) AllTensors() map[TensorID]engine.Tensor {
	tensors := make(map[TensorID]engine.Tensor, len(c.tensors))
	for _, tensor := range c.tensors {
		tensors[tensor.id] = tensor
	}
	return tensors
}

func (c *CalculationScope) RegisterTensors(tensors []*api.Tensor) error {
	for _, definition := range tensors {
		id := TensorID(definition.GetId())
		if _, ok := c.tensors[id]; ok {
			return fmt.Errorf("tensor %d already registered", definition.GetId())
		}

		t := &tensor{
			id:         id,
			definition: definition,
		}
		if inlineData := definition.GetInlineData(); inlineData != nil {
			t.inlineData = inlineData
		}

		if computation := definition.GetComputation(); computation != nil {
			dependencies := engine.GetDependencies(computation)
			t.dependencies = append(t.dependencies, dependencies...)
		}

		c.tensors[id] = t
	}
	return nil
}

func (c *CalculationScope) Evaluate(wantTensors []TensorID) error {
	evaluationOrder, err := engine.BuildDAG(c, wantTensors)
	if err != nil {
		return err
	}

	for _, tensorID := range evaluationOrder {
		tensor, ok := c.tensors[tensorID]
		if !ok {
			return fmt.Errorf("tensor %d not found", tensorID)
		}
		if err := c.evaluateTensor(tensor); err != nil {
			return err
		}
	}

	return nil
}

func (c *CalculationScope) evaluateTensor(tensor *tensor) error {
	if tensor.inlineData != nil {
		return nil
	}

	if computation := tensor.definition.GetComputation(); computation != nil {
		operation := computation.GetOperation()
		switch operation := operation.(type) {
		case *api.TensorOperation_LinearScale:
			result := &api.Tensor{}
			source := TensorID(operation.LinearScale.GetSource())
			sourceTensor, found := c.tensors[source]
			if !found {
				return fmt.Errorf("source tensor %d not found", source)
			}
			if err := sourceTensor.CopyDataTo(result); err != nil {
				return err
			}
			scale := operation.LinearScale.GetScale()
			values := result.GetInlineData().GetValues()
			for i := range values {
				values[i] *= scale
			}
			tensor.inlineData = &api.InlineData{Values: values}
			return nil

		case *api.TensorOperation_RmsNorm:
			result := &api.Tensor{}
			source := TensorID(operation.RmsNorm.GetSource())
			sourceTensor, found := c.tensors[source]
			if !found {
				return fmt.Errorf("source tensor %d not found", source)
			}
			if err := sourceTensor.CopyDataTo(result); err != nil {
				return err
			}
			epsilon := float32(1e-5)
			if v := operation.RmsNorm.GetEpsilon(); v != 0 {
				epsilon = v
			}

			values := result.GetInlineData().GetValues()
			sum_x2 := float32(0)
			for i := range values {
				v := values[i]
				sum_x2 += v * v
			}
			mean := sum_x2 / float32(len(values))
			rms := float32(1.0 / math.Sqrt(float64(mean)+float64(epsilon)))
			for i := range values {
				values[i] *= rms
			}
			tensor.inlineData = &api.InlineData{Values: values}
			return nil

		default:
			return fmt.Errorf("unsupported operation: %v", operation)
		}

	}

	return fmt.Errorf("tensor %d has no computation", tensor.definition.GetId())
}
