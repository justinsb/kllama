package fallback

import (
	"fmt"
	"math"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
)

type CalculationScope struct {
	tensors map[int32]*tensor
}

func NewCalculationScope() (*CalculationScope, error) {
	return &CalculationScope{
		tensors: make(map[int32]*tensor),
	}, nil
}

func (c *CalculationScope) Close() error {
	return nil
}

func (c *CalculationScope) GetTensor(id int32) (engine.Tensor, bool) {
	tensor, ok := c.tensors[id]
	return tensor, ok
}

func (c *CalculationScope) getTensor(id int32) (*tensor, bool) {
	tensor, ok := c.tensors[id]
	return tensor, ok
}

func (c *CalculationScope) RegisterTensors(tensors []*api.Tensor) error {
	for _, definition := range tensors {
		if _, ok := c.tensors[definition.GetId()]; ok {
			return fmt.Errorf("tensor %d already registered", definition.GetId())
		}

		t := &tensor{
			definition: definition,
		}
		if inlineData := definition.GetInlineData(); inlineData != nil {
			t.inlineData = inlineData
		}

		if computation := definition.GetComputation(); computation != nil {
			dependencies := engine.GetDependencies(computation)
			t.dependencies = append(t.dependencies, dependencies...)
		}

		c.tensors[definition.GetId()] = t
	}
	return nil
}

type tensor struct {
	definition *api.Tensor
	inlineData *api.InlineData

	dependencies []int32
}

func (t *tensor) CopyDataTo(result *api.Tensor) error {
	if t.inlineData == nil {
		return fmt.Errorf("tensor %d has no inline data", t.definition.GetId())
	}
	// TODO: copy the inline data
	result.InlineData = t.inlineData
	return nil
}

func (c *CalculationScope) Evaluate(wantTensors []int32) error {
	evaluationOrder := make([]*tensor, 0, len(c.tensors))
	done := make(map[int32]bool)

	for {
		progress := false
		for id, tensor := range c.tensors {
			if done[id] {
				continue
			}

			ready := true
			for _, dep := range tensor.dependencies {
				if !done[dep] {
					ready = false
					break
				}
			}
			if ready {
				done[id] = true
				evaluationOrder = append(evaluationOrder, tensor)
				progress = true
			}
		}
		if !progress {
			break
		}
	}

	// TODO: Check that all output tensors are in the DAG
	// TODO: Only evaluate output tensors?

	for _, tensor := range evaluationOrder {
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
			source := operation.LinearScale.GetSource()
			sourceTensor, found := c.getTensor(source)
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
			source := operation.RmsNorm.GetSource()
			sourceTensor, found := c.getTensor(source)
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
