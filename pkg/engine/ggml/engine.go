package ggml

import (
	"fmt"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
)

type CalculationScope struct {
	tensors map[int32]*tensor

	ggmlContext *GgmlContext
}

func NewCalculationScope() (*CalculationScope, error) {
	ggmlContext, err := NewGgmlContext(NewGgmlInitParams(1024 * 1024 * 1024))
	if err != nil {
		return nil, err
	}
	return &CalculationScope{
		tensors:     make(map[int32]*tensor),
		ggmlContext: ggmlContext,
	}, nil
}

func (c *CalculationScope) Close() error {
	c.ggmlContext.Free()
	// TODO: free tensors?
	return nil
}

func (c *CalculationScope) Evaluate(wantTensors []int32) error {
	graph, err := c.ggmlContext.NewGgmlCGraph()
	if err != nil {
		return fmt.Errorf("failed to create graph: %w", err)
	}
	defer graph.Free()

	// Note: order matters when calling BuildForwardExpand!
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

	for _, tensor := range evaluationOrder {
		if tensor.ggmlTensor != nil {
			continue
		}
		if err := c.addComputedTensor(tensor); err != nil {
			return fmt.Errorf("failed to add GGML tensor: %w", err)
		}
	}

	for _, tensor := range evaluationOrder {
		if tensor.ggmlTensor == nil {
			return fmt.Errorf("tensor %d has no GGML tensor", tensor.definition.GetId())
		}
		graph.BuildForwardExpand(tensor.ggmlTensor)
	}

	// TODO: Check that all output tensors are in the DAG
	// TODO: Only evaluate output tensors?

	numThreads := 16
	if err := graph.ComputeWithCtx(c.ggmlContext, numThreads); err != nil {
		return fmt.Errorf("failed to compute graph: %w", err)
	}

	return nil
}

func (c *CalculationScope) GetTensor(id int32) (engine.Tensor, bool) {
	tensor, ok := c.tensors[id]
	return tensor, ok
}

func (c *CalculationScope) getTensor(id int32) (*tensor, bool) {
	tensor, ok := c.tensors[id]
	if !ok {
		return nil, false
	}
	return tensor, true
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
			if len(inlineData.Dimensions) == 1 {
				n := len(inlineData.GetValues())
				ggmlTensor, err := c.ggmlContext.NewGgmlTensor1D(GGML_TYPE_F32, n)
				if err != nil {
					return fmt.Errorf("creating GGML tensor: %v", err)
				}
				if err := ggmlTensor.SetValues(inlineData.GetValues()); err != nil {
					return fmt.Errorf("setting values: %v", err)
				}
				t.ggmlTensor = ggmlTensor
			} else {
				return fmt.Errorf("inline data %d has %d dimensions, expected 1", definition.GetId(), len(inlineData.Dimensions))
			}
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
	// inlineData *api.InlineData

	dependencies []int32
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

func (c *CalculationScope) addComputedTensor(tensor *tensor) error {
	if tensor.ggmlTensor != nil {
		return nil
	}

	if source := tensor.definition.GetSource(); source != nil {
		// TODO
		// if inlineData := source.GetInlineData(); inlineData != nil {
		// 	tensor.inlineData = inlineData
		// }
	}

	if computation := tensor.definition.GetComputation(); computation != nil {
		operation := computation.GetOperation()
		switch operation := operation.(type) {
		// case *api.TensorOperation_LinearScale:
		// 	result := &api.Tensor{}
		// 	source := operation.LinearScale.GetSource()
		// 	sourceTensor, found := c.GetTensor(source)
		// 	if !found {
		// 		return fmt.Errorf("source tensor %d not found", source)
		// 	}
		// 	if err := c.Evaluate(sourceTensor); err != nil {
		// 		return err
		// 	}
		// 	if err := sourceTensor.CopyDataTo(result); err != nil {
		// 		return err
		// 	}
		// 	scale := operation.LinearScale.GetScale()
		// 	values := result.GetInlineData().GetValues()
		// 	for i := range values {
		// 		values[i] *= scale
		// 	}
		// 	tensor.inlineData = &api.InlineData{Values: values}

		case *api.TensorOperation_RmsNorm:
			source := operation.RmsNorm.GetSource()
			sourceTensor, found := c.getTensor(source)
			if !found {
				return fmt.Errorf("source tensor %d not found", source)
			}
			if sourceTensor.ggmlTensor == nil {
				return fmt.Errorf("source tensor %d has no GGML tensor", source)
			}

			epsilon := float32(1e-5)
			if v := operation.RmsNorm.GetEpsilon(); v != 0 {
				epsilon = v
			}
			rmsNorm := c.ggmlContext.GgmlRMSNorm(sourceTensor.ggmlTensor, epsilon)

			tensor.ggmlTensor = rmsNorm
			return nil

		default:
			return fmt.Errorf("unsupported operation: %v", operation)
		}

		return nil
	}

	return fmt.Errorf("tensor %d has no computation", tensor.definition.GetId())
}
