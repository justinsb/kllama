package ggml

import (
	"fmt"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
)

type TensorID = engine.TensorID

type CalculationScope struct {
	tensors map[TensorID]*tensor

	ggmlContext *GgmlContext
}

func NewCalculationScope() (*CalculationScope, error) {
	ggmlContext, err := NewGgmlContext(NewGgmlInitParams(1024 * 1024 * 1024))
	if err != nil {
		return nil, err
	}
	return &CalculationScope{
		tensors:     make(map[TensorID]*tensor),
		ggmlContext: ggmlContext,
	}, nil
}

func (c *CalculationScope) Close() error {
	c.ggmlContext.Free()
	// TODO: free tensors?
	return nil
}

func (c *CalculationScope) Evaluate(wantTensors []TensorID) error {
	graph, err := c.ggmlContext.NewGgmlCGraph()
	if err != nil {
		return fmt.Errorf("failed to create graph: %w", err)
	}
	defer graph.Free()

	// Note: order matters when calling BuildForwardExpand!
	evaluationOrder, err := engine.BuildDAG(c, wantTensors)
	if err != nil {
		return fmt.Errorf("failed to build DAG: %w", err)
	}

	for _, tensorID := range evaluationOrder {
		tensor, ok := c.tensors[tensorID]
		if !ok {
			return fmt.Errorf("tensor %d not found", tensorID)
		}
		if tensor.ggmlTensor != nil {
			continue
		}
		if err := c.addComputedTensor(tensor); err != nil {
			return fmt.Errorf("failed to add GGML tensor: %w", err)
		}
	}

	for _, tensorID := range evaluationOrder {
		tensor, ok := c.tensors[tensorID]
		if !ok {
			return fmt.Errorf("tensor %d not found", tensorID)
		}
		if tensor.ggmlTensor == nil {
			return fmt.Errorf("tensor %d has no GGML tensor", tensorID)
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

		c.tensors[id] = t
	}
	return nil
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
			source := TensorID(operation.RmsNorm.GetSource())
			sourceTensor, found := c.tensors[source]
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

		case *api.TensorOperation_DotMultiply:
			sourceTensors, err := c.getSourceTensors(operation.DotMultiply.GetSources()...)
			if err != nil {
				return err
			}
			for _, sourceTensor := range sourceTensors {
				if sourceTensor.ggmlTensor == nil {
					return fmt.Errorf("source tensor %d has no GGML tensor", sourceTensor.id)
				}
			}
			if len(sourceTensors) != 2 {
				return fmt.Errorf("expected 2 source tensors, got %d", len(sourceTensors))
			}

			dotProduct := c.ggmlContext.GgmlMul(sourceTensors[0].ggmlTensor, sourceTensors[1].ggmlTensor)

			tensor.ggmlTensor = dotProduct
			return nil

		case *api.TensorOperation_Add:
			sourceTensors, err := c.getSourceTensors(operation.Add.GetSources()...)
			if err != nil {
				return err
			}
			for _, sourceTensor := range sourceTensors {
				if sourceTensor.ggmlTensor == nil {
					return fmt.Errorf("source tensor %d has no GGML tensor", sourceTensor.id)
				}
			}

			add := c.ggmlContext.GgmlAdd(sourceTensors[0].ggmlTensor, sourceTensors[1].ggmlTensor)
			tensor.ggmlTensor = add
			return nil
		default:
			return fmt.Errorf("unsupported operation: %T %+v", operation, operation)
		}
	}

	return fmt.Errorf("tensor %d has no computation", tensor.definition.GetId())
}

func (c *CalculationScope) getSourceTensors(dependencies ...int32) ([]*tensor, error) {
	out := make([]*tensor, len(dependencies))
	for i, dependency := range dependencies {
		dependencyTensor, found := c.tensors[TensorID(dependency)]
		if !found {
			return nil, fmt.Errorf("source tensor %d not found", dependency)
		}
		out[i] = dependencyTensor
	}
	return out, nil
}
