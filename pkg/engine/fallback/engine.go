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
		t := newTensor(definition)

		if _, ok := c.tensors[t.id]; ok {
			return fmt.Errorf("tensor %d already registered", definition.GetId())
		}

		c.tensors[t.id] = t
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
			tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor.dimensions}
			tensor.dimensions = tensor.inlineData.Dimensions
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
			tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor.dimensions}
			tensor.dimensions = tensor.inlineData.Dimensions
			return nil

		case *api.TensorOperation_DotMultiply:
			sourceTensors, err := c.getSourceTensors(operation.DotMultiply.GetSources()...)
			if err != nil {
				return err
			}
			if len(sourceTensors) < 2 {
				return fmt.Errorf("expected at least 2 source tensors, got %d", len(sourceTensors))
			}
			for _, sourceTensor := range sourceTensors {
				if sourceTensor.inlineData == nil {
					return fmt.Errorf("source tensor %d has no inline data", sourceTensor.id)
				}
			}
			if len(sourceTensors) != 2 {
				return fmt.Errorf("expected 2 source tensors, got %d", len(sourceTensors))
			}
			sourceTensor0 := sourceTensors[0]
			sourceTensor1 := sourceTensors[1]

			if !sameSize(sourceTensor0, sourceTensor1) {
				return fmt.Errorf("tensors %v and %v have different sizes", sourceTensor0, sourceTensor1)
			}

			if sourceTensor0.NDimensions() == 1 {
				v0 := sourceTensor0.inlineData.GetValues()
				v1 := sourceTensor1.inlineData.GetValues()
				values := make([]float32, len(v0))
				for i := range v0 {
					values[i] = v0[i] * v1[i]
				}
				tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor0.dimensions}
				tensor.dimensions = tensor.inlineData.Dimensions
				return nil
			} else {
				return fmt.Errorf("unsupported tensor dimensions: %d", sourceTensor0.NDimensions())
			}

		case *api.TensorOperation_MatrixMultiply:
			sourceTensors, err := c.getSourceTensors(operation.MatrixMultiply.GetSources()...)
			if err != nil {
				return err
			}
			if len(sourceTensors) < 2 {
				return fmt.Errorf("expected at least 2 source tensors, got %d", len(sourceTensors))
			}
			for _, sourceTensor := range sourceTensors {
				if sourceTensor.inlineData == nil {
					return fmt.Errorf("source tensor %d has no inline data", sourceTensor.id)
				}
			}
			if len(sourceTensors) != 2 {
				return fmt.Errorf("expected 2 source tensors, got %d", len(sourceTensors))
			}
			sourceTensor0 := sourceTensors[0]
			sourceTensor1 := sourceTensors[1]

			if sourceTensor0.NDimensions() == 2 && sourceTensor1.NDimensions() == 2 {
				if sourceTensor0.ColumnCount() != sourceTensor1.RowCount() {
					return fmt.Errorf("tensors %v and %v cannot be multiplied", sourceTensor0, sourceTensor1)
				}

				nOut := sourceTensor0.RowCount() * sourceTensor1.ColumnCount()
				values := make([]float32, nOut)

				p0 := sourceTensor0.inlineData.GetValues()
				p1 := sourceTensor1.inlineData.GetValues()

				columnStride0 := sourceTensor0.ColumnCount()
				columnStride1 := sourceTensor1.ColumnCount()

				for i := 0; i < sourceTensor0.RowCount(); i++ {
					for j := 0; j < sourceTensor1.ColumnCount(); j++ {
						sum := float32(0)
						for k := 0; k < sourceTensor0.ColumnCount(); k++ {
							sum += p0[i*int(columnStride0)+k] * p1[k*int(columnStride1)+j]
						}
						values[i*int(columnStride1)+j] = sum
					}
				}

				tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor0.dimensions}
				tensor.dimensions = tensor.inlineData.Dimensions
				return nil
			} else {
				return fmt.Errorf("unsupported tensor dimensions: %d and %d	", sourceTensor0.NDimensions(), sourceTensor1.NDimensions())
			}

		case *api.TensorOperation_Add:
			sourceTensors, err := c.getSourceTensors(operation.Add.GetSources()...)
			if err != nil {
				return err
			}
			if len(sourceTensors) != 2 {
				return fmt.Errorf("expected 2 source tensors, got %d", len(sourceTensors))
			}
			sourceTensor0 := sourceTensors[0]
			sourceTensor1 := sourceTensors[1]

			if !sameSize(sourceTensor0, sourceTensor1) {
				return fmt.Errorf("tensors %v and %v have different sizes", sourceTensor0, sourceTensor1)
			}

			if sourceTensor0.NDimensions() == 1 {
				values0 := sourceTensor0.inlineData.GetValues()
				values1 := sourceTensor1.inlineData.GetValues()
				values := make([]float32, len(values0))
				for i := range values {
					values[i] = values0[i] + values1[i]
				}
				tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor0.dimensions}
				tensor.dimensions = tensor.inlineData.Dimensions
				return nil
			} else {
				return fmt.Errorf("unsupported tensor dimensions: %d", sourceTensor0.NDimensions())
			}

		case *api.TensorOperation_Silu:
			result := &api.Tensor{}
			source := TensorID(operation.Silu.GetSource())
			sourceTensor, found := c.tensors[source]
			if !found {
				return fmt.Errorf("source tensor %d not found", source)
			}
			if err := sourceTensor.CopyDataTo(result); err != nil {
				return err
			}

			if sourceTensor.NDimensions() == 1 {
				src := sourceTensor.inlineData.GetValues()
				values := make([]float32, len(src))
				for i, v := range src {
					v64 := float64(v)
					values[i] = float32(v64 / (1.0 + math.Exp(-v64)))
				}
				tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor.dimensions}
				tensor.dimensions = tensor.inlineData.Dimensions
				return nil
			} else {
				return fmt.Errorf("unsupported tensor dimensions: %d", sourceTensor.NDimensions())
			}

		case *api.TensorOperation_Softmax:
			result := &api.Tensor{}
			source := TensorID(operation.Softmax.GetSource())
			sourceTensor, found := c.tensors[source]
			if !found {
				return fmt.Errorf("source tensor %d not found", source)
			}
			if err := sourceTensor.CopyDataTo(result); err != nil {
				return err
			}

			if sourceTensor.NDimensions() == 1 {
				src := sourceTensor.inlineData.GetValues()

				// See https://en.wikipedia.org/wiki/Softmax_function,
				// in particular the "Numerical stability" section
				max := src[0]
				for _, v := range src {
					if v > max {
						max = v
					}
				}

				sumExp := float64(0)
				for _, v := range src {
					sumExp += math.Exp(float64(v - max))
				}

				values := make([]float32, len(src))
				for i, v := range src {
					values[i] = float32(math.Exp(float64(v-max)) / sumExp)
				}

				tensor.inlineData = &api.InlineData{Values: values, Dimensions: sourceTensor.dimensions}
				tensor.dimensions = tensor.inlineData.Dimensions
				return nil
			} else {
				return fmt.Errorf("unsupported tensor dimensions: %d", sourceTensor.NDimensions())
			}

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
