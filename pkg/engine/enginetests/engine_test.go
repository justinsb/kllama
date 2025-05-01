package enginetests

import (
	"math"
	"testing"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
	"github.com/justinsb/kllama/pkg/engine/fallback"
	"github.com/justinsb/kllama/pkg/engine/ggml"
)

func engines() map[string]func() (engine.Scope, error) {
	return map[string]func() (engine.Scope, error){
		"ggml":     func() (engine.Scope, error) { return ggml.NewCalculationScope() },
		"fallback": func() (engine.Scope, error) { return fallback.NewCalculationScope() },
	}
}

func TestEngine(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{1, 2, 3}}},
					{Id: 2, Computation: &api.TensorOperation{Operation: &api.TensorOperation_RmsNorm{RmsNorm: &api.RMSNorm{Source: 1}}}},
				},
				OutputTensors: []int32{2},
			}

			response, err := engine.Evaluate(scope, request)
			if err != nil {
				t.Fatalf("failed to evaluate: %v", err)
			}

			t.Logf("response: %v", response)

			// Verify that we can close scope multiple times, and we don't reference freed memory reading response
			if err := scope.Close(); err != nil {
				t.Fatalf("failed to free scope: %v", err)
			}

			if len(response.Results) != 1 {
				t.Fatalf("expected 1 result, got %d", len(response.Results))
			}

			if response.Results[0].InlineData == nil {
				t.Fatalf("expected inline data, got nil")
			}
			values := response.Results[0].InlineData.Values
			if len(values) != 3 {
				t.Fatalf("expected 3 values, got %d", len(values))
			}
			expected := []float32{0.46290955, 0.9258191, 1.3887286}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}

func TestDotMultiply(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{1, 2, 3}}},
					{Id: 2, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{4, 5, 6}}},
					{Id: 3, Computation: &api.TensorOperation{Operation: &api.TensorOperation_DotMultiply{DotMultiply: &api.DotMultiply{Sources: []int32{1, 2}}}}},
				},
				OutputTensors: []int32{3},
			}

			response, err := engine.Evaluate(scope, request)
			if err != nil {
				t.Fatalf("failed to evaluate: %v", err)
			}

			t.Logf("response: %v", response)

			// Verify that we can close scope multiple times, and we don't reference freed memory reading response
			if err := scope.Close(); err != nil {
				t.Fatalf("failed to free scope: %v", err)
			}

			if len(response.Results) != 1 {
				t.Fatalf("expected 1 result, got %d", len(response.Results))
			}

			if response.Results[0].InlineData == nil {
				t.Fatalf("expected inline data, got nil")
			}
			values := response.Results[0].InlineData.Values
			if len(values) != 3 {
				t.Fatalf("expected 3 values, got %d", len(values))
			}
			expected := []float32{4, 10, 18}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{1, 2, 3}}},
					{Id: 2, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{4, 5, 6}}},
					{Id: 3, Computation: &api.TensorOperation{Operation: &api.TensorOperation_DotMultiply{DotMultiply: &api.DotMultiply{Sources: []int32{1, 2}}}}},
					{Id: 4, Computation: &api.TensorOperation{Operation: &api.TensorOperation_Add{Add: &api.Add{Sources: []int32{3, 5}}}}},
					{Id: 5, InlineData: &api.InlineData{Dimensions: []int32{3}, Values: []float32{7, 8, 9}}},
				},
				OutputTensors: []int32{4},
			}

			response, err := engine.Evaluate(scope, request)
			if err != nil {
				t.Fatalf("failed to evaluate: %v", err)
			}

			t.Logf("response: %v", response)

			// Verify that we can close scope multiple times, and we don't reference freed memory reading response
			if err := scope.Close(); err != nil {
				t.Fatalf("failed to free scope: %v", err)
			}

			if len(response.Results) != 1 {
				t.Fatalf("expected 1 result, got %d", len(response.Results))
			}

			if response.Results[0].InlineData == nil {
				t.Fatalf("expected inline data, got nil")
			}
			values := response.Results[0].InlineData.Values
			if len(values) != 3 {
				t.Fatalf("expected 3 values, got %d", len(values))
			}
			expected := []float32{11, 18, 27}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}

func FloatingPointEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, value := range a {
		if math.Abs(float64(value-b[i])) > 0.00001 {
			return false
		}
	}
	return true
}
