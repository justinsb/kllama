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
			expected := []float32{11, 18, 27}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}

func TestSilu(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{11}, Values: []float32{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}}},
					{Id: 3, Computation: &api.TensorOperation{Operation: &api.TensorOperation_Silu{Silu: &api.Silu{Source: 1}}}},
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

			expected := []float32{-0.033464253, -0.07194484, -0.14227761, -0.23840584, -0.26894143, 0, 0.7310586, 1.7615942, 2.8577223, 3.928055, 4.9665356}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}

func TestSoftmax(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{11}, Values: []float32{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}}},
					{Id: 3, Computation: &api.TensorOperation{Operation: &api.TensorOperation_Softmax{Softmax: &api.Softmax{Source: 1}}}},
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

			expected := []float32{2.8698709e-05, 7.801117e-05, 0.00021205637, 0.00057642895, 0.0015668964, 0.0042592655, 0.011577884, 0.031471953, 0.08554964, 0.23254804, 0.6321311}
			if !FloatingPointEqual(values, expected) {
				t.Errorf("expected %+v, got %+v", expected, values)
			}
		})
	}
}
func TestMatrixMultiply(t *testing.T) {
	for name, newEngine := range engines() {
		t.Run(name, func(t *testing.T) {
			scope, err := newEngine()
			if err != nil {
				t.Fatalf("failed to create engine: %v", err)
			}
			defer scope.Close()

			request := &api.CalculateRequest{
				Tensors: []*api.Tensor{
					{Id: 1, InlineData: &api.InlineData{Dimensions: []int32{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}}},
					{Id: 2, InlineData: &api.InlineData{Dimensions: []int32{3, 2}, Values: []float32{7, 8, 9, 10, 11, 12}}},
					{Id: 3, Computation: &api.TensorOperation{Operation: &api.TensorOperation_MatrixMultiply{MatrixMultiply: &api.MatrixMultiply{Sources: []int32{1, 2}}}}},
				},
				OutputTensors: []int32{3},
			}

			response, err := engine.Evaluate(scope, request)
			if err != nil {
				t.Fatalf("failed to evaluate: %v", err)
			}

			t.Logf("response: %v", response)

			// Note that response is _not_ valid if we close here

			if len(response.Results) != 1 {
				t.Fatalf("expected 1 result, got %d", len(response.Results))
			}

			if response.Results[0].InlineData == nil {
				t.Fatalf("expected inline data, got nil")
			}
			values := response.Results[0].InlineData.Values

			expected := []float32{58, 64, 139, 154}
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
