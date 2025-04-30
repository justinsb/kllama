package engine

import (
	"fmt"

	api "github.com/justinsb/kllama/api/v1alpha1"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func Evaluate(scope Scope, req *api.CalculateRequest) (*api.CalculateResponse, error) {
	if err := scope.RegisterTensors(req.GetTensors()); err != nil {
		return nil, err
	}

	response := &api.CalculateResponse{}
	//	for _, outputTensorID := range req.GetOutputTensors() {
	//		tensor := scope.GetTensor(outputTensorID)
	//		if err := tensor.CopyDataTo(response.Results[outputTensorID]); err != nil {
	//			return nil, err
	//		}
	//	}

	if err := scope.Evaluate(req.GetOutputTensors()); err != nil {
		return nil, err
	}

	for _, outputTensorID := range req.GetOutputTensors() {
		tensor, found := scope.GetTensor(outputTensorID)
		if !found {
			return nil, status.Errorf(codes.InvalidArgument, "tensor %d not found", outputTensorID)
		}
		result := &api.Tensor{
			Id: outputTensorID,
		}
		if err := tensor.CopyDataTo(result); err != nil {
			return nil, err
		}
		response.Results = append(response.Results, result)
	}

	return response, nil
}

func GetDependencies(computation *api.TensorOperation) []int32 {
	operation := computation.GetOperation()
	switch operation := operation.(type) {

	case *api.TensorOperation_RmsNorm:
		source := operation.RmsNorm.GetSource()
		return []int32{source}

	default:
		panic(fmt.Sprintf("unsupported operation: %v", operation))
	}
}
