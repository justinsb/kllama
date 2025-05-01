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

	wantTensors := make([]TensorID, len(req.GetOutputTensors()))
	for i, id := range req.GetOutputTensors() {
		wantTensors[i] = TensorID(id)
	}
	if err := scope.Evaluate(wantTensors); err != nil {
		return nil, err
	}

	allTensors := scope.AllTensors()
	for _, outputTensorID := range req.GetOutputTensors() {
		tensor, found := allTensors[TensorID(outputTensorID)]
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

func GetDependencies(computation *api.TensorOperation) []TensorID {
	operation := computation.GetOperation()
	switch operation := operation.(type) {

	case *api.TensorOperation_RmsNorm:
		source := TensorID(operation.RmsNorm.GetSource())
		return []TensorID{source}

	case *api.TensorOperation_DotMultiply:
		return protoToTensorIDs(operation.DotMultiply.GetSources())

	case *api.TensorOperation_Add:
		return protoToTensorIDs(operation.Add.GetSources())

	default:
		panic(fmt.Sprintf("unsupported operation: %T %+v", operation, operation))
	}
}

func protoToTensorIDs(ids []int32) []TensorID {
	tensorIDs := make([]TensorID, len(ids))
	for i, id := range ids {
		tensorIDs[i] = TensorID(id)
	}
	return tensorIDs
}
