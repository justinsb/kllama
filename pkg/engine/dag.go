package engine

import "fmt"

func BuildDAG(scope Scope, wantTensors []TensorID) ([]TensorID, error) {
	allTensors := scope.AllTensors()

	evaluationOrder := make([]TensorID, 0, len(allTensors))
	done := make(map[TensorID]bool)

	for {
		progress := false
		for _, tensor := range allTensors {
			id := tensor.TensorID()
			if done[id] {
				continue
			}

			ready := true
			for _, dep := range tensor.Dependencies() {
				if !done[dep] {
					ready = false
					break
				}
			}
			if ready {
				done[id] = true
				evaluationOrder = append(evaluationOrder, id)
				progress = true
			}
		}
		if !progress {
			break
		}
	}

	for _, id := range wantTensors {
		if !done[id] {
			return nil, fmt.Errorf("tensor %d could not be computed (unreachable in computation graph)", id)
		}
	}

	// TODO: Only evaluate output tensors?

	return evaluationOrder, nil
}
