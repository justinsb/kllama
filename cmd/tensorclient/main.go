package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
)

func main() {
	ctx := context.Background()
	err := run(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {

	serverAddr := "127.0.0.1:9876"

	klog.InitFlags(nil)
	flag.Parse()

	log := klog.FromContext(ctx)

	var opts []grpc.DialOption
	if false {
		// *tls {
		// if *caFile == "" {
		// 	*caFile = data.Path("x509/ca_cert.pem")
		// }
		// creds, err := credentials.NewClientTLSFromFile(*caFile, *serverHostOverride)
		// if err != nil {
		// 	log.Fatalf("Failed to create TLS credentials: %v", err)
		// }
		// opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	conn, err := grpc.NewClient(serverAddr, opts...)
	if err != nil {
		return fmt.Errorf("failed to connect to server %q: %w", serverAddr, err)
	}
	defer conn.Close()
	client := api.NewBigCalculatorClient(conn)

	log.Info("Starting tensorclient", "server", serverAddr)

	// request := &api.CalculateRequest{
	// 	Tensors: []*api.Tensor{
	// 		{Id: 1, InlineData: &api.InlineData{Values: []float32{1, 2, 3}}},
	// 		{Id: 2, Computation: &api.TensorOperation{Operation: &api.TensorOperation_LinearScale{LinearScale: &api.LinearScale{Source: 1, Scale: 2}}}},
	// 	},
	// 	OutputTensors: []int32{2},
	// }

	request := &api.CalculateRequest{
		Tensors: []*api.Tensor{
			{Id: 1, InlineData: &api.InlineData{Values: []float32{1, 2, 3}}},
			{Id: 2, Computation: &api.TensorOperation{Operation: &api.TensorOperation_RmsNorm{RmsNorm: &api.RMSNorm{Source: 1}}}},
		},
		OutputTensors: []int32{2},
	}
	response, err := client.Calculate(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to calculate: %w", err)
	}
	log.Info("Response", "response", response)

	return nil
}
