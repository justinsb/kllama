package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"

	api "github.com/justinsb/kllama/api/v1alpha1"
	"github.com/justinsb/kllama/pkg/engine"
	"github.com/justinsb/kllama/pkg/engine/fallback"
	"google.golang.org/grpc"
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
	listen := ":9876"

	klog.InitFlags(nil)
	flag.Parse()

	log := klog.FromContext(ctx)
	lis, err := net.Listen("tcp", listen)
	if err != nil {
		return fmt.Errorf("listening on %q: %w", listen, err)
	}
	var opts []grpc.ServerOption
	// if *tls {
	// 	if *certFile == "" {
	// 		*certFile = data.Path("x509/server_cert.pem")
	// 	}
	// 	if *keyFile == "" {
	// 		*keyFile = data.Path("x509/server_key.pem")
	// 	}
	// 	creds, err := credentials.NewServerTLSFromFile(*certFile, *keyFile)
	// 	if err != nil {
	// 		log.Fatalf("Failed to generate credentials: %v", err)
	// 	}
	// 	opts = []grpc.ServerOption{grpc.Creds(creds)}
	// }
	grpcServer := grpc.NewServer(opts...)

	calcServer := &CalcServer{}
	api.RegisterBigCalculatorServer(grpcServer, calcServer)
	log.Info("Starting tensorserver", "listen", listen)
	if err := grpcServer.Serve(lis); err != nil {
		return fmt.Errorf("serving GRPC: %w", err)
	}

	return nil
}

type CalcServer struct {
	api.UnimplementedBigCalculatorServer
}

func (s *CalcServer) Calculate(ctx context.Context, req *api.CalculateRequest) (*api.CalculateResponse, error) {
	scope, err := fallback.NewCalculationScope()
	if err != nil {
		return nil, err
	}
	defer scope.Close()

	response, err := engine.Evaluate(scope, req)
	if err != nil {
		return nil, err
	}

	return response, nil
}
