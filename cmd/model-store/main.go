package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/examples/AI/modelcloud/pkg/blobs"
	"k8s.io/klog/v2"
)

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	log := klog.FromContext(ctx)

	listen := ":8080"
	cacheDir := os.Getenv("CACHE_DIR")
	if cacheDir == "" {
		// We expect CACHE_DIR to be set when running on kubernetes, but default sensibly for local dev
		cacheDir = "~/.cache/blobserver/blobs"
	}
	flag.StringVar(&listen, "listen", listen, "listen address")
	flag.StringVar(&cacheDir, "cache-dir", cacheDir, "cache directory")
	flag.Parse()

	if strings.HasPrefix(cacheDir, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("getting home directory: %w", err)
		}
		cacheDir = filepath.Join(homeDir, strings.TrimPrefix(cacheDir, "~/"))
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return fmt.Errorf("creating cache directory %q: %w", cacheDir, err)
	}

	cacheBucket := os.Getenv("CACHE_BUCKET")
	if cacheBucket == "" {
		return fmt.Errorf("must specify CACHE_BUCKET env var")
	}

	var blobstore blobs.Blobstore

	if strings.HasPrefix(cacheBucket, "gs://") {
		cacheBucket = strings.TrimPrefix(cacheBucket, "gs://")
		log.Info("using GCS cache", "bucket", cacheBucket)

		blobstore = &blobs.GCSBlobstore{
			Bucket: cacheBucket,
		}
	} else {
		return fmt.Errorf("CACHE_BUCKET must be a GCS bucket URL (gs://<bucketName>)")
	}

	blobCache := &blobCache{
		BaseDir:   cacheDir,
		blobstore: blobstore,
	}

	s := &httpServer{
		blobCache: blobCache,
	}

	klog.Infof("serving on %q", listen)
	if err := http.ListenAndServe(listen, s); err != nil {
		return fmt.Errorf("serving on %q: %w", listen, err)
	}

	return nil
}

type httpServer struct {
	blobCache *blobCache
}

func (s *httpServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	tokens := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")
	if len(tokens) == 1 {
		if r.Method == "GET" {
			hash := tokens[0]
			s.serveGETBlob(w, r, hash)
			return
		}
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	http.Error(w, "not found", http.StatusNotFound)
}

func (s *httpServer) serveGETBlob(w http.ResponseWriter, r *http.Request, hash string) {
	ctx := r.Context()

	log := klog.FromContext(ctx)

	// TODO: Validate hash is hex, right length etc

	f, err := s.blobCache.GetBlob(ctx, hash)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		log.Error(err, "error getting blob")
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}
	defer f.Close()
	p := f.Name()

	klog.Infof("serving blob %q", p)
	http.ServeFile(w, r, p)
}

type blobCache struct {
	BaseDir   string
	blobstore blobs.Blobstore
}

func (c *blobCache) GetBlob(ctx context.Context, hash string) (*os.File, error) {
	// log := klog.FromContext(ctx)

	localPath := filepath.Join(c.BaseDir, hash)
	f, err := os.Open(localPath)
	if err == nil {
		return f, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("opening blob %q: %w", hash, err)
	}

	// TODO: We could try to download the blob here

	return nil, status.Errorf(codes.NotFound, "blob %q not found", hash)
}
