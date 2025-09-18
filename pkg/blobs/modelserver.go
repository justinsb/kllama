package blobs

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"k8s.io/klog/v2"
)

type ModelServer struct {
	// BlobserverURL is the base URL to the blobserver, typically http://blobserver
	BlobserverURL *url.URL
}

var _ BlobReader = &ModelServer{}

func (l *ModelServer) Download(ctx context.Context, info BlobInfo, destPath string) error {
	url := l.BlobserverURL.JoinPath(info.Hash)
	return l.downloadToFile(ctx, url.String(), destPath)
}

func (l *ModelServer) downloadToFile(ctx context.Context, url string, destPath string) error {
	log := klog.FromContext(ctx)

	dir := filepath.Dir(destPath)
	tempFile, err := os.CreateTemp(dir, "model")
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}

	shouldDeleteTempFile := true
	defer func() {
		if shouldDeleteTempFile {
			if err := os.Remove(tempFile.Name()); err != nil {
				log.Error(err, "removing temp file", "path", tempFile.Name)
			}
		}
	}()

	shouldCloseTempFile := true
	defer func() {
		if shouldCloseTempFile {
			if err := tempFile.Close(); err != nil {
				log.Error(err, "closing temp file", "path", tempFile.Name)
			}
		}
	}()

	if err := l.downloadToWriter(ctx, url, tempFile); err != nil {
		return fmt.Errorf("downloading from %q: %w", url, err)
	}

	if err := tempFile.Close(); err != nil {
		return fmt.Errorf("closing temp file: %w", err)
	}
	shouldCloseTempFile = false

	if err := os.Rename(tempFile.Name(), destPath); err != nil {
		return fmt.Errorf("renaming temp file: %w", err)
	}
	shouldDeleteTempFile = false

	return nil
}

func (l *ModelServer) downloadToWriter(ctx context.Context, url string, w io.Writer) error {
	log := klog.FromContext(ctx)

	log.Info("downloading from url", "url", url)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}

	startedAt := time.Now()

	httpClient := &http.Client{}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("doing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		if resp.StatusCode == 404 {
			return fmt.Errorf("blob not found: %w", os.ErrNotExist)
		}
		return fmt.Errorf("unexpected status downloading from upstream source: %v", resp.Status)
	}

	n, err := io.Copy(w, resp.Body)
	if err != nil {
		return fmt.Errorf("downloading from upstream source: %w", err)
	}

	log.Info("downloaded blob", "url", url, "bytes", n, "duration", time.Since(startedAt))

	return nil
}
