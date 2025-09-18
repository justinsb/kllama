package blobs

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"cloud.google.com/go/storage"
	"k8s.io/klog/v2"
)

type GCSBlobstore struct {
	Bucket string
}

var _ Blobstore = (*GCSBlobstore)(nil)

func (j *GCSBlobstore) Upload(ctx context.Context, sourcePath string, info BlobInfo) error {
	log := klog.FromContext(ctx)

	src, err := os.Open(sourcePath)
	if err != nil {
		return fmt.Errorf("opening source file: %w", err)
	}
	defer src.Close()

	objectKey := info.Hash
	gcsURL := "gs://" + j.Bucket + "/" + objectKey

	client, err := storage.NewClient(ctx)
	if err != nil {
		return fmt.Errorf("creating GCS storage client: %w", err)
	}
	defer client.Close()

	obj := client.Bucket(j.Bucket).Object(objectKey)
	objAttrs, err := obj.Attrs(ctx)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			objAttrs = nil
			log.Info("object not found in GCS", "url", gcsURL)
			// Fallthrough to upload object
		} else {
			return fmt.Errorf("getting object attributes for %q: %w", gcsURL, err)
		}
	}
	if objAttrs != nil {
		log.Info("object already exists in GCS", "url", gcsURL)
		return nil
	}

	log.Info("uploading blob to GCS", "source", sourcePath, "destination", gcsURL)

	startedAt := time.Now()
	w := obj.NewWriter(ctx)
	n, err := io.Copy(w, src)
	if err != nil {
		return fmt.Errorf("uploading to GCS: %w", err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("closing GCS writer: %w", err)
	}

	log.Info("uploaded blob to GCS", "url", gcsURL, "bytes", n, "duration", time.Since(startedAt))

	return nil
}

func (j *GCSBlobstore) Download(ctx context.Context, info BlobInfo, destinationPath string) error {
	log := klog.FromContext(ctx)

	objectKey := info.Hash
	gcsURL := "gs://" + j.Bucket + "/" + objectKey

	client, err := storage.NewClient(ctx)
	if err != nil {
		return fmt.Errorf("creating GCS storage client: %w", err)
	}
	defer client.Close()

	// TODO: Do a head so we can output clearer log messages?

	log.Info("downloading blob from GCS", "source", gcsURL, "destination", destinationPath)

	startedAt := time.Now()
	r, err := client.Bucket(j.Bucket).Object(objectKey).NewReader(ctx)
	if err != nil {
		return fmt.Errorf("opening object from GCS %q: %w", gcsURL, err)
	}
	defer r.Close()

	n, err := writeToFile(ctx, r, destinationPath)
	if err != nil {
		return fmt.Errorf("downloading from GCS: %w", err)
	}

	log.Info("downloaded blob from GCS", "source", gcsURL, "destination", destinationPath, "bytes", n, "duration", time.Since(startedAt))

	return nil
}

func writeToFile(ctx context.Context, src io.Reader, destinationPath string) (int64, error) {
	log := klog.FromContext(ctx)

	dir := filepath.Dir(destinationPath)
	tempFile, err := os.CreateTemp(dir, "download")
	if err != nil {
		return 0, fmt.Errorf("creating temp file: %w", err)
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

	n, err := io.Copy(tempFile, src)
	if err != nil {
		return n, fmt.Errorf("downloading from upstream source: %w", err)
	}

	if err := tempFile.Close(); err != nil {
		return n, fmt.Errorf("closing temp file: %w", err)
	}
	shouldCloseTempFile = false

	if err := os.Rename(tempFile.Name(), destinationPath); err != nil {
		return n, fmt.Errorf("renaming temp file: %w", err)
	}
	shouldDeleteTempFile = false

	return n, nil
}
