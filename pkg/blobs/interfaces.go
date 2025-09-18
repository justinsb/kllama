package blobs

import "context"

type BlobReader interface {
	// If no such object exists, Download should return an error for which errors.Is(err, os.ErrNotExist) is true.
	Download(ctx context.Context, info BlobInfo, destPath string) error
}

type Blobstore interface {
	BlobReader
	// Upload uploads the file at sourcePath to the blobstore, using the given hash as the object key.
	// If an object with the same hash already exists, Upload should do nothing and return no error.
	Upload(ctx context.Context, sourcePath string, info BlobInfo) error
}

type BlobInfo struct {
	Hash string
}
