FROM golang:1.25.1 AS builder

ADD go.mod /workspace/
ADD go.sum /workspace/
ADD cmd/model-store /workspace/cmd/model-store/
ADD pkg /workspace/pkg/

WORKDIR /workspace

RUN CGO_ENABLED=0 go build -o /model-store ./cmd/model-store

# TODO: Don't have permission to write to volumes if nonroot
#FROM gcr.io/distroless/static-debian12:nonroot
FROM gcr.io/distroless/static-debian12:latest

COPY --from=builder /model-store /model-store

ENTRYPOINT ["/model-store"]