FROM golang:1.25.1 AS builder

ADD go.mod /workspace/
ADD go.sum /workspace/
ADD cmd/vllm-frontend /workspace/cmd/vllm-frontend/
ADD pkg /workspace/pkg/

WORKDIR /workspace

RUN CGO_ENABLED=0 go build -o /vllm-frontend ./cmd/vllm-frontend


FROM vllm/vllm-openai:v0.10.1.1

COPY --from=builder /vllm-frontend /vllm-frontend

ENTRYPOINT ["/vllm-frontend"]