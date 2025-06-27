---

## **Design Doc: A Kubernetes-Native LLM Serving Platform with StableHLO and IREE**

### **1. Overview**

This document outlines the high-level architecture for a dynamic, resource-aware LLM serving platform built on Kubernetes. The system is designed to be Kubernetes-native, using Custom Resource Definitions (CRDs) as its primary control-plane API.

The core concept is to treat model compilation as a managed, automated "ahead-of-time (AOT) compilation as a service" within the cluster. This decouples the slow, resource-intensive compilation process from the fast, real-time inference path. The system accepts LLMs packaged in the **StableHLO** format, compiles them for specific hardware targets using the **IREE** compiler, and serves them via auto-scaling Kubernetes Deployments.

### **2. Goals and Non-Goals**

#### **Goals**

*   **Kubernetes-Native API:** The entire system will be controlled via `kubectl` and Kubernetes manifests (CRDs). There will be no separate, external API.
*   **Hardware Portability:** Leverage StableHLO and IREE to support a heterogeneous cluster with various hardware types (CPU, NVIDIA GPUs, AMD GPUs, etc.) from a single model source.
*   **Automated Compilation:** The system will automatically detect the hardware available in the cluster and trigger compilation jobs to produce optimized serving artifacts.
*   **Efficient, Scalable Serving:** Use lightweight, compiled artifacts and Kubernetes' Horizontal Pod Autoscaler (HPA) to efficiently serve models and scale based on demand.
*   **Decoupled Architecture:** Separate the concerns of model management, compilation, and serving into distinct, manageable components.

#### **Non-Goals**

*   **Real-time JIT Compilation:** The design avoids full just-in-time compilation at pod startup due to the unacceptable latency it would introduce for serving. Runtime *specialization* of pre-compiled artifacts is handled by IREE, but not full AOT compilation.
*   **Model Training or Fine-Tuning:** This system is focused exclusively on the inference lifecycle.
*   **Support for other ML formats:** The initial design will standardize on StableHLO as the input format to ensure portability and stability.

### **3. Core Components & Architecture**

The system is composed of a central Kubernetes Operator that manages the lifecycle of several custom resources.



1.  **Kubernetes Operator:** The brain of the system. A single controller that watches the custom resources (`Model`, `CompiledModel`, `InferenceEndpoint`) and reconciles the state of the cluster to match the user's desired state.

2.  **Custom Resource Definitions (CRDs):** The user-facing API for the system.
    *   `Model`: Represents a source LLM in StableHLO format.
    *   `CompiledModel`: An internal resource, managed by the Operator, representing a `Model` compiled for a specific hardware target.
    *   `InferenceEndpoint`: Represents a request to serve a `Model` on specific hardware, making it available at a network endpoint.

3.  **Compilation Pipeline (Kubernetes Jobs):** The Operator launches Kubernetes `Jobs` to perform the AOT compilation. Each job runs the `iree-compile` tool, targeting a specific hardware architecture based on node labels in the cluster.

4.  **Artifact Storage (OCI Registry):** The system uses a standard OCI-compliant container registry for storing all artifacts:
    *   The initial StableHLO model.
    *   The final IREE-compiled serving images, which bundle the compiled `.vmfb` artifact with the `iree-runtime` and a lightweight web server.

5.  **Serving Runtimes (Deployments & Pods):** For each `InferenceEndpoint`, the Operator creates a Kubernetes `Deployment`. The pods in this deployment run the lightweight serving image produced by the compilation pipeline.

### **4. API Design (Custom Resources)**

#### **A. `Model`**

Defines the source of truth for an LLM.

```yaml
apiVersion: serving.kllama.ai/v1alpha1
kind: Model
metadata:
  name: llama3-8b-instruct
spec:
  # The source of the StableHLO model artifact.
  # The image should contain the .mlir file.
  source:
    oci:
      image: "my-registry/models/llama3-8b-instruct-stablehlo:latest"
status:
  conditions:
  - type: Ready
    status: "True"
    reason: SourceValidated
    message: "Model source OCI image found and is valid."
```

#### **B. `CompiledModel` (Operator-Managed)**

Represents the output of a compilation job. Users do not create this directly.

```yaml
apiVersion: serving.kllama.ai/v1alpha1
kind: CompiledModel
metadata:
  name: llama3-8b-instruct-nvidia-ampere
  ownerReferences:
  - apiVersion: serving.kllama.ai/v1alpha1
    kind: Model
    name: llama3-8b-instruct
spec:
  modelRef:
    name: llama3-8b-instruct
  # The hardware this model was compiled for.
  # This maps directly to node labels and IREE compiler flags.
  target:
    architecture: "nvidia-gpu"
    microarchitecture: "ampere" # e.g., ampere, hopper, rdna3
status:
  phase: "Succeeded" # Pending -> Compiling -> Succeeded / Failed
  # The location of the final, runnable serving image.
  artifact:
    oci:
      image: "my-registry/compiled-models/llama3-8b-instruct-nvidia-ampere:v1"
      digest: "sha256:..."
  compilationJob:
    name: "compile-llama3-8b-instruct-nvidia-ampere-xyz"
```

#### **C. `InferenceEndpoint`**

A user's request to deploy a model for serving.

```yaml
apiVersion: serving.kllama.ai/v1alpha1
kind: InferenceEndpoint
metadata:
  name: llama3-prod-chat
spec:
  # Reference to the model we want to serve.
  modelRef:
    name: llama3-8b-instruct

  # Define where and how to run the model.
  # The operator uses this to find the correct CompiledModel.
  target:
    architecture: "nvidia-gpu"
    microarchitecture: "ampere"

  # Autoscaling configuration for the serving pods.
  scaling:
    minReplicas: 1
    maxReplicas: 10
    metric:
      type: Resource
      resource:
        name: "nvidia.com/gpu"
        target:
          type: Utilization
          averageUtilization: 75
status:
  url: "http://llama3-prod-chat.default.svc.cluster.local"
  replicas: 3
  conditions:
  - type: Ready
    status: "True"
    reason: DeploymentReady
    message: "Inference endpoint is active and ready to serve requests."
```

### **5. Detailed Workflow**

1.  **Model Registration:** An MLOps engineer packages a StableHLO model into a container image and pushes it to the registry. They then apply a `Model` CRD to the cluster, pointing to that image.
2.  **Compilation Trigger:** The Operator detects the new `Model` resource. It scans the cluster nodes for hardware-identifying labels (e.g., `gpu-arch=ampere`). For each unique hardware target, it creates a `CompiledModel` resource with `status.phase: Compiling`.
3.  **AOT Compilation:** For each `CompiledModel`, the Operator launches a Kubernetes `Job`. This job:
    a. Pulls the source StableHLO image.
    b. Runs `iree-compile` with flags derived from the `target` spec (e.g., `--iree-hal-target-backends=cuda --iree-cuda-llvm-target-arch=sm_80`).
    c. Packages the resulting `.vmfb` artifact, the `iree-runtime`, and a simple FastAPI/gRPC server into a new container image.
    d. Pushes the new serving image to the registry.
4.  **Artifact Registration:** When the job succeeds, the Operator updates the corresponding `CompiledModel` status to `Succeeded` and populates the `status.artifact.oci.image` field with the path to the newly built serving image.
5.  **Serving Request:** An application developer, wanting to use the model, creates an `InferenceEndpoint` CRD. They specify the `modelRef` and the `target` hardware they want to run on.
6.  **Deployment:** The Operator sees the `InferenceEndpoint`. It finds the `CompiledModel` that matches the requested model and hardware target. If a `Succeeded` `CompiledModel` exists, the Operator creates:
    a. A **Deployment** using the serving image from the `CompiledModel`'s status. The Deployment's pod spec will include a `nodeSelector` to ensure it lands on the correct hardware.
    b. A **Service** to provide a stable network endpoint for the Deployment.
    c. A **HorizontalPodAutoscaler** configured according to the `scaling` parameters in the `InferenceEndpoint` spec.
7.  **Serving & Scaling:** The pods start, load the IREE module, and begin serving traffic. The HPA automatically scales the number of pods up or down based on the configured metrics.

### **6. Autoscaling Strategy**

Autoscaling is handled natively by the Kubernetes `HorizontalPodAutoscaler`. The `InferenceEndpoint` CRD provides a simplified interface for configuring it. The Operator will translate the `spec.scaling` section into a full HPA manifest.

This allows for scaling based on standard metrics like CPU and memory, as well as hardware-specific metrics (e.g., GPU utilization) provided by device plugins, or custom metrics (e.g., requests per second) via an adapter like the Prometheus Adapter.

### **7. Performance & Alternatives**

This architecture is a strategic choice prioritizing **portability, flexibility, and control** over chasing peak performance on a single vendor's hardware.

*   **vs. vLLM:** vLLM's primary advantage is its `PagedAttention` memory management, which can yield superior throughput in high-concurrency scenarios on GPUs. Our proposed design, using a general-purpose compiler, will not have this specialized scheduling logic out-of-the-box.
*   **vs. TensorRT-LLM:** For deployments on a homogenous NVIDIA cluster, TensorRT-LLM will likely provide the highest performance due to its use of hand-tuned, proprietary NVIDIA kernels.

The OpenXLA/IREE approach provides a "good enough" high-performance baseline that runs *everywhere*. It avoids vendor lock-in and future-proofs the platform. For ultimate performance, the JAX/Pallas "escape hatch" could be used to write custom, high-performance kernels for critical operations like attention, while still leveraging the overall framework.

This design is eminently feasible and provides a robust, cloud-native foundation for serving LLMs at scale in a dynamic hardware environment.
---
