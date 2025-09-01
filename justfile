
set dotenv-load

# Configuration
IMAGE_NAME := "stonefish"
PROJECT_ID := env_var_or_default("PROJECT_ID", "geometric-notch-290722")
CLUSTER_NAME := env_var_or_default("CLUSTER_NAME", "stonefish-cluster")
CLUSTER_ZONE := env_var_or_default("CLUSTER_ZONE", "us-central1-a")

DOCKER_IMAGE_NAME := "gcr.io/" + PROJECT_ID + "/" + IMAGE_NAME + ":latest"

# Local development commands
build:
	@echo "=== Building locally ==="
	docker build -f Dockerfile -t  {{DOCKER_IMAGE_NAME}} . --progress=plain

build-local:
	@echo "=== Building Locally (buildx) ==="
	docker buildx build --load -f Dockerfile -t {{DOCKER_IMAGE_NAME}}  . --progress=plain
	@echo "Local image built: {{DOCKER_IMAGE_NAME}}"
	docker images {{DOCKER_IMAGE_NAME}}

test:
	@echo "=== Running Tests ==="
	docker run -v $PWD:/stonefish {{DOCKER_IMAGE_NAME}} pytest -s

benchmark *args:
	@echo "=== Benchmarking Model ==="
	docker run --rm -it -v "$PWD:/workspace" -w /workspace {{DOCKER_IMAGE_NAME}} python stonefish/benchmark.py {{args}}

eval *args:
	docker run --rm -v "$PWD:/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace {{DOCKER_IMAGE_NAME}} python -m stonefish.eval {{args}}

train-docker *args:
	docker run -it --rm -v "$PWD:/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace -e WANDB_API_KEY {{DOCKER_IMAGE_NAME}} torchrun --standalone --nnodes 1 --nproc_per_node=1 -m stonefish.train {{args}}

train-local *args:
	uv run --env-file .env torchrun --nnodes 1 --nproc_per_node=1 -m stonefish.train {{args}}

shell:
	@echo "=== Local Debug Shell ==="
	docker run -it --rm -v "$PWD:/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace -e WANDB_API_KEY {{DOCKER_IMAGE_NAME}} /bin/bash

# Kubernetes deployment commands
build-push:
	@echo "=== Building and pushing multi-platform Docker image ==="
	docker buildx create --name stonefishbuilder --use || true
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t gcr.io/{{PROJECT_ID}}/{{IMAGE_NAME}}:latest \
		--load \
		--push .
	@echo "Multi-platform image pushed to gcr.io/{{PROJECT_ID}}/{{IMAGE_NAME}}:latest"

# Kubernetes cluster management
create-cluster:
	@echo "=== Creating GKE cluster {{CLUSTER_NAME}} ==="
	gcloud container clusters create {{CLUSTER_NAME}} \
		--zone={{CLUSTER_ZONE}} \
		--num-nodes=1 \
		--machine-type=e2-standard-8 \
		--disk-size=100 \
		--disk-type=pd-standard \
		--enable-autoupgrade \
		--release-channel=stable \
		--quiet

# Create L4 GPU node pool
create-l4-pool:
	@echo "=== Creating L4 GPU node pool ==="
	gcloud container node-pools create l4-pool-4gpu \
		--cluster={{CLUSTER_NAME}} \
		--zone={{CLUSTER_ZONE}} \
		--machine-type=g2-standard-48 \
		--accelerator=type=nvidia-l4,count=4 \
		--num-nodes=1 \
		--spot \
		--disk-size=200GB \
		--disk-type=pd-ssd \
		--scopes=https://www.googleapis.com/auth/cloud-platform

# Apply Kubernetes resources
apply-storage:
	@echo "=== Creating storage ==="
	kubectl apply -f k8s/output-pvc.yaml
	kubectl get pvc

create-wandb-secret:
	@echo "=== Creating Weights & Biases secret ==="
	kubectl create secret generic wandb-secret --from-literal=api-key="${WANDB_API_KEY}" --dry-run=client -o yaml | kubectl apply -f -

train-deploy:
	@echo "=== Deploying training job ==="
	kubectl delete job stonefish-training --ignore-not-found
	kubectl apply -f k8s/train.yaml

benchmark-deploy *args:
	@echo "=== Benchmarking Model On k8s ==="
	kubectl apply -f k8s/benchmark.yaml

# Creates the
up: create-cluster create-l4-pool apply-storage create-wandb-secret

# Cleanup
cleanup:
	@echo "=== Cleaning up resources ==="
	kubectl delete job stonefish-training --ignore-not-found
	kubectl delete pvc stonefish-output-pvc --ignore-not-found
	gcloud container clusters delete {{CLUSTER_NAME}} --zone={{CLUSTER_ZONE}} --quiet

stop-train:
	@echo "=== Stopping Training Job ==="
	kubectl delete job stonefish-training

shell-deploy:
	@echo "=== Debug Shell ==="
	kubectl delete job stonefish-training --ignore-not-found
	kubectl apply -f k8s/shell.yaml
