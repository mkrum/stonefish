
IMAGE_NAME := "stonefish"

build:
	docker build -f Dockerfile -t {{IMAGE_NAME}}:cpu . --progress=plain

build-gpu:
	docker build -f Dockerfile.gpu -t {{IMAGE_NAME}}:gpu . --progress=plain

test:
	docker run -v $PWD:/stonefish {{IMAGE_NAME}}:cpu pytest -s

eval *args:
	docker run --rm -v "$PWD:/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace {{IMAGE_NAME}}:cpu python -m stonefish.eval {{args}}

local-train *args:
	docker run -it --rm -v "$PWD:/workspace" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace {{IMAGE_NAME}}:cpu python -m stonefish.train {{args}}
