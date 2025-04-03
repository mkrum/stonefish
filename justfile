
IMAGE_NAME := "stonefish"

build:
	docker build -f Dockerfile -t {{IMAGE_NAME}}:cpu . --progress=plain

build-gpu:
	docker build -f Dockerfile.gpu -t {{IMAGE_NAME}}:gpu . --progress=plain

test:
	docker run {{IMAGE_NAME}}:cpu pytest -s

# Add dependency with Poetry
add package:
	poetry add {{package}}
