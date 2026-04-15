## Instructions:
## 1. conda activate my_env
## 2. make test

## Install local package
install:
	pip install -e .

## Run unit tests
test: 
	python3 -m unittest -v tests/*.py
