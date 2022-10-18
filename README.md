# GrainNN2: A dynamic heterogeneous graph neural network for large-scale 3D grain microstructure evolution.

## Build

```
export TORCH=torch_version
export CUDA=cuda_version # eg. 113 for cuda/11.3
pip install -r requirements.txt
```

## Usage
training
```
python3 grainNN2.py train --mode=train
```

testing

```
python3 grainNN2.py test --mode=test
```
