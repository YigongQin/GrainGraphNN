# GrainNN2: A dynamic heterogeneous graph neural network for large-scale 3D grain microstructure evolution.

## Build
use the local CUDA version

CUDA 10
```
export TORCH=1.11.0+cu102
export CUDA=cu102
```
CUDA 11
```
export TORCH=1.12.0+cu113
export CUDA=cu113
```

```
pip3 install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip3 install -r requirements.txt
```

## Usage
training
```
python3 grainNN2.py --model_type=regressor --model_id=0 --device=cuda
```

testing

```
python3 test.py --seed=0
```
