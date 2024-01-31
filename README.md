# GrainGNN: A dynamic heterogeneous graph neural network for large-scale 3D grain microstructure evolution.

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
python3 train.py --model_type=regressor --model_id=0 --device=cuda
```

multi-GPU training
```
python3 dist_train.py --model_type=regressor --model_id=0 --device=cuda
```

testing

```
python3 test.py --seed=0
```

## Cite

If you are using the codes in this repository, please cite the following paper
```
@misc{qin2024graingnn,
      title={GrainGNN: A dynamic graph neural network for predicting 3D grain microstructure}, 
      author={Yigong Qin and Stephen DeWitt and Balasubramanian Radhakrishnan and George Biros},
      year={2024},
      eprint={2401.03661},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```
