## APCN
This is a code demo for the paper "Augmented Prototype Contrastive Network for Few-Shot Eye Disease Classification".

### Requirements
```
black>=20.8b1
isort>=5.10.1
jupyter>=1.0.0
loguru>=0.5.3
matplotlib>=3.3.4
mypy>=0.971
pandas>=1.2.1
pyarrow>=12.0.0
pylint>=2.17.2
pytest>=7.3.1
pytest-mock>=3.10.0
tensorboard>=2.8.0
torch>=1.9.0
torchvision>=0.10.0
tqdm>=4.56.0
typer>=0.9.0
```
### Datasets
- Dataset1：CUB
- Dataset2: K7000
- Datasaet: K8000 

You can download the source and target datasets mentioned above at  https://pan.baidu.com/s/1KdWr4jQQCVC8pv-6-x6ORA password: tewq, and move to folder `data`.  

An example datasets folder has the following structure:

```
data
├── CUB
│   ├── train
│   ├── test
│   ├── train.json
│   └── test.json
├── K7000
│   ├── train
│   ├── test
│   ├── train.json
│   └── test.json
└── K8000
    ├── train
    ├── test
    ├── train.json
    └── test.json
```
### QuickStart

1. Download the required datasets and move to folder `data`.

### 

### Acknowledgement
[easyfsl](https://github.com/sicara/easy-few-shot-learning)

[FSL-Mate](https://github.com/tata1661/FSL-Mate)