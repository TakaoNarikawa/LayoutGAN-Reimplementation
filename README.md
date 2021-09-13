LayoutGAN Reimplementation
====

PyTorch reimplementation of "LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators" publishsed in ICLR 2019: https://openreview.net/forum?id=HJxB5sRcFQ.

## Requirement

- PyTorch : 1.8.1
- TorchVision : 0.9.1
- PyTorch-Lightning : 1.3.4

## Getting Started

### Point Layout (using MNIST dataset)

1. Download `pre_data_cls.npy` from [Link](https://drive.google.com/file/d/1R1iRZxADR_RcDsuR4gyStyLAo7i5LRAH/view?usp=sharing).
  This is from [Official Tensorflow Implementation Repository](https://github.com/JiananLi2016/LayoutGAN-Tensorflow)
2. Run `python3 train.py`. Use `--gpus 1` option for GPU.

### BBox Layout (using PubLayNet dataset)

1. Download `labels.tar.gz` from [PubLayNet Official](https://developer.ibm.com/exchanges/data/all/publaynet/) and decompress it as below.

```
PubLayNet
├ train.json
├ val.json
└ preprocess.py
```

2. Run `python3 preprocess.py` in `/PubLayNet`. Then you will have `train.npy` and `val.npy`
3. Run `python3 train.py --train_mode bbox`. Use `--gpus 1` option for GPU.


## Results

### Point Layout


<img src="https://github.com/TakaoNarikawa/LayoutGAN-Reimplementation/blob/main/screenshots/mnist.gif?raw=true" width=250px height=250px /> <img src="https://github.com/TakaoNarikawa/LayoutGAN-Reimplementation/blob/main/screenshots/mnist_30.png" width=250px height=250px />

### BBox Layout

<img src="https://github.com/TakaoNarikawa/LayoutGAN-Reimplementation/blob/main/screenshots/publaynet.gif?raw=true" width=250px height=250px /> <img src="https://github.com/TakaoNarikawa/LayoutGAN-Reimplementation/blob/main/screenshots/publaynet_300.png" width=250px height=250px />

