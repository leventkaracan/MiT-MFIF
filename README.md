# Multi-image transformer for multi-focus image fusion (MiT-MFIF)

- [Introduction](#introduction)
- [Code Requirements](#requirements)
- [Datasets](#data-structure)
    - [Lytro Dataset](#lytro)
    - [MFFW Dataset](#mffw)
- [Training MiT-MFIF](#training)
- [Testing MiT-MFIF](#testing)
- [Publications](#publications)

## Introduction

MiT-MFIF is a multi-focus image fusion model to fuse input images, which have different depths are in-focus so that all-in-focus image can be obtained. With MiT-MFIF, we propose a new vision transformer architecture, called by Multi-image Transformer (MiT) to provide global connectivity across input images besides locality with well-grounded architecture.Our qualitative and quantitative evaluations demonstrate that MiT-MFIF  outperforms existing MFIF methods, predicting more accurate focus maps.


For more details, you can reach the [paper](https://www.sciencedirect.com/science/article/pii/S0923596523001406) 
## Installation

MiT-MFIF is coded with PyTorch

It requires the following installations:

```
python 3.8.3
pytorch (1.7.1)
cuda 11.1
```


## Training Data

Given a dataset root path in which there are folders containing input multi-focus images and corresponding all-in-focus images, you can train your own model.



## Test Datasets

### Lytro 


### MFFW 

 

#### Please send [us](mailto:levent.karacan@iste.edu.tr) a request e-mail to download dataset.


## Training MiT-MFIF

You can train MiT-MFIF using the following scripts:

`run_train.sh`

## Testing MiT-MFIF

You can test MiT-MFIF using test.sh script. You can reach the pre-trained model under the model directory.

`run_train.sh`

## Evaluation

## References

    """
    We build our FeedForward network on LocalViT to ensure locality in proposed Multi-image Transformer
    "Li, Y., Zhang, K., Cao, J., Timofte, R., & Van Gool, L. (2021). 
    Localvit: Bringing locality to vision transformers. 
    arXiv preprint arXiv:2104.05707."
    """

## Publications

```
 @article{karacan2023mitmfif,
title = {Multi-image transformer for multi-focus image fusion},
journal = {Signal Processing: Image Communication},
volume = {119},
pages = {117058},
year = {2023},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2023.117058},
url = {https://www.sciencedirect.com/science/article/pii/S0923596523001406},
author = {Levent Karacan}
}
```
