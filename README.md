# Multi-image transformer for multi-focus image fusion (MiT-MFIF)

## Introduction

MiT-MFIF, an  multi-focus image fusion model, fuses input images with varying depths of field into a comprehensive all-in-focu image. It introduces MiT, a novel vision transformer architecture, ensuring both local and  global connectivity. Remarkably, MiT-MFIF achieves MFIF without additional post-processing steps, streamlining the fusion process and boosting efficiency. Our evaluations, qualitative and quantitative, affirm its superior performance over existing methods, predicting more accurate focus maps.

For a comprehensive understanding and deeper insights, we invite you to explore the [paper](https://www.sciencedirect.com/science/article/pii/S0923596523001406).


## Installation

MiT-MFIF is coded with PyTorch.

It requires the following installations:

```
python 3.8.3
pytorch (1.7.1)
cuda 11.1
```


## Training Data

Given a dataset root path in which there are folders containing input multi-focus images and corresponding all-in-focus images, you can train your own model.

We follow the [MFIF-GAN](https://github.com/ycwang-libra/MFIF-GAN) to generate training data from [Pascal VOC12](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) dataset.

## Test Datasets

You may find the test data under the datasets folder. Please refer to the related papers if you use them in your research.

### [Lytro](https://github.com/xingchenzhang/MFIFB)
```M. Nejati, S. Samavi, S. Shirani, "Multi-focus Image Fusion Using Dictionary-Based Sparse Representation", Information Fusion, vol. 25, Sept. 2015, pp. 72-84. ```

### [MFFW](https://github.com/xingchenzhang/MFIFB)
```Xu, S., Wei, X., Zhang, C., Liu, J., & Zhang, J. (2020). MFFW: A new dataset for multi-focus image fusion. arXiv preprint arXiv:2002.04780.```

### [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU)

```Zhang, H., Le, Z., Shao, Z., Xu, H., & Ma, J. (2021). MFF-GAN: An unsupervised generative adversarial network with adaptive and gradient joint constraints for multi-focus image fusion. Information Fusion, 66, 40-53.```

 
## Training MiT-MFIF

You can train MiT-MFIF using the following script. 

`python main.py --root_traindata  ./mfif_dataset/  --model_save_dir ./models/  --model_name mfif`

## Testing MiT-MFIF

You can test MiT-MFIF using the following script. You can reach the pre-trained model under the "model" directory.

`python test.py --root_testdata  ./datasets --test_dataset LytroDataset --root_result ./results  --root_model ./models/ --model_name mit-mfif_best`

## Evaluation

To evaluate the MiT-MFIF, we utilize the following Matlab implementations.

 [https://github.com/zhengliu6699/imageFusionMetrics](https://github.com/zhengliu6699/imageFusionMetrics)
 
 [https://github.com/xytmhy/Evaluation-Metrics-for-Image-Fusion](https://github.com/xytmhy/Evaluation-Metrics-for-Image-Fusion)


## Implementation Notes

In our code, some code pieces are adapted from the [STTN](https://github.com/researchmm/STTN), [LocalViT](https://github.com/ofsoundof/LocalViT), and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

## Results

We have included the results for three datasets (Lytro, MFFW, MFI-WHU) in the "results" folder.

## Contact

Feel free to reach out to me with any questions regarding MiT-MFIF or to explore collaboration opportunities in solving diverse computer vision and image processing challenges. For additional details about my research please visit [my personal webpage](https://leventkaracan.github.io/).

## Citing MiT-MFIF

```
@article{karacan2023mitmfif,
title = {Multi-image transformer for multi-focus image fusion},
journal = {Signal Processing: Image Communication},
volume = {119},
pages = {117058},
year = {2023},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2023.117058},
author = {Levent Karacan}
}
```

