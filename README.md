# CMCS-Temporal-Action-Localization

Code for 'Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization' (CVPR2019).

[Paper](http://www.vie.group/media/pdf/1273.pdf) and [Supplementary](http://www.vie.group/media/pdf/1273-supp.zip).

## Recommended Environment
* Python 3.5
* Cuda 9.0
* PyTorch 0.4

## Prerequisites
* Install dependencies: `pip3 install -r requirements.txt`.
* [Install Matlab API for Python](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (matlab.engine).
* Prepare THUMOS14 and ActivityNet datasets.

### Feature Extraction
We employ UntrimmedNet or I3D features in the paper. 

1. Download the features:
* [THUMOS14 Features](https://pan.baidu.com/s/1YT1KhKND5G_msZZ_zkWv9g)
* [ActivityNet Features](https://pan.baidu.com/s/1KJxNE-_We-9AcBs0N6xTXA) (Input videos are 25fps)
2. Join the zip files by `zip --fix {} --out {}` and unzip the files.
3. Put the extracted folder into the parent folder of this repo. (Or change the paths in the config file.)

If you want to extract features by yourself, please refer to these two repos:
* [UNT Features](https://github.com/wanglimin/UntrimmedNet)
* [I3D Features](https://github.com/Finspire13/pytorch-i3d-feature-extraction)

Other features can also be used.

### Generate Static Clip Masks:

Static clip masks are used for hard negative mining. They are included in the download features.
If you want to generate the masks by yourself, please refer to `tools/get_flow_intensity_anet.py`.

### Check ActivityNet Videos
URL links of some videos in this dataset are no longer valid. Check the availability and generate this file: [anet_missing_videos.npy](https://github.com/Finspire13/Weakly-Action-Detection/blob/Release-CVPR19/misc/anet_missing_videos.npy).

## Run

1. Train models with weak supervision (Skip this if you use our trained model):
```
python3 train.py --config-file {} --train-subset-name {} --test-subset-name {} --test-log
```

2. Test and save the class activation sequences (CAS):
```
python3 test.py --config-file {} --train-subset-name {} --test-subset-name {} --no-include-train
```

3. Action localization using the CAS:
```
python3 detect.py --config-file {} --train-subset-name {} --test-subset-name {} --no-include-train
```

For THUMOS14, predictions are saved in `output/predictions` and final performances are saved in a npz file in `output`.
For ActivityNet, predictions are saved in `output/predictions` and final performances can be obtained via the dataset evaluation API.

#### Settings
Our method is evaluated on THUMOS14 and ActivityNet with I3D or UNT features. Experiment settings and their auguments are listed as following. 

|   |           config-file          | train-subset-name | test-subset-name |
|---|:------------------------------:|:-----------------:|:----------------:|
| 1 |     configs/thumos-UNT.json    |        val        |       test       |
| 2 |     configs/thumos-I3D.json    |        val        |       test       |
| 3 |  configs/anet12-local-UNT.json |       train       |        val       |
| 4 |  configs/anet12-local-I3D.json |       train       |        val       |
| 5 |  configs/anet13-local-I3D.json |       train       |        val       |
| 6 | configs/anet13-server-I3D.json |       train       |       test       |


## Trained Models

Our trained models are provided [in this folder](https://github.com/Finspire13/Weakly-Action-Detection/tree/Release-CVPR19/models). To use these trained models, run `test.py` and `detect.py` with the config files [in this folder](https://github.com/Finspire13/Weakly-Action-Detection/tree/Release-CVPR19/configs/trained).

## Citation
@InProceedings{Liu_2019_CVPR,
author = {Liu, Daochang and Jiang, Tingting and Wang, Yizhou},
title = {Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}

## License
MIT

