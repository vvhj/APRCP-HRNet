# New version code is underwriting. It will releas after testing on classify for Imagenet

# APRCP HRNet:  Adaptive Pruning Rate Channel Purning for HRNet Applied to 2D Human Pose Estimation 
The paper is in draft review. I hope the article will be hired. 

I don't know if there are any risks in open source code before employment, but I have promised to update the new results so the new result is released. 

I hope this work can help you and if you have any question or are interested in this direction you can join in the QQ group 767732179.

I hope to learn and progress with you.

The newst result has reach none accuracy drop with 58.2% Params pruned.

Some feature work is underwork. I will update and maintain in time, and welcome you to provide your own scheme for communication.

![Illustrating the architecture of the proposed HRNet](/figures/v2.PNG)

Fig1. the architecture of the proposed HRNet

![Illustrating the pruning area of the proposed HRNet](/figures/Prunarea.PNG)

Fig2. the pruning area of the proposed HRNet

### Old Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset 
| Arch               | Input size | #Params | GFLOPs |   ACC |    AP | Ap .5 | AP .75 |   AR   |
|--------------------|------------|---------|--------|-------|-------|-------|--------|--------|
|   pose_hrnet_w32   |    256x192 | 28.5M   |    7.1 | 0.883 | 0.765 | 0.935 |  0.837 |  0.841 |
|   pose_hrnet_w48   |    384x288 | 63.6M   |   32.9 | 0.887 | 0.781 | 0.936 |  0.849 |  0.860 |
|    **w32_best**    |    256x192 | 17.9M   |    4.4 | 0.882 | 0.763 | 0.936 |  0.837 |  0.841 |
|    **w48_best**    |   384x288  | 43.8M   |   21.0 | 0.888 | 0.781 | 0.936 |  0.849 |  0.859 |
|   **w32_extreme**  |    256x192 |  7.5M   |    2.2 | 0.863 | 0.732 | 0.926 |  0.813 |  0.809 |
|   **w48_extreme**  |   384x288  | 18.8M   |    9.8 | 0.885 | 0.775 | 0.935 |  0.847 |  0.853 |

### New Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
|      Model    |Criterion| r  | APRP | Params(PR) | GFLOPS(PR) |  AP  | AP .5|AP .75| AP M | AP L |  AR  |
|---------------|---------|----|------|------------|------------|------|------|------|------|------|------|
|HRNet-W32      |   ——    | —— |  ——  |   28.5m    |    7.1     | 76.5 | 93.5 | 83.7 | 73.9 | 80.8 | 79.3 |
|HRNet-W48      |   ——    | —— |  ——  |   63.6m    |    32.9    | 78.1 | 93.6 | 84.9 | 75.3 | 83.1 | 80.9 |
|APRC-HRNet-W48 |   v1    |0.36|Simple|45.9m(27.8%)|21.0(36.1%) | 78.1 | 93.6 | 84.9 | 75.3 | 83.1 | 80.7 |
|APRC-HRNet-W48 |   v1    |0.37|Golden|45.2m(28.9%)|21.0(36.1%) | 78.1 | 93.6 | 84.9 | 75.3 | 83.1 | 80.7 |
|APRC-HRNet-W48 |   v2    |0.58|Simple|28.7m(54.9%)|17.3(47.3\%)| 78.1 | 93.5 | 84.8 | 74.8 | 83.1 | 80.9 |
|APRC-HRNet-W48 |   v2    |0.61|Golden|26.6m(58.2%)|16.4(50.1\%)| 78.2 | 93.6 | 84.7 | 75.2 | 83.0 | 80.7 |
|APRC-HRNet-W48 |   v1    |0.78|Manual|19.7m(69.0%)| 9.8(70.3\%)| 77.5 | 93.5 | 84.7 | 74.3 | 82.2 | 80.0 |
|APRC-HRNet-W48 |   v2    |0.78|Manual|16.6m(73.9%)|11.7(64.5\%)| 77.7 | 93.5 | 84.7 | 74.5 | 82.2 | 80.2 |
### New Results on COCO test2017 
|      Model    |Criterion| r  | APRP | Params(PR) | GFLOPS(PR) |  AP  | AP .5|AP .75| AP M | AP L |  AR  |
|---------------|---------|----|------|------------|------------|------|------|------|------|------|------|
|HRNet-W32      |   ——    | —— |  ——  |   28.5m    |    7.1     | 74.9 | 92.5 | 82.8 | 71.3 | 80.9 | 80.1 |
|HRNet-W48      |   ——    | —— |  ——  |   63.6m    |    32.9    | 75.5 | 92.5 | 83.3 | 71.9 | 81.5 | 80.5 |
|APRC-HRNet-W48 |   v1    |0.36|Simple|45.9m(27.8%)|21.0(36.1%) | 75.2 | 92.5 | 83.0 | 71.6 | 81.2 | 80.4 |
|APRC-HRNet-W48 |   v1    |0.37|Golden|45.2m(28.9%)|21.0(36.1%) | 75.2 | 92.5 | 83.1 | 71.5 | 81.4 | 80.3 |
|APRC-HRNet-W48 |   v2    |0.58|Simple|28.7m(54.9%)|17.3(47.3\%)| 75.3 | 92.5 | 83.0 | 71.7 | 81.3 | 80.4 |
|APRC-HRNet-W48 |   v2    |0.61|Golden|26.6m(58.2%)|16.4(50.1\%)| 75.3 | 92.5 | 83.3 | 71.7 | 81.2 | 80.4 |
|APRC-HRNet-W48 |   v1    |0.78|Manual|19.7m(69.0%)| 9.8(70.3\%)| 74.6 | 92.4 | 82.4 | 71.0 | 80.6 | 79.8 |
|APRC-HRNet-W48 |   v2    |0.78|Manual|16.6m(73.9%)|11.7(64.5\%)| 74.6 | 92.2 | 82.4 | 71.0 | 80.6 | 79.7 |
### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- GFLOPs is for convolution and linear layers only.
- _best is best purning rate of HRnet and _extreme is higher purning rate.
- APRP is the selection mothed using to generate APR.

## Environment
The code is developed using python 3.6 on Centos7. NVIDIA GPUs are needed. The code is developed and tested using 2 NVIDIA 2080Ti GPU cards. 
## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   ```
   pip3 install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
6. Init log(tensorboard log directory) directory:

   ```
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── experiments
   ├── lib
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models of original HRNet from ([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth
   ```
For  APRCP HRNet you can get our prtrain model in : https://drive.google.com/file/d/1-EXl9dSatzmUSGpWGuBFlcPPM9T8Gcfr/view?usp=drivesdk
And new result using v1 and v2 is here: https://drive.google.com/file/d/16qW7gPrtjaQzyiuE9xEkkqBxaDYSvOoa/view?usp=sharing

For a purned model, there are two main file:
   ```
   pruneXXX.txt // to build model
   XXXXXXXX.pth // weight of model
   ```
We first use pruneXXX.txt to get model structure,then copy weight form XXXXXXXX.pth
   
### Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_data}, and make them look like this:
```
${POSE_data}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Purning select and Retraining
#### Purning select on COCO train2017 dataset
1. Edit config file. For example w48_384x288_adam_lr1e-3.yaml,
```
GPUS: (0,1)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
DATASET:
  COLOR_RGB: true
  DATASET: 'coco'
  DATA_FORMAT: jpg
  FLIP: true
  NUM_JOINTS_HALF_BODY: 8
  PROB_HALF_BODY: 0.3
  ROOT: '/root/work/datasets/coco'
  ROT_FACTOR: 45
  SCALE_FACTOR: 0.35
  TEST_SET: 'val2017'
  TRAIN_SET: 'train2017'

  PRETRAINED: 'models/pose_coco/pose_hrnet_w48_384x288.pth'

TEST:
  BATCH_SIZE_PER_GPU: 24
  COCO_BBOX_FILE: 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
  BBOX_THRE: 1.0
  IMAGE_THRE: 0.0
  IN_VIS_THRE: 0.2
  MODEL_FILE: 'models/pose_coco/pose_hrnet_w48_384x288.pth'
  NMS_THRE: 1.0
  OKS_THRE: 0.9
  USE_GT_BBOX: true
  FLIP_TEST: true
  POST_PROCESS: true
  SHIFT_HEATMAP: true

```

2. channel puring rate select
```
python3 tools/normal_regular_select \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output\
or use:
python3 tools/golden_cut_select.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output\
```
Note in line 138 : max_perf,max_acc = getpruneffects(0,"original")

getpruneffects should be replaced by getpruneffects_v2 or getpruneffects_v3 if you want to use v2 or v3 pruning mothed.

#### Retraining on COCO train2017 dataset

```
python3 tools/retrain.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output --percent [you get in purning select or another float in range(0,1)] \
```
or
```
python3 tools/retrain_v2.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output --percent [you get in purning select or another float in range(0,1)] \
```
or
```
python3 tools/retrain_v3.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output --percent [you get in purning select or another float in range(0,1)] \
```
#### test on COCO dataset
Modifiy experiments\coco\hrnet\w48_384x288_adam_lr1e-3_pt36.yaml
"MODEL_FILE" in  experiments\coco\hrnet\w48_384x288_adam_lr1e-3_pt36.yaml
```
python3 retraintest.py --ncfg [{scale or shift}{$r$}.txt]
```
([{scale or shift}{$r$}.txt] Corresponding to "MODEL_FILE" in  experiments\coco\hrnet\w48_384x288_adam_lr1e-3_pt36.yaml)

### Citation
Thanks follower work:
If you use our code or models in your research, please cite with:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {CoRR},
  volume    = {abs/1908.07919},
  year={2019}
}
```
