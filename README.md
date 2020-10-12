# APRCP HRNet:  Adaptive Pruning Rate Channel Purning for HRNet Applied to 2D Human Pose Estimation \
The paper is underwriter，and some further work will put out when the paper finished. These work is now part of mine undergraduate graduation design. 

The newst result has reach none accuracy drop with 62.34% GFLOPs, the code will be updata recently. 

![Illustrating the architecture of the proposed HRNet](/figures/hrnet.png)
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | #Params | GFLOPs |   ACC |    AP | Ap .5 | AP .75 |   AR   |
|--------------------|------------|---------|--------|-------|-------|-------|--------|--------|
|   pose_hrnet_w32   |    256x192 | 28.5M   |    7.1 | 0.883 | 0.765 | 0.935 |  0.837 |  0.841 |
|   pose_hrnet_w48   |    384x288 | 63.6M   |   32.9 | 0.887 | 0.781 | 0.936 |  0.849 |  0.860 |
|    **w32_best**    |    256x192 | 17.9M   |    4.4 | 0.882 | 0.763 | 0.936 |  0.837 |  0.841 |
|    **w48_best**    |   384x288  | 43.8M   |   21.0 | 0.888 | 0.781 | 0.936 |  0.849 |  0.859 |
|   **w32_extreme**  |    256x192 |  7.5M   |    2.2 | 0.863 | 0.732 | 0.926 |  0.813 |  0.809 |
|   **w48_extreme**  |   384x288  | 18.8M   |    9.8 | 0.885 | 0.775 | 0.935 |  0.847 |  0.853 |

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- GFLOPs is for convolution and linear layers only.
- _best is best purning rate of HRnet and _extreme is higher purning rate

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

7. Download pretrained models of original HRNet from our zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
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
```

#### Retraining on COCO train2017 dataset

```
python3 tools/retrain.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output --percent [you get in purning select or another float in range(0,1)] \
```

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
