# DGTR: Dual-Branch Graph Transformer Network for 3D Human Mesh Reconstruction from Video
## Introduction
This repository is the official [Pytorch](https://pytorch.org/) implementation of "Dual-Branch Graph Transformer Network for 3D Human Mesh Reconstruction from Video" 

## Video
<iframe height=360 width=640 src="./asset/IROS24_1464_VI_i.mp4">

## Running DGTR

## Installation
```bash
conda create -n DGTR python=3.7 -y
pip install torch==1.4.0 torchvision==0.5.0
pip install -r requirements.txt
```

### Data preparation
1. Download [base_data](https://drive.google.com/drive/folders/1PXWeHeo1e5gyXqLpEhIpatlNLd-8lpmc?usp=sharing) and SMPL pkl ([male&female](https://smpl.is.tue.mpg.de/) and [neutral](https://smplify.is.tue.mpg.de/)), and then put them into ${ROOT}/data/base_data/. Rename SMPL pkl as SMPL_{GENDER}.pkl format. For example, mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl.

2. Download [data](https://drive.google.com/drive/folders/1h0FxBGLqsxNvUL0J43WkTxp7WgYIBLy-?usp=sharing) provided by TCMR (except InstaVariety dataset). Pre-processed InstaVariety is uploaded by VIBE authors [here](https://owncloud.tuebingen.mpg.de/index.php/s/MKLnHtPjwn24y9C). Put them into ${ROOT}/data/preprocessed_data/

3. Download [models](https://drive.google.com/drive/folders/1PXWeHeo1e5gyXqLpEhIpatlNLd-8lpmc?usp=sharing) for testing. Put them into ${ROOT}/data/pretrained_models/

4. Download images (e.g., [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)) for rendering. Put them into ${ROOT}/data/3dpw/

The data directory structure should follow the below hierarchy.
```
${ROOT}  
|-- data  
  |-- base_data  
    |-- J_regressor_extra.npy  
    |-- ...
  |-- preprocessed_data
    |-- 3dpw_train_db.pt
    |-- ...
  |-- pretrained_models
    |-- table1_3dpw_weights.pth.tar
    |-- ...
  |-- 3dpw
    |-- imageFiles
      |-- courtyard_arguing_00
      |-- ...
```

