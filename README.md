## Effective Video Abnormal Event Detection by Learning A Consistency-Aware High-Level Feature Extractor

This repository is the official implementation of "Effective Video Abnormal Event Detection by Learning A Consistency-Aware High-Level Feature Extractor" submitted to ACM MM 2022 by anonymous author(s).

## 1. Requirements

(1) The basic running environment is as follows:

```
ubuntu 16.04
cuda 10.1
cudnn 7.6.4
python 3.7.1
pytorch 1.6.0
torchvision 0.7.0
numpy 1.20.2
opencv-contrib-python 4.1.1.26
```

## 2. Data preparation and preprocessing

(1) Download UCSDped1/ped2 from [official source](http://svcl.ucsd.edu/projects/anomaly/dataset.htm) and complete pixel-wise ground truth of UCSDped1 from [website](https://hci.iwr.uni-heidelberg.de/content/video-parsing-abnormality-detection), Avenue and Shanghaitech from [OneDrive](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F) or [BaiduNetdisk](https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ) (code: i9b3, provided by [StevenLiuWen](https://github.com/StevenLiuWen/ano_pred_cvpr2018)) , and [ground truth](www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip) of avenue from [official source](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html). Then create a folder named `raw_datasets` in root directory to store the downloaded datasets. The directory structure should be organized to match `vad_datasets.py` as follows: 

```
.
├── ...
├── raw_datasets
 │   ├── avenue
 │   │   ├── ground_truth_demo
 │   │   ├── testing
 │   │   └── training
 │   ├── ShanghaiTech
 │   │   ├── Testing
 │   │   ├── training
 │   │   └── training.zip
 │   ├── UCSDped2
 │   │   ├── Test
 │   │   └── Train
├── calc_optical_flow.py
├── ...
```

(2) To localize foreground objects, please follow the [instructions](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) to install mmdetection (v2.11.0). Then download the pretrained object detector [YOLOv3](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/configs/yolo/README.md), and move it to `fore_det/obj_det_checkpoints` .

(3) To utilize optical flow, please follow the [instructions](https://github.com/vt-vl-lab/flownet2.pytorch) to install FlowNet2, then download the pretrained model  [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing), and move the downloaded model `FlowNet2_checkpoint.pth.tar` into `./FlowNet2_src/pretrained`.  Finally run `calc_optical_flow.py` (in PyTorch 0.3.0): `python calc_optical_flow.py`. This will generate a new folder named `optical_flow` that contains the optical flow of the different datasets. 

## 3. Pre-training 

We follow this [repository](https://github.com/weiaicunzai/pytorch-cifar100) to perform pre-training of the teacher network. To yield the pre-trained teacher network, please run `train.py` in `./pre_training/pytorch-cifar100`: `python train.py -net resnet34`. We also provide the pre-trained teacher model used in our experiments at `./pre_training/pytorch-cifar100/checkpoint/resnet34/resnet34-200-regular.pth`.  You can skip this pre-training step and directly use the saved model for the following testing.

## 4.  Testing with saved models

(1) Edit the file `config.cfg` according to the model you want to test in  `./data ` : e.g. `dataset_name` (UCSDped2/avenue/ShanghaiTech), the distillation and consistency based anomaly score weight `w_raw` and `w_consistency ` (refer to the settings in our paper).  

(2) Run  `test.py`: `python test.py`.

## 5. Training

Please run `train.py`: `python train.py`. Before training, you can edit the file `config.cfg` according to your own requirements or implementation details reported in this paper.