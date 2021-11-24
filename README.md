# FGD
Paper: [Focal and Global Knowledge Distillation for Detectors](https://arxiv.org/abs/2111.11837)
## Install MMDetection and MS COCO2017
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmdet==2.11.0 and mmcv-full==1.2.4
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmdetectin's codes.
  - Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.
  - Replace the mmdet/apis/train.py and tools/train.py in mmdetection's codes with mmdet/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmdetection's codes.
  - Unzip COCO dataset into data/coco/
## Train

```
#single GPU
python tools/train.py configs/distillers/fgd/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py

#multi GPU
bash tools/dist_train.sh configs/distillers/fgd/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8
```

## Transfer
```
# Tansfer the FGD model into mmdet model
python pth_transfer.py --fgd_path $fgd_ckpt --output_path $new_mmdet_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt --eval bbox

#multi GPU
bash tools/dist_test.sh configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt 8 --eval bbox
```
## Results

|    Model    |  Backbone  | mAP  |                            config                            |                          weight                          | code |
| :---------: | :--------: | :--: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |
|  RetinaNet  | ResNet-50  | 40.7 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/1TwF9W13eHg6Sxkrr-4VTqg) | wsfw |
|  RetinaNet  | ResNet-101 | 41.7 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_2x_coco.py) |                                                          |      |
| Faster RCNN | ResNet-50  | 42.0 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/14WjoMqxILoPaKfY5QsCK8w) | dgpf |
| Faster RCNN | ResNet-101 | 44.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py) |                                                          |      |
|  RepPoints  | ResNet-50  | 42.0 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py) | [baidu](https://pan.baidu.com/s/1EJo9uQuZhimm7HI92TNThQ) | qx5d |
|  RepPoints  | ResNet-101 | 43.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r101_fpn_gn-neck+head_2x_coco.py) |                                                          |      |
|    FCOS     | ResNet-50  | 42.7 | [config](https://github.com/yzd-v/FGD/blob/master/configs/fcos/fcos_center-normbbox-giou_r50_caffe_fpn_gn-head_mstrain_1x_coco.py) | [baidu](https://pan.baidu.com/s/16uCTa81ZzG7EoizdfnXhzQ) | sedt |
|  MaskRCNN   | ResNet-50  | 42.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/101eOFcD8JDwqrFuYcxcBIA) | sv8m |



## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

Thanks to the work [GCNet](https://github.com/xvjiarui/GCNet) and [mmetection-distiller](https://github.com/pppppM/mmdetection-distiller).