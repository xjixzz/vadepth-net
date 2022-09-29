# VADepth


> **Visual Attention-based Self-supervised Absolute Depth Estimation using Geometric Priors in Autonomous Driving**

> [arxiv pdf ](https://arxiv.org/abs/2205.08780)
#### 1. Install
```
conda create -n vadepth python=3.7 -y
conda activate vadepth
conda install pytorch==1.10.1 torchvision==0.2.1 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -y
conda install opencv=3.4.2 pillow=8.4.0 -y
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Pillow==8.4.0 matplotlib==3.1.2 scikit-image==0.16.2 tqdm==4.57.0 tensorboardX==1.5 protobuf==3.19.1 timm==0.4.12
```

#### 2. Prepare kitti dataset
Following [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data)

#### 3. Train
download pretrained [VAN](https://github.com/Visual-Attention-Network/VAN-Classification)
```
mkdir pretrained
wget -P ./pretrained/ https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar
```

(1) train VADepth on kitti without absolute depth loss
```
CUDA_VISIBLE_DEVICES=${device} python train.py --data_path ${data_path} --log_dir ${log_dir} --model_name ${model_name}
```

(2) train VADepth on kitti with absolute depth loss
```
CUDA_VISIBLE_DEVICES=${device} python train.py --height_loss_weight 1e-2 --data_path ${data_path} --log_dir ${log_dir} --model_name ${model_name}
```

#### 4. Test on kitti
export gt depth of the eigen split
```
python export_gt_depth.py --data_path ${data_path} --split eigen 
```
(1) test on the eigen split of kitti with median scaling
```
CUDA_VISIBLE_DEVICES=${device} python van_evaluate_depth.py --data_path ${data_path} --load_weights_folder "${model_weight}" --eval_mono
```
The test results of our [model](https://drive.google.com/file/d/14T3hs2gDAmBeuQ8KNspwTBFgD8JW2D1M/view?usp=sharing)  trained without absolute depth loss are as follows:
| abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
|---------|--------|-------|----------|-------|-------|-------|
| 0.104   | 0.774  | 4.552 | 0.181    | 0.892 | 0.965 | 0.983 |

(2) test on the eigen split of kitti without median scaling
```
CUDA_VISIBLE_DEVICES=${device} python van_evaluate_depth.py --data_path ${data_path} --load_weights_folder "${model_weight}" --eval_mono --disable_median_scaling
```
The test results of our [model](https://drive.google.com/file/d/1dc1-lazvltqG3Xh-cyduOl4kk7X-IEU2/view?usp=sharing) trained with absolute depth loss are as follows:
| abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
|---------|--------|-------|----------|-------|-------|-------|
| 0.109   | 0.785  | 4.624 | 0.190    | 0.875 | 0.960 | 0.982 |




#### Acknowledgment
Our implementation is mainly based on [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data), [VAN](https://github.com/Visual-Attention-Network/VAN-Classification) and [DANet](https://github.com/junfu1115/DANet/). Thanks for their authors.
