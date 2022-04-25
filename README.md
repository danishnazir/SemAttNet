# train_code_11438

This guide tells us how to reproduce SemAttNet results. Overall, we utilize an incremental training approach. In the first step, we train our novel three-branch backbone. Then, in the second step, we utilize the pre-trained backbone for training CSPN++ with Atrous convolutions. The results of our model can be verified from the The results of our model can be verified from the [official KITTI leaderboard](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).


## Contents
1. Dependency
0. Data
0. Training Three-branch Backbone
0. Train SemAttNet

## Dependency
We include a list of required packages in req.txt. We reccomend that you create a conda environment with all of the packages by running the following command.
```
conda create -n <environment-name> --file req.txt
```

Please chose environment-name of of your choice and replace it in the command.

Uptill now, we have tested the training of our model on the following GPU's.

1. NVIDIA GTX RTX3090
0. NVIDIA A100
0. NVIDIA A100-80GB
0. NVIDIA RTXA6000

## Data

Please download KITTI Depth Completion Dataset from this [URL](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion). It is organized as follows.
```
├── kitti_depth
|   ├── depth
|   |   ├──data_depth_annotated
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_velodyne
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_selection
|   |   |  ├── test_depth_completion_anonymous
|   |   |  |── test_depth_prediction_anonymous
|   |   |  ├── val_selection_cropped
```
Please download RGB images i.e. KITTI Raw data from this [URL](http://www.cvlibs.net/datasets/kitti/raw_data.php). It is structured as follows.

```
├── kitti_raw
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```
 Please visit this [URL](https://drive.google.com/file/d/1Yq-vcIuu9USrpKYc9J6Svu19QiEZYf92/view?usp=sharing) to download the semanitc maps, which are also required to train our model. They are organized as follows
```
├── semantic_maps
|   |── depth
|   |   |── data_depth_selection
|   |   |   |── val_selection_cropped
|   |   |   |── test_depth_completion_anonymous
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```

## Training Three-branch Backbone

In the first step, we train our three-branch backbone consisting of color-guided (CG), semantic-guided (SG), and depth-guided (DG) branches. The command for training the three-branch backbone is given as follows.

```
python main.py -n bb -b 8 --data-folder [path of depth folder inside kitti_depth e.g. kitti_depth/depth/] --data-folder-rgb [path of rgb images e.g. kitti_raw/] --data-semantic [path of semantic maps e.g. semantic_maps/] --patience 4   
```
We train the backbone for 60 epochs and expect the convergence at **RMSE = 753.01**. A separate directory with training logs containing Val and train CSV files with respective RMSE values will be created in training_logs/ folder.


## Training SemAttNet

### Part 1

In the first part, we initialize the CSPN++ module by running the training for **5** epochs. Moreover, we freeze the parameters in the backbone as we don't want to update the parameters of the backbone.

```
python main.py -n sem_att -b 8 --data-folder [path of depth folder inside kitti_depth e.g. kitti_depth/depth/] --data-folder-rgb [path of rgb images e.g. kitti_raw/] --data-semantic [path of semantic maps e.g. semantic_maps/] --patience 8 -f --resume [path of pretrained three-branch backbone]
```


### Part 2

After initialzation , we can now fully train SemAttNet. We first train till **71<sup>st</sup>** epoch with **patience = 8** and then we restart the training by setting **patience = 3**. 
#### Patience = 8
```
python main.py -n sem_att -b 16 --lr 0.025298 --data-folder [path of depth folder inside kitti_depth e.g. kitti_depth/depth/] --data-folder-rgb [path of rgb images e.g. kitti_raw/] --data-semantic [path of semantic maps e.g. semantic_maps/] --patience 8 --resume [path of initialized CSPN++ model]
```

#### Patience = 3
```
python main.py -n sem_att -b 16 --lr 0.025298 --data-folder [path of depth folder inside kitti_depth e.g. kitti_depth/depth/] --data-folder-rgb [path of rgb images e.g. kitti_raw/] --data-semantic [path of semantic maps e.g. semantic_maps/] --patience 3 --resume [path of pre-trained SemAttNet model (till epoch = 71)]
```
We keep training till **95<sup>th</sup>** epoch to get the final fully trained model.


