# code_11438
## Contents
1. [Dependency](#dependency)
0. [Data](#data)
1. [Models](#models)
2. [Commands](#commands)


## Dependency
- We include a list of required packages in req.txt. We reccomend that you create a conda environment with all of the packages by running the following command.
```
conda create -n <environment-name> --file req.txt
```
Please chose environment-name of of your choice and replace it in the command.

Uptill now, we have tested our model on the following GPU's.
1. NVIDIA GTX RTX3090
2. NVIDIA A100
3. NVIDIA A100-80GB
4. NVIDIA RTXA6000



## Data
- Download the KITTI depth Validation and Test set from this [URL](https://drive.google.com/file/d/1vG2sxPF1_Zmvo0tnmUbvCLVAynp_v7N1/view?usp=sharing). Please unzip the dataset folder.
The overall data directory of the dataset is structured as follows:
```
├── data_depth_selection
|   ├── test_depth_completion_anonymous
|   |   |── image
|   |   |── intrinsics
|   |   |── semantic
|   |   |── velodyne_raw
|   |── test_depth_prediction_anonymous
|   |   |── image
|   |   |── intrinsics
|   |── val_selection_cropped
|   |   |── groundtruth_depth
|   |   |── image
|   |   |── intrinsics
|   |   |── semantics
|   |   |── velodyne_raw
```
## Models

Please download the pre-trained model from this [URL](https://drive.google.com/file/d/1plg4zGCLYndP0xtkh_gjG1RZ4YzPeiDN/view?usp=sharing).


## Commands

After you have downloaded the model and installed all of the required packages listed in req.txt, our results on KITTI validation dataset and test dataset can be validated 
by running the following commands.

### Validation Dataset
```
CUDA_VISIBLE_DEVICES="0"  python main.py -n sem_att -e [path of pre-trained model i.e. model_best_backup.pth.tar] --data-folder [path of data_depth_selection folder] --val_results "val_results/"
```
After successfull run, you can see the quantiative results summary in the table and the qualitative results can be viewed in the folder named "val_results/". Each image file inside the "val_results/" consists of Sparse, Groundtruth and Refined Depth stitched together. For example,
<div align=center><img src="https://github.com/cvpr2022-dc/code_11438/blob/main/images/val_demo.png" width = "100%" height = "100%" /></div>
The first image (from left) represents the LiDAR sparse depth, second represent LiDAR sparse ground truth map and third represent the output of SemAttNet. 


### Test Dataset
```
CUDA_VISIBLE_DEVICES="0"  python main.py -n sem_att -e [path of pre-trained model i.e. model_best_backup.pth.tar] --data-folder [path of data_depth_selection folder] --test-save "test_results/" --test
```
The results on test dataset are saved as 16bit depth maps in "test_results/" folder. The depths maps can be uploaded to [KITTI depth completion benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) to validate our claims on KITTI benchmark.


