import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

# To some extent, Code is borrowed from the following URL 
#https://github.com/JUGGHM/PENet_ICRA2021/blob/main/dataloaders/kitti_loader.py


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        # transform = val_transform
        glob_d = os.path.join(
            args.data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )



        def get_rgb_paths(p):
            ps = p.split('/')
            date_liststr = []
            date_liststr.append(ps[-5][:10])
            # pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
            #                ps[-2:-1] + ['data'] + ps[-1:])
            pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
            pnew = os.path.join(args.data_folder_rgb, pnew)
            return pnew

        def get_semantic_paths(p):
            ps = p.split('/')
            date_liststr = []
            date_liststr.append(ps[-5][:10])
            # pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
            #                ps[-2:-1] + ['data'] + ps[-1:])
            pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
            pnew = os.path.join(args.data_semantic, pnew)
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )

            def get_rgb_paths(p):
                ps = p.split('/')
                date_liststr = []
                date_liststr.append(ps[-5][:10])
                # pnew = '/'.join(ps[:-7] +
                #   ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
                pnew = os.path.join(args.data_folder_rgb, pnew)
                return pnew

            def get_semantic_paths(p):
                ps = p.split('/')
                date_liststr = []
                date_liststr.append(ps[-5][:10])
                # pnew = '/'.join(ps[:-7] +
                #   ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
                pnew = os.path.join(args.data_semantic, pnew)
                return pnew


        elif args.val == "select":
            # transform = no_transform
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                "val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "val_selection_cropped/groundtruth_depth/*.png"
            )

            def get_rgb_paths(p):
                return p.replace("groundtruth_depth", "image")

            def get_semantic_paths(p):
                path = p.split("/")
                p_ = path[:-1]
                p_ = '/'.join(p_)
                p_ = p_.replace("groundtruth_depth", "semantic")
                p_ = p_ + "/" + path[-1].replace("groundtruth_depth", "image") 
                return p_
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "test_depth_completion_anonymous/image/*.png")

        glob_semantic = os.path.join(
            args.data_folder,
            "test_depth_completion_anonymous/semantic/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "data_depth_selection/test_depth_prediction_anonymous/image/*.png")
        glob_semantic = os.path.join(
            args.data_semantic,
            "depth/data_depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        paths_semantic = [get_semantic_paths(p) for p in paths_gt]

    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_semantic = sorted(glob.glob(glob_semantic))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_semantic):
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        print(len(paths_rgb), len(paths_d), len(paths_gt))
        # for i in range(999):
        #    print("#####")
        #    print(paths_rgb[i])
        #    print(paths_d[i])
        #    print(paths_gt[i])
        # raise (RuntimeError("Produced different sizes for datasets"))
    paths = {"rgb": paths_rgb,"semantic" : paths_semantic, "d": paths_d, "gt": paths_gt}
    
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def semantic_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename).convert("RGB")
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    semantic_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return semantic_png




def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, semantic,sparse, target, position, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = args.val_h
    owidth = args.val_w

    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transforms_list = [
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]


    transform_geometric = transforms.Compose(transforms_list)

    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    if semantic is not None:
        semantic = transform_geometric(semantic)
    # sparse = drop_depth_measurements(sparse, 0.9)

    if position is not None:
        bottom_crop_only = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
        position = bottom_crop_only(position)


    # random crop
    #if small_training == True:
    if args.not_random_crop == False:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[i:i + rheight, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[i:i + rheight, j:j + rwidth]

        if semantic is not None:
            if semantic.ndim == 3:
                semantic = semantic[i:i + rheight, j:j + rwidth, :]
            elif semantic.ndim == 2:
                semantic = semantic[i:i + rheight, j:j + rwidth]

        if sparse is not None:
            if sparse.ndim == 3:
                sparse = sparse[i:i + rheight, j:j + rwidth, :]
            elif sparse.ndim == 2:
                sparse = sparse[i:i + rheight, j:j + rwidth]

        if target is not None:
            if target.ndim == 3:
                target = target[i:i + rheight, j:j + rwidth, :]
            elif target.ndim == 2:
                target = target[i:i + rheight, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[i:i + rheight, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[i:i + rheight, j:j + rwidth]

    return rgb, semantic, sparse, target, position

def val_transform(rgb, semantic, sparse, target, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if semantic is not None:
        semantic = transform(semantic)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if position is not None:
        position = transform(position)

    return rgb, semantic, sparse, target, position


def no_transform(rgb, semantic, sparse, target, position, args):
    return rgb,semantic, sparse, target, position


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_near)

    return rgb_read(path_near)


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
       
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None

        semantic = semantic_read(self.paths['semantic'][index]) if \
            (self.paths['semantic'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None

        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        return rgb, semantic, sparse, target

    def __getitem__(self, index):
        rgb, semantic,sparse, target = self.__getraw__(index)


        rgb, semantic ,sparse, target, position = self.transform(rgb, semantic, sparse, target, None, self.args)

        rgb, gray = handle_gray(rgb, self.args)


        candidates = {"rgb": rgb, "semantic":semantic,  "d": sparse, "gt": target, \
                      "g": gray, 'position': position}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])
