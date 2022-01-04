import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype('uint8')

def save_image_gray(img_merge, filename):
    #image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, img_merge)
    

def merge_into_row(ele,  rgb_conf_final, semantic_conf_final, d_conf, rgb_depth_final,semantic_depth_final, d_depth, coarse_depth,pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    img_list1 = []



    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)

    if 'semantic' in ele:
        semantic = np.squeeze(ele['semantic'][0, ...].data.cpu().numpy())
        semantic = np.transpose(semantic, (1, 2, 0))
        img_list.append(semantic)

    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))

    
    if rgb_conf_final is not None:

        rgb_conf_final = mask_vis(rgb_conf_final[0, ...].data.cpu().numpy())

        rgb_conf_final = np.transpose(rgb_conf_final, (1, 2, 0))
        img_list1.append(rgb_conf_final)
        #img_list.append(preprocess_depth(semantic_conf_1[0, ...]))
    
    if semantic_conf_final is not None:
        semantic_conf_final = mask_vis(semantic_conf_final[0, ...].data.cpu().numpy())

        semantic_conf_final = np.transpose(semantic_conf_final, (1, 2, 0))
        img_list1.append(semantic_conf_final)
        #img_list.append(preprocess_depth(semantic_conf_1[0, ...]))    

    if d_conf is not None:
        d_conf = mask_vis(d_conf[0, ...].data.cpu().numpy())

        d_conf = np.transpose(d_conf, (1, 2, 0))
        img_list1.append(d_conf)
        #img_list.append(preprocess_depth(semantic_conf_1[0, ...])) 

    if rgb_depth_final is not None:

        #img_list.append(preprocess_depth(semantic_conf_1[0, ...]))

        img_list.append(preprocess_depth(rgb_depth_final[0, ...]))
        img_list.append(preprocess_depth(semantic_depth_final[0, ...]))
        img_list.append(preprocess_depth(d_depth[0, ...]))        
        img_list.append(preprocess_depth(coarse_depth[0, ...]))        
        img_list.append(preprocess_depth(pred[0, ...]))        


    if 'gt' in ele:
        
        #img_list.append(preprocess_depth(pred[0, ...]))
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge1 = np.hstack(img_list)
    img_merge1 = img_merge1.astype('uint8')

    img_merge2 = np.hstack(img_list1)
    img_merge2 = img_merge2.astype('uint8')

    return img_merge1, img_merge2


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename):
    #torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(sparse, gt, pred, filename):
    #from tensor
    #img = validcrop(img)
    img_list = []

    sparse = np.squeeze(sparse[0, ...]).data.cpu().numpy()
    sparse = depth_colorize(sparse)
    pred = np.squeeze(pred[0, ...]).data.cpu().numpy()
    pred = depth_colorize(pred)

    gt = np.squeeze(gt[0, ...]).data.cpu().numpy()
    gt = depth_colorize(gt)

    img_list.append(sparse)
    img_list.append(gt)
    img_list.append(pred)

    img_merge1 = np.hstack(img_list)
    img_merge1 = img_merge1.astype('uint8')

    img = cv2.cvtColor(img_merge1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
