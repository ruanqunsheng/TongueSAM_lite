from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import image_encoder
import torch
import torch.nn as nn
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import transform, io, segmentation
import os
join = os.path.join 
from functools import partial
from torch.nn import functional as F
from tqdm import tqdm

input_path='/home/cs/project/medsam_tongue/data/tongue_origin/img/'
batch_size=32
save_path='/home/disk1/cs/project/TongueSAM_lite/data/tongueset1/'
device='cuda:1'
def preprocess( x: torch.Tensor) -> torch.Tensor:        
    pixel_mean=torch.tensor([123.675, 116.28, 103.53]).reshape(1,3,1,1).to(device)
    pixel_std=torch.tensor([58.395, 57.12, 57.375]).reshape(1,3,1,1).to(device)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
num=0
image_data_list=[]
for f in tqdm(os.listdir(input_path)):    
    image_data = io.imread(join(input_path, f))
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
    image_data_pre[image_data == 0] = 0
    image_data_pre = transform.resize(image_data_pre, (400, 400), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
    image_data_pre = np.uint8(image_data_pre)
    sam_transform = ResizeLongestSide(1024)
    resize_img = sam_transform.apply_image(image_data_pre)
    image_data_list.append(resize_img)
    if len(image_data_list) == batch_size:             
        input_images = torch.as_tensor(np.array(image_data_list)).permute(0, 3, 1, 2).unsqueeze(dim=0).to(device)        
        input_images = preprocess(input_images)        
        image_data_list = []                                        
        np.savez_compressed(join(save_path, str(num)+'.npz'), img=input_images.cpu())
        num+=1        