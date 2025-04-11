import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
from skimage import io
from  utils_metrics import *
from skimage import transform, io, segmentation
from segment.yolox import YOLOX
import random
import math
from functools import partial
##############################################################################################################
ts_npz_path='/home/disk/cs/project/dataset/segmentation/tongueset3_npz/test/'
model_type = 'vit_b'
checkpoint = '/home/disk/cs/project/TongueSAM_lite/pretrained_model/aug.pth'
device = 'cuda:0'
batch_size=32
prompt_type='box'
point_num=15
segment=YOLOX()
###############################################################################################################    
class NpzDataset(Dataset): 
    def __init__(self, data_root):            
        self.npz_data=np.load(data_root)
        self.ori_gts = self.npz_data['gts']
        self.img_embeddings = self.npz_data['img_embeddings']
        self.imgs=self.npz_data['imgs']
        self.model=segment        
        self.point_num=point_num
        
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):           
        # img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        img=self.imgs[index]
        

        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(img)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'                                                                    
        img_embed = sam_model.image_encoder(input_image)
# ############################box##############################################################        
        if self.model!=None:                        
            img=Image.fromarray(img)
            img= self.model.get_miou_png(img)                                                      
            y_indices, x_indices = np.where(img > 0)            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices) 
            bboxes = np.array([x_min, y_min, x_max, y_max])
            bboxes=np.array([x_min,y_min,x_max,y_max]) 
                          
            points=np.where(img > 0)                        
            random_points = random.choices(range(len(points[0])), k=self.point_num)            
            random_points = [(points[0][i], points[1][i]) for i in random_points]            
            
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)                   
            bboxes = np.array([x_min, y_min, x_max, y_max])  
            points=np.where(gt2D > 0)                        
            random_points = random.choices(range(len(points[0])), k=self.point_num)            
            random_points = [(points[0][i], points[1][i]) for i in random_points]              

        return img_embed.clone().detach().float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float(),torch.tensor(img).float(),torch.tensor(random_points).float()
#####################################################Begin############################################################################
train_losses = []
val_losses = []
best_iou=0
best_pa=0
best_acc=0


sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

epoch_loss = 0    
sam_model.eval()
val_gts=[]
val_preds=[]    
with torch.no_grad():                                                        
    for f in tqdm(os.listdir(ts_npz_path)):                                  
        ts_dataset = NpzDataset(join(ts_npz_path,f))            
        ts_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=True)
        for step, (image_embedding, gt2D, boxes,img,points) in enumerate(ts_dataloader):                        
            image_embedding=image_embedding.squeeze()                                                                                                                       
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)                                        
            box = sam_trans.apply_boxes(box_np, (img.shape[-2], img.shape[-1]))                                        
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]                                                           
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(                        
                points=None,
                boxes=box_torch,
                masks=None,
            )                       
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )                                                                                          
        for i in range(mask_predictions.shape[0]):
            mask = mask_predictions[i]
            mask = mask.cpu().detach().numpy().squeeze()
            mask = cv2.resize((mask > 0.5).astype(np.uint8),(gt2D.shape[2], gt2D.shape[3]))                                                      
            gt_data=gt2D[i].cpu().numpy().astype(np.uint8)                 
            val_gts.append(gt_data.astype(np.uint8))
            val_preds.append(mask.astype(np.uint8))                          
iou,pa,acc=compute_mIoU(val_gts,val_preds) 
        