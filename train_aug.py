import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import torch.nn as nn
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
import random
import math
from functools import partial
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
from dataloader import *
from segment.yolox import YOLOX
##############################################################################################################
num_epochs = 100
ts_npz_path='/home/disk1/cs/project/dataset/VOC2012/npz/test/'
npz_tr_path = '/home/disk1/cs/project/dataset/VOC2012/npz/train/'
model_type = 'vit_b'
checkpoint = '/home/disk1/cs/project/SAM/pretrained_model/sam.pth'
model_save_path='./logs/'
device = 'cuda:0'
if_save=True
if_freeze=True
if_onlytest=False
batch_size=4
lr_decay_type= "cos"
Init_lr= 5e-5
segment=YOLOX()
###############################################################################################################    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
#####################################################Begin############################################################################
Min_lr=Init_lr*0.01
lr_limit_max    = Init_lr 
lr_limit_min    = 3e-4 
Init_lr_fit     = min(max(batch_size / batch_size * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit      = min(max(batch_size / batch_size * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, num_epochs)
train_losses = []
val_losses = []
best_iou=0
best_pa=0
best_acc=0

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
vit=nn.DataParallel(sam_model.image_encoder).to(device)
seg_loss = monai.losses.DiceCELoss(sigmoid=False, squared_pred=True, reduction='mean')#%% train

os.makedirs(model_save_path, exist_ok=True)
##############################################################################
with open('/home/disk1/cs/project/segmentation/dataset/tongue_all/VOC2007/ImageSets/Segmentation/train.txt') as f:
        train_lines = f.readlines()
with open('/home/disk1/cs/project/segmentation/dataset/tongueset3/VOC2007/ImageSets/Segmentation/test.txt') as f:
    val_lines = f.readlines()
input_shape = [400, 400]
train_dataset   = SAMDataset(train_lines, input_shape, 2, True)
val_dataset     = SAMDataset(val_lines, input_shape, 2, False)
gen= DataLoader(train_dataset, shuffle = True, batch_size = batch_size, collate_fn = SAM_dataset_collate)                                    
gen_val= DataLoader(val_dataset  , shuffle = True, batch_size = batch_size,collate_fn = SAM_dataset_collate)                                    
                            
##############################################################################
for epoch in range(num_epochs):  
    print(f'EPOCH: {epoch}')   
    train_loss = 0    
    val_loss = 0    
###############################################################Test##################################################################
    sam_model.eval()
    val_gts=[]
    val_preds=[]    
    with torch.no_grad():                     
        for iteration, batch in tqdm(enumerate(gen_val)):              
            imgs, pngs, labels = batch
            lower_bound, upper_bound = np.percentile(imgs, 0.5), np.percentile(imgs, 99.5)
            image_data_pre = np.clip(imgs, lower_bound, upper_bound)            
            imgs=F.interpolate(image_data_pre, size=(1024, 1024), mode='bilinear', align_corners=False)
            imgs=imgs.to(device)
            pngs=pngs.to(device)
            labels=labels.to(device)

            prompt=torch.zeros([8,4])
            for i in range(imgs.shape[0]):
                if segment!=None:
                    img=np.transpose(np.array(imgs[i].cpu()), (1, 2, 0))
                    boxes= YOLOX().get_prompt(img)                                                                 
                    if boxes is not None:
                        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)                   
                        box = sam_trans.apply_boxes(boxes, (400,400))                                                
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)            
                    else:                                
                        box_torch = None
                    
                else:
                    y_indices, x_indices = np.where(gt_data[i] > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)                   
                    boxes = np.array([x_min, y_min, x_max, y_max])            
                    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)                                        
                    box = sam_trans.apply_boxes(boxes, (1024,1024))                                        
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    if len(box_torch.shape) == 2:
                        box_torch = box_torch[:, None, :]
                print(box_torch)
                if box_torch!= None:                    
                    prompt[i]=box_torch
                else:
                    prompt=None
                    break                
            print(prompt)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            
                                                
            input_image = sam_model.preprocess(imgs) 
                        
            image_embedding = vit(input_image)
            
            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=True,
                )   
            
            mask = F.interpolate(mask_predictions, size=(500, 500), mode='bilinear', align_corners=False)
            val_loss += seg_loss(mask, labels)            
            for i in range(mask_predictions.shape[0]): 
                mask = F.interpolate(mask_predictions[i].unsqueeze(dim=0), size=(500, 500), mode='bilinear', align_corners=False).squeeze()
                gt_data = labels[i].squeeze().to(device)                
                mask = torch.argmax(mask, dim=0)                
                gt = torch.argmax(gt_data, dim=0)
                #########################
                gt[gt_data[21] == 1] = mask[gt_data[21] == 1]                                                                
                #########################
                gt_data=gt                
                if m == 1:
                    mask_img = (mask * 255/20).cpu().numpy().astype(np.uint8)
                    cv2.imwrite(model_save_path + str(i) + '.png', mask_img)                                        
                    gt_img = (gt.cpu()*255/20).numpy().astype(np.uint8)
                    cv2.imwrite(model_save_path + str(i) + '_gt.png', gt_img)                            
                val_gts.append(gt.cpu().numpy().astype(np.uint8))
                val_preds.append(mask.cpu().numpy().astype(np.uint8))                                                                                         
            m=0
            
        val_losses.append(val_loss.cpu())
        iou,pa,acc=compute_mIoU(val_gts,val_preds) 
        
        if  iou> best_iou:
            best_iou=iou            
            best_pa=pa
            best_acc=acc        
            if if_onlytest:
                continue
            if if_save==True:
                torch.save(sam_model.state_dict(), join(model_save_path, 'best_'+str(epoch)+'.pth'))# plot loss                  
        print('best_miou:'+str(best_iou))
        print('best_pa:'+str(best_pa))
        print('best_acc:'+str(best_acc))
        if if_onlytest:
                continue
###############################################################Train##################################################################
    sam_model.train()
    lr = lr_scheduler_func(epoch)    
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr,weight_decay=0)
    for iteration, batch in tqdm(enumerate(gen)):
        with torch.no_grad():  
            imgs, pngs, labels = batch
            lower_bound, upper_bound = np.percentile(imgs, 0.5), np.percentile(imgs, 99.5)
            image_data_pre = np.clip(imgs, lower_bound, upper_bound)            
            imgs=F.interpolate(image_data_pre, size=(1024, 1024), mode='bilinear', align_corners=False)
            imgs=imgs.to(device)
            pngs=pngs.to(device)
            labels=labels.to(device)
            input_image = sam_model.preprocess(imgs)                         
            image_embedding = vit(input_image)
        if if_freeze==True:
            with torch.no_grad():                    
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )     
        else:                                 
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )               
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
            )            

        mask_predictions=(F.interpolate(mask_predictions, size=(500,500), mode='bilinear', align_corners=False)).squeeze()
        gt2D=mask_predictions.squeeze().to(device)
        # i,j,k= torch.where((gt2D[:, 21, :, :] == 1))                
        # gt2D=gt2D[:,:21,:,:]            
        # gt2D[i,:,j,k]=mask_predictions[i,:,j,k]
        
        # mask_predictions=mask_predictions[:,1:,:,:]
        # gt2D=gt2D[:,1:,:,:]
        
        
        loss = seg_loss(mask_predictions, gt2D)  
        train_loss+=loss      
        optimizer.zero_grad()        
        loss.backward()        
        optimizer.step()
    train_losses.append(train_loss.detach().cpu())    
################################################################################################################################  
    if if_onlytest is False:
        plt.plot(train_losses)
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('train_loss')
        plt.show() 
        plt.savefig(join(model_save_path, 'train_loss.png'))
        plt.close()
        plt.plot(val_losses)
        plt.title('Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('val_loss')
        plt.show() 
        plt.savefig(join(model_save_path, 'val_loss.png'))
        plt.close()
        