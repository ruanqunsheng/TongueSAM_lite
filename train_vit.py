from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import image_encoder
import torch
import torch.nn as nn
import numpy as np
from skimage import io, segmentation
import os
join = os.path.join 
from functools import partial
from torch.nn import functional as F
from torch.utils.data import Dataset
torch.manual_seed(2023)
from tqdm import tqdm
np.random.seed(2023)
class NpzDataset(Dataset): 
    def __init__(self, data_root):            
        self.npz_data=np.load(data_root)        
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):           
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        img=self.imgs[index]        
                 

        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float(),torch.tensor(img).float(),torch.tensor(random_points).float()
###############
checkpoint='./pretrained_model/sam.pth'
data_path='/home/disk/cs/project/dataset/transfer/tongueset1/'
device='cuda:0'
num_epochs = 30
model_save_path = './bottle_neck_nores/vit_lite_resbegin/'
batch_size=4
Init_lr= 1e-4
###############
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry['vit_b_lite'](checkpoint=checkpoint).to(device)  
model = sam_model.image_encoder.to(device)
sam_model = sam_model_registry['vit_b'](checkpoint=checkpoint).to(device)
model_truth = sam_model.image_encoder.to(device)
model=nn.DataParallel(model)
model_truth=nn.DataParallel(model_truth)
for name, param in model.named_parameters():    
    if 'resblock' not in name:      
        param.requires_grad = False
for name, param in model_truth.named_parameters():      
    param.requires_grad = False
with open(checkpoint, "rb") as f:
    state_dict = torch.load(f)
    model.load_state_dict(state_dict,strict=False)    
    model_truth.load_state_dict(state_dict,strict=False)
################    
# layer_name = "module.blocks.0.mlp.lin1.weight"
# if layer_name in model_truth.state_dict():
#     layer_params = model_truth.state_dict()[layer_name]
#     print(f"参数名称: {layer_name}")
#     print(f"参数值:\n{layer_params}")
#     layer_params = model.state_dict()[layer_name]
#     print(f"参数名称: {layer_name}")
#     print(f"参数值:\n{layer_params}")
# else:
#     print(f"模型中不存在名为 {layer_name} 的层")
###############
optimizer = torch.optim.Adam(model.parameters(), Init_lr,weight_decay=0,capturable=True)        
npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
###############
for epoch in range(num_epochs):    
    print(f'EPOCH: {epoch}')
    for npz_file in tqdm(npz_files):
        input_data = torch.tensor(np.load(os.path.join(data_path, npz_file))['img']).squeeze().to(device)                                
        i=0
        total_loss = 0.0 
        while i<32:
            input=input_data[i:i+batch_size]
            i+=batch_size
            output=model(input).to(device)
            target=model_truth(input).to(device)
            criterion = nn.MSELoss()
            loss = criterion(output, target).to(device)
            total_loss += loss.item()        
            optimizer.zero_grad()        
            loss.backward()        
            optimizer.step()       
        average_loss = total_loss / (32/batch_size)
                            
    torch.save(model.state_dict(), join(model_save_path, 'res11_blk1.pth'))# plot loss  
torch.save(model.state_dict(), join(model_save_path, 'res11_blk1_final.pth'))# plot loss  