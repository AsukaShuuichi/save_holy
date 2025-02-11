print("Starting main...")
import torch
import torch.nn as nn
import fused_model
print("Import fuse_model successfully")
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision import transforms
from torch.utils.data import DataLoader
import build_dataset
from tqdm import tqdm

import torch.nn as nn
import torch
import timm
from vision_mamba.Vim.vim import models_mamba
#请你意识到一个问题，并不需要在这里定义ViTmodel因为本来就有
import timeit
import combine_model
from torch.cuda.amp import autocast, GradScaler
print("Import combine_model successfully")
'''
model_Vit_name="vit_base_patch16_224"
model_Vit=timm.create_model(model_Vit_name,pretrained=True,num_classes=0)

model_Vim=models_mamba.VisionMamba(patch_size=4,num_classes=22)
Vit_feature_extractor=fused_model.ViTFeatureExtractor(model_Vit,output_channels=3,vim_patches=model_Vim.patch_embed.num_patches,target_size=(224,224))
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
'''
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#以上是之前的model
model= combine_model.VisionMamba(in_channels=768, lsa_out=384, mdm_out=384)
#好吧，这里的token维度就是embedding的维度

data_root="/data/zyli/100_driver/Cam1_day/Cam1"

#此处用于定义路径
train_txt=os.path.join(data_root,"driver_train.txt")
val_txt=os.path.join(data_root,"driver_val.txt")
test_txt=os.path.join(data_root,"driver_test.txt")

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])




#这里用于实例化,用于定义访问逻辑，如何读取、加载和处理数据
train_dataset=build_dataset.CustomDataset(root=data_root,txt_file=train_txt,transform=trans)
val_dataset=build_dataset.CustomDataset(root=data_root,txt_file=train_txt,transform=trans)
test_dataset=build_dataset.CustomDataset(root=data_root,txt_file=test_txt,transform=trans)

#用于创建Dataloader，进行图片数据的加载
#为何这里需要打乱呢，是为了增加随机性防止过拟合，使得结果更具有代表性
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,pin_memory=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,pin_memory=True)

'''
model=fused_model.SerialModel(Vit_feature_extractor,model_Vim)
'''
#这里采用的是顺序连接
model=model.to(device)

#以下是超参数
EPOCHS=300
BATCH_SIZE=1
IN_CHANNELS=3
IMG_SIZE=224
PATCH_SIZE=4

EMBED_DIM=(PATCH_SIZE**2)*IN_CHANNELS

NUM_PATCH=(IMG_SIZE//PATCH_SIZE)**2

DROPOUT=0.001

NUM_HEAD=8
ACTIVATION="gelu"
NUM_ENCODER=768
NUM_CLASS=22

LEARNING_RATE=1e-7
ADAM_WEIGHT_DECAY=0
ADAM_BETAS=(0.9,0.999)
max_grad_norm = 1.0
torch.autograd.set_detect_anomaly(True)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),betas=ADAM_BETAS,lr=LEARNING_RATE,weight_decay=ADAM_WEIGHT_DECAY)

start=timeit.default_timer()


# 梯度累积参数
accumulation_steps = 16  # 每16个小批量累积一次梯度




# 初始化GradScaler
#说的此处gradscaler被划去是编译器的问题
#scaler = GradScaler()
#防止梯度消失
for epoch in range(EPOCHS):
    # 在训练代码最前面添加
    
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for idx, img_label in enumerate(tqdm(train_loader, position=0, leave=True)):

        img = img_label["image"].float().to(device,non_blocking=True)
        label = img_label["label"].long().to(device,non_blocking=True)
        if (label < 0).any() or (label >= NUM_CLASS).any():
            print(f"Label out of range! Min: {label.min()}, Max: {label.max()}")
            exit()  # 直接退出，防止错误传播
        #img=img.half()
        #禁用混合精度
        # 使用 autocast 来开启混合精度
        #with autocast():
            # 前向传播
        y_pred = model(img)
        
        y_pred_label = torch.argmax(y_pred, dim=1)

        # 计算损失
        if torch.isnan(y_pred).any():
            print("y_pred contains NaN!")
            exit()

        loss = criterion(y_pred, label)
        train_running_loss += loss.item()

        # 梯度缩放和反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)

        
        optimizer.step()  # 更新权重
       
        optimizer.zero_grad()  # 清除梯度

        torch.cuda.empty_cache()  # 清理缓存
        
    train_loss = train_running_loss / (idx + 1)

    # 在验证集上评估
    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0

    with torch.no_grad():
        for idx, img_label in enumerate(tqdm(val_loader, position=0, leave=True)):
            img = img_label["image"].float().to(device,non_blocking=True)
            label = img_label["label"].long().to(device,non_blocking=True)
            #img=img.half()
            # 前向传播
            y_pred = model(img)
            
            y_pred_label = torch.argmax(y_pred,dim=1)

            val_labels.extend(label.detach())  # 不进行转移
            val_preds.extend(y_pred_label.detach())

            # 计算损失
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()

            torch.cuda.empty_cache()

        val_loss = val_running_loss / (idx + 1)

        print("-" * 30)
        print(f"Train Loss Epoch {epoch + 1}: {train_loss:.4f}")
        print(f"Train Acc Epoch {epoch + 1}: {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print("-" * 30)

       
        






end_time=timeit.default_timer()
cost=start-end_time

print(f"Total time {cost}:.4f")