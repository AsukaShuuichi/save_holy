import torch.nn as nn
import torch
import timm
from vision_mamba.Vim.vim import models_mamba
#请你意识到一个问题，并不需要在这里定义ViTmodel因为本来就有
import torch.nn.functional as F


class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model,output_channels,vim_patches,target_size=(224, 224)):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = vit_model
        self.target_size=target_size
        self.target_size = target_size  # Vision Mamba 的 H, W
        self.vim_patches=vim_patches
        self.project_dim = nn.Conv2d(
            in_channels=self.vit.embed_dim,  # ViT 输出维度 D
            out_channels=output_channels,  # Vision Mamba 的输入通道数 C，那不就是RGB的通道数3
            kernel_size=1  # 1x1 卷积降维
        )

    def forward(self, x):
        # ViT 特征提取 (B, D)
        features = self.vit.forward_features(x)  # ViT 输出 (B, D)
        print(f"Shape before interpolate: {features.shape}")
        if len(features.shape)==2:
            B,D=features.shape
            T=self.vim_patches
            H=W=self.target_size[0]
            #记住这里改了哈
            features_expanded=features.unsqueeze(2).unsqueeze(3)
            padding_height = max(0, H - features_expanded.shape[2])  # 填充高度
            padding_width = max(0, W - features_expanded.shape[3])   # 填充宽度
            features_padded = F.pad(features_expanded, (0, padding_width, 0, padding_height))
            features=features_padded.view(B,D,H,W)
        elif len(features.shape)==3:
            B,D,T=features.shape
            H=W=int(T**0.5)
        # 扩展空间维度 (B, D, H, W)
        # 投影到三通道
        features=self.project_dim(features)
        #调整目标分辨率
        features = nn.functional.interpolate(
            features, size=self.target_size, mode="bilinear", align_corners=False
        )  # (B, D, H, W)
        print(f"Shape after interpolate: {features.shape}")
        # 映射到 Vision Mamba 输入通道 (B, C, H, W)
        
        return features
    



class SerialModel(nn.Module):
    def __init__(self,vit_feature_extractor,vim_model):
        #串联模型的初始化
        super(SerialModel,self).__init__()
        self.feature_extractor=vit_feature_extractor
        self.vim_model=vim_model
        #自动进行子模型保存（将子模型作为一个个子模块）

    def forward(self,x):
        #ViT进行特征提取
        features=self.feature_extractor(x)
        #输入vision mamba
        output=self.vim_model(features)
        return output
    