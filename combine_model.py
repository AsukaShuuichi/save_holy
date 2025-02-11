import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    """深度卷积模块"""
    def __init__(self, in_channels, kernel_size, stride=1):
        super(DepthwiseConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels
        )

    def forward(self, x):
        return self.dw_conv(x)


class ScaledDotProductAttention(nn.Module):
    """标准化点积注意力"""
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class VSSM(nn.Module):
    """Vision State Space Model"""
    def __init__(self, input_dim, hidden_dim):
        super(VSSM, self).__init__()
        self.linear_a = nn.Linear(input_dim, hidden_dim)
        self.linear_b = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = F.silu(self.linear_a(x))
        y = self.linear_b(h)
        return y


class LSABranch(nn.Module):
    """LSA 分支"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LSABranch, self).__init__()
        self.depthwise_conv = DepthwiseConv(in_channels, kernel_size)
        self.linear = nn.Linear(in_channels, out_channels)
        self.attention = ScaledDotProductAttention(out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  
        q = k = v = self.linear(x)
        attn_out = self.attention(q, k, v)
        return attn_out.permute(0, 2, 1).view(b, c, h, w)  


class MDMBranch(nn.Module):
    """MDM 分支"""
    def __init__(self, in_channels, hidden_dim):
        super(MDMBranch, self).__init__()
        self.linear_a = nn.Linear(in_channels, hidden_dim)
        self.linear_b = nn.Linear(in_channels, hidden_dim)
        self.depthwise_conv = DepthwiseConv(hidden_dim, kernel_size=3)
        self.vssm = VSSM(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  
        branch_a = F.silu(self.linear_a(x))
        branch_b = self.linear_b(x)
        branch_b = branch_b.permute(0, 2, 1).view(b, c, h, w)  
        branch_b = F.silu(self.depthwise_conv(branch_b))
        branch_b = branch_b.view(b, c, -1).permute(0, 2, 1)  
        branch_b = self.vssm(branch_b)
        branch_b = self.layer_norm(branch_b)
        combined = branch_a + branch_b
        out = self.linear_out(combined)
        return out.view(b, c, h, w)


class VisionMamba(nn.Module):
    """完整模型"""
    def __init__(self, in_channels, lsa_out, mdm_out, num_classes=22):
        super(VisionMamba, self).__init__()
        self.channel_adjust = nn.Conv2d(3, in_channels, kernel_size=1)  
        self.batch_norm = nn.BatchNorm2d(in_channels)  

        # LSA 和 MDM 分支
        self.lsa_branch = LSABranch(in_channels // 2, lsa_out, kernel_size=3)
        self.mdm_branch = MDMBranch(in_channels // 2, mdm_out)
        self.num_classes = num_classes

        # 调整 LSA 和 MDM 输出的通道数，使其匹配 in_channels
        self.concat = nn.Conv2d(lsa_out + mdm_out, in_channels, kernel_size=1)
        
        # 自适应池化，将 H, W 归一化为 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接分类层
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.channel_adjust(x)  
        x = self.batch_norm(x)  

        # 通道分割
        x_lsa, x_mdm = torch.split(x, x.shape[1] // 2, dim=1)

        # 分支处理
        lsa_out = self.lsa_branch(x_lsa)
        mdm_out = self.mdm_branch(x_mdm)

        # 融合分支输出
        fused = torch.cat([lsa_out, mdm_out], dim=1)  
        output = self.concat(fused)  
        output = self.pool(output)  
        output = output.view(output.size(0), -1)  

        output = self.fc(output)
        return output
    
    
