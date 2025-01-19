import torch
from torch import nn
from mmdet3d.registry import MODELS
import os
import torch.nn.functional as F
import math
from typing import Optional, List
from torch import Tensor
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
 
# DETR 正余弦位置编码
@MODELS.register_module(name="PositionEmbeddingSine")    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        注意 num_pos_feats 应该是 图像通道数 hidden——dim的1/2
        temperature是温度系数
        # 温度系数越大，表示正弦和余弦函数的周期越长，幅度越小，这样可以使得相邻位置的特征向量更加相似。
        # 温度系数越小，表示正弦和余弦函数的周期越短，幅度越大，这样可以使得相邻位置的特征向量更加不同。[1] [2]
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(x.device)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# # 可学习的位置编码,还待验证
# @MODELS.register_module(name="PositionEmbeddingLearned")   
# class PositionEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """
#     def __init__(self, num_pos_feats=256):
#         super().__init__()
#         self.row_embed = nn.Embedding(50, num_pos_feats)
#         self.col_embed = nn.Embedding(50, num_pos_feats)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.uniform_(self.row_embed.weight)
#         nn.init.uniform_(self.col_embed.weight)

#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos



# 可学习的位置编码,还待验证
@MODELS.register_module(name="PositionEmbeddingLearned")
class PositionEmbeddingLearned(nn.Module):
    """
    绝对位置编码，学习型。
    """

    def __init__(self, num_pos_feats=256, max_height=496, max_width=432):
        super().__init__()
        # 调整嵌入层的大小以匹配特征图的最大宽度和高度
        self.row_embed = nn.Embedding(max_height, num_pos_feats)
        self.col_embed = nn.Embedding(max_width, num_pos_feats)
        # 调用reset_parameters方法来初始化嵌入权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用均匀分布初始化行嵌入和列嵌入的权重
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        # 从NestedTensor中提取张量
        x = tensor_list.tensors
        # 获取张量的高度和宽度
        h, w = x.shape[-2:]
        # 创建一个从0到w-1的整数序列，用于列嵌入
        i = torch.arange(w, device=x.device)
        # 创建一个从0到h-1的整数序列，用于行嵌入
        j = torch.arange(h, device=x.device)
        # 获取列嵌入和行嵌入
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # 将列嵌入和行嵌入合并，并调整形状以匹配输入张量x的批次大小和维度
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # 返回位置编码
        return pos








# def build_position_encoding(args):
#     N_steps = args.hidden_dim // 2
#     if args.position_embedding in ('v2', 'sine'):
#         # TODO find a better way of exposing other arguments
#         position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
#     elif args.position_embedding in ('v3', 'learned'):
#         position_embedding = PositionEmbeddingLearned(N_steps)
#     else:
#         raise ValueError(f"not supported {args.position_embedding}")

#     return position_embedding
def build_position_encoding(hidden_dim,pe_type):
    N_steps = hidden_dim // 2
    if pe_type in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif pe_type in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {pe_type}")

    return position_embedding



# 假设已经有了一个函数来生成可学习的位置编码
def generate_learnable_position_encoding(H, W, num_pos_feats, device):
    # 这里应该是生成可学习位置编码的代码
    # 返回位置编码张量
    pass


# bing推荐添加位置编码的流程
# # 提取图像特征和点云特征
# img_feats = ...  # 图像特征
# pts_feat = ...  # 点云特征

# # 为图像特征添加可学习的位置编码
# img_feats_pos = img_feats + generate_learnable_position_encoding(height, width, num_pos_feats, img_feats.device)

# # 为点云特征添加可学习的位置编码
# pts_feat_pos = pts_feat + generate_learnable_position_encoding(H, W, num_pos_feats, pts_feat.device)

# # 第一层交叉注意力：图像特征作为query，点云特征作为key，不带位置编码的点云特征作为value
# IP_feature = self.IP_cross_attention(
#     query=img_feats_pos,  # 带位置编码的图像特征
#     key=pts_feat_pos,     # 带位置编码的点云特征
#     value=pts_feat,       # 不带位置编码的点云特征
#     H=height,
#     W=width
# )

# # 融合融合后的图像特征和原始图像特征
# IP_feature2D = torch.concat((IP_feature, img_feats[0]), dim=1)

# # 压缩融合特征
# compressed_IP_feature2D = self.IP_compress_layer(IP_feature2D)

# # 由于compressed_IP_feature2D已经隐含了位置信息，我们在这里不再添加新的位置编码

# # 第二层交叉注意力：点云特征作为query，压缩后的图像-点云特征作为key，不带位置编码的压缩后的图像-点云特征作为value
# fusion_feature = self.PI_cross_attention(
#     query=pts_feat_pos,                      # 带位置编码的点云特征
#     key=compressed_IP_feature2D,             # 不带位置编码的压缩后的图像-点云特征
#     value=compressed_IP_feature2D,           # 不带位置编码的压缩后的图像-点云特征
#     H=H,
#     W=W
# )

# # 最终的融合特征将被用于后续的三维目标检测
