import torch
from torch import nn
from torch.nn import functional as f
from mmdet3d.registry import MODELS


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.key_channels = key_channels  # 键向量的通道数
        self.head_count = head_count  # 多头注意力机制中头的数量
        self.value_channels = value_channels  # 值向量的通道数

        # 定义键、查询和值的卷积层（1x1卷积）
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        # 重投影层，用于将多头合并后的结果投影回输入通道数
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()  # 获取输入的批次数n和高h宽w
        # 将输入通过键、查询、值的卷积层，并调整形状以便进行矩阵乘法
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        # 计算每个头的键、查询和值的维度
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            # 对每个头，单独计算softmax归一化后的键和查询
            key = f.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = f.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            # 计算上下文向量：先通过键和值的矩阵乘法，然后与查询进行矩阵乘法
            context = key @ value.transpose(1, 2)  # 转置以匹配维度
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            # 将当前头计算的注意力结果添加到列表中
            attended_values.append(attended_value)

        # 将所有头计算的结果在通道维度上合并
        aggregated_values = torch.cat(attended_values, dim=1)
        # 通过重投影层将合并后的结果投影回原始的通道数
        reprojected_value = self.reprojection(aggregated_values)
        # 将注意力模块的输出与原始输入相加，实现残差连接
        attention = reprojected_value + input_

        return attention

@MODELS.register_module(name="CrossEfficientAttention")
class CrossEfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.key_channels = key_channels  # 键向量的通道数
        self.head_count = head_count  # 多头注意力机制中头的数量
        self.value_channels = value_channels  # 值向量的通道数

        # 定义键、查询和值的卷积层（1x1卷积）
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        # 重投影层，用于将多头合并后的结果投影回输入通道数
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, query_feature, key_feature, value_feature):
        qn, _, qh, qw = query_feature.size()  # query形状
        kvn, _, kvh, kvw = key_feature.size()  # kv形状
        # 将输入通过键、查询、值的卷积层，并调整形状以便进行矩阵乘法
        queries = self.queries(query_feature).reshape(qn, self.key_channels, qh * qw)
        keys = self.keys(key_feature).reshape((kvn, self.key_channels, kvh * kvw))
        values = self.values(value_feature).reshape((kvn, self.value_channels, kvh * kvw))

        # 计算每个头的键、查询和值的维度
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            # 对每个头，单独计算softmax归一化后的键和查询
            key = f.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = f.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            # 计算上下文向量：先通过键和值的矩阵乘法，然后与查询进行矩阵乘法
            context = key @ value.transpose(1, 2)  # 转置以匹配维度
            attended_value = (context.transpose(1, 2) @ query).reshape(qn, head_value_channels, qh, qw)
            # 将当前头计算的注意力结果添加到列表中
            attended_values.append(attended_value)

        # 将所有头计算的结果在通道维度上合并
        aggregated_values = torch.cat(attended_values, dim=1)
        # 通过重投影层将合并后的结果投影回原始的通道数
        reprojected_value = self.reprojection(aggregated_values)
        # 将注意力模块的输出与原始输入相加，实现残差连接
        attention = reprojected_value + query_feature

        return attention