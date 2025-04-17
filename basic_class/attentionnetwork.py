import torch
import torch.nn as nn

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, output_dim):
        """
        Args:
            input_dim: 每个邻居或障碍物输入特征的维度
            embed_dim: 注意力模块中嵌入的维度（需要与num_heads兼容）
            num_heads: 多头注意力的头数
            output_dim: 最终聚合后的输出维度
        """
        super(AttentionNetwork, self).__init__()
        # 如果输入特征维度和嵌入维度不一致，可以先做一个线性映射
        self.input_projection = nn.Linear(input_dim, embed_dim)
        # 定义多头自注意力模块（这里设置 batch_first=True 方便输入形状为 (B, seq_len, embed_dim)）
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # 最后使用一个全连接层将聚合后的表示映射到固定维度输出
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor，形状为 (batch_size, num_elements, input_dim)
               其中 num_elements 可以是邻居数量或者障碍物数量，不必固定。
        Returns:
            out: Tensor，形状为 (batch_size, output_dim)，固定维度的输出
        """
        # 投影到嵌入空间
        x_proj = self.input_projection(x)  # (B, seq_len, embed_dim)
        # 使用自注意力机制，注意这里的查询、键、值都使用相同的输入
        attn_output, _ = self.multihead_attn(x_proj, x_proj, x_proj)
        # 对所有元素进行池化（例如平均池化），聚合成固定维度
        pooled = attn_output.mean(dim=1)  # (B, embed_dim)
        # 映射到最终输出维度
        out = self.fc(pooled)  # (B, output_dim)
        return out