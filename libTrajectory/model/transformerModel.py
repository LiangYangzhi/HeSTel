import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers=15, dim_feedforward=128):
        """
        input_dim: 输入数据的维度。在将数据传递给Transformer编码器之前，需要将输入数据线性投影到模型期望的维度d_model。
        num_layers: Transformer编码器中编码器层的数量。每个编码器层包含一个自注意力层和一个前馈网络。4
        d_model: Transformer模型中嵌入向量的维度，也是所有层中特征向量的维度。
        nhead: 注意力机制中的多头数量。在多头注意力机制中，模型可以并行处理多个注意力计算，每个头关注输入序列的不同部分。
        dim_feedforward: 前馈网络（feedforward network）中间层的维度。前馈网络通常在自注意力层之后应用。128
        """
        super(Transformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.input_projection(src)
        memory = self.transformer_encoder(src)
        output = self.output_projection(memory)
        return output[:, -1, :]
