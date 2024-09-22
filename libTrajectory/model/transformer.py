import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, input_dim):
        super(Transformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.input_projection(src)
        memory = self.transformer_encoder(src)
        output = self.output_projection(memory)
        return output


class TwinTowerTransformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, input_dim_user, input_dim_item):
        """
        num_layers           编码器层数
        d_model              模型中嵌入的维度
        nhead                多头注意力的头数
        dim_feedforward      前馈网络的维度
        max_seq_length       序列最大长度
        """
        super(TwinTowerTransformer, self).__init__()
        self.user_tower = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim_user)
        self.item_tower = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim_item)
        self.combination_layer = nn.Linear(d_model * 2, 1)  # 或者使用其他方式合并

    def forward(self, user_data, item_data):
        user_embedding = self.user_tower(user_data)
        item_embedding = self.item_tower(item_data)
        return user_embedding, item_embedding


# device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pos_table = np.array([
#         [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
#         if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
#         pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
#         pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
#         if device == "cuda":
#             self.pos_table = torch.FloatTensor(pos_table).cuda()               # enc_inputs: [seq_len, d_model]
#         else:
#             self.pos_table = torch.FloatTensor(pos_table)
#
#     def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
#         enc_inputs += self.pos_table[:enc_inputs.size(1), :]
#         if device == "cuda":
#             return self.dropout(enc_inputs.cuda())
#         else:
#             return self.dropout(enc_inputs)
#
#
# class TransformerModel(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
#                  dim_feedforward, max_seq_length):
#         """
#         src_vocab_size       源语言词汇表大小
#         tgt_vocab_size       目标语言词汇表大小
#         d_model              模型中嵌入的维度
#         nhead                多头注意力的头数
#         num_encoder_layers   编码器层数
#         num_decoder_layers   解码器层数
#         dim_feedforward      前馈网络的维度
#         max_seq_length       序列最大长度
#         """
#         super(TransformerModel, self).__init__()
#         self.src_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_length)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
#         self.fc_out = nn.Linear(d_model, tgt_vocab_size)
#
#     def forward(self, src, tgt):
#         src = self.positional_encoding(self.src_embedding(src))
#         tgt = self.positional_encoding(self.tgt_embedding(tgt))
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)
