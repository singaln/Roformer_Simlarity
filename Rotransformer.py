# coding=utf-8
# @Time:2021/6/215:57
# @author: SinGaln

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadsAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadsAttention, self).__init__()
        self.args = args

        if args.embedding_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.embedding_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.embedding_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.embedding_size, self.all_head_size)
        self.key = nn.Linear(args.embedding_size, self.all_head_size)
        self.value = nn.Linear(args.embedding_size, self.all_head_size)

        self.dropout = nn.Dropout(args.attention_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def sinusoidal_position_embeddings(self, inputs):
        output_dim = self.args.embedding_size // self.args.num_attention_heads
        seq_len = inputs.size(1)
        position_ids = torch.arange(
            0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(
            0, output_dim // 2, dtype=torch.float32, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum('n,d->nd', position_ids, indices)  # [seq_len, output_dim // 2]
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)  # [seq_len, output_dim // 2, 2]
        embeddings = torch.reshape(embeddings, (seq_len, output_dim))  # [seq_len, output_dim]
        embeddings = embeddings[None, None, :, :]  # [1, 1, seq_len, output_dim]
        return embeddings

    def forward(self, inputs, attention_mask=None):
        mixed_query_layer = self.query(inputs)  # [batch_size, seq_len, hidden_size]
        mixed_key_layer = self.key(inputs)  # [batch_size, seq_len, hidden_size]
        mixed_value_layer = self.value(inputs)  # [batch_size, seq_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, num_heads, seq_len, heads_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [batch_size, num_heads, seq_len, heads_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [batch_size, num_heads, seq_len, heads_size]

        sinusoidal_positions = self.sinusoidal_position_embeddings(inputs)
        # 计算cos
        cos_pos = torch.repeat_interleave(sinusoidal_positions[..., 1::2], 2, dim=-1)
        # 计算sin
        sin_pos = torch.repeat_interleave(sinusoidal_positions[..., ::2], 2, dim=-1)
        '''
            query_layer[..., 1::2]为按列取最后一维的偶数列  shape:[batch_size, num_heads, seq_len, head_dim / 2]
            query_layer[..., ::2]为按列取的最后一维的奇数列  shape:[batch_size, num_heads, seq_len, head_dim / 2]

            通过stack拼接后得到的为增加了一维，如下例所示：
            a = [[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]]

            b = [[[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]]]

            c = torch.stack(a,b,dim=0)
            tensor([[[[ 1,  2,  3],
                  [ 4,  5,  6],
                  [ 7,  8,  9]]],

                [[[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]]]]) torch.Size([2, 1, 3, 3])
            d = torch.stack(a,b,dim=1)
            tensor([[[[ 1,  2,  3],
                  [ 4,  5,  6],
                  [ 7,  8,  9]],

                 [[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]]]]) torch.Size([1, 2, 3, 3])
            e = torch.stack(a,b,dim=-1)
            tensor([[[[ 1, 10],
                  [ 2, 20],
                  [ 3, 30]],

                 [[ 4, 40],
                  [ 5, 50],
                  [ 6, 60]],

                 [[ 7, 70],
                  [ 8, 80],
                  [ 9, 90]]]]) torch.Size([1, 3, 3, 2])
            通过以上例子就可以知道，这两个矩阵拼接后的维度增加了一维，并且是两个矩阵最后一维的元素进行拼接，如上述的e一样，
            所以torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]],dim=-1) shape:[batch_size, num_heads, seq_len, head_size/2, 2]
            最后通过reshape把最后的两维进行合并得到qw2,kw2 shape:[batch_size, num_heads,seq_len, head_dim]
            '''
        qw2 = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]],
                          dim=-1).reshape_as(query_layer)  # [batch_size, num_heads,seq_len, head_dim]
        query_layer = query_layer * cos_pos + qw2 * sin_pos  # [batch_size, num_heads, seq_len, head_dim]
        kw2 = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]],
                          dim=-1).reshape_as(key_layer)  # [batch_size, num_heads,seq_len, head_dim]
        key_layer = key_layer * cos_pos + kw2 * sin_pos  # [batch_size, num_heads, seq_len, head_dim]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.all_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对attention scores 按列进行归一化
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # dropout
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_heads, seq_len, head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # [batch_size, seq_len, embedding_size]
        return context_layer, attention_scores

class Position_Wise_Feed_Forward(nn.Module):
    def __init__(self, args):
        super(Position_Wise_Feed_Forward, self).__init__()
        self.args = args

        self.linear1 = nn.Linear(args.embedding_size, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.embedding_size)
        self.dropout = nn.Dropout(args.feed_dropout_rate)
        self.layer_norm = nn.LayerNorm(args.embedding_size)

    def forward(self, x):
        # x:[batch_size, seq_len, embedding_size]
        outputs = self.dropout(self.linear2(nn.functional.relu(self.linear1(x))))
        outputs = outputs + x  # 残差连接
        outputs = self.layer_norm(outputs)
        return outputs

class Pooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.embedding_size, args.embedding_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RoTransformerEncoder(nn.Module):
    def __init__(self, args):
        super(RoTransformerEncoder, self).__init__()
        self.args = args

        self.multi_attention = MultiHeadsAttention(args)
        self.feed_forward = Position_Wise_Feed_Forward(args)
        self.pooler = Pooler(args)
        self.dense = nn.Linear(args.embedding_size, 4)


    def forward(self, x):
        context, attention_score = self.multi_attention(x)
        outputs = self.feed_forward(context)
        outputs = self.pooler(outputs)
        print("outputs", outputs, outputs.shape)
        logits = self.dense(outputs)
        print("logits", logits, logits.shape)
        return logits