import torch
import numpy as np
import torch.nn as nn
import copy
import math
from torch.nn import functional as F
from torch.autograd import Variable, Function


def Linear(inputdim, outputdim, bias=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    return linear


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head_num == 0
        self.d_k = d_model // head_num
        self.head = head_num
        self.linears = clone(Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores.masked_fill_(mask == 0, -1e9)
        p_att = F.softmax(scores, -1)
        if self.dropout:
            p_att = self.dropout(p_att)
        return torch.matmul(p_att, v)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.head * self.d_k)
        x = self.linears[-1](x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.w_1(x), inplace=True)
        if self.dropout:
            h = self.dropout(h)
        return self.w_2(h)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x, sublayer):
        if self.dropout:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + sublayer(x))

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def it_guide_thi(out_im, out_t, thr, biatten_t2i, lamda = 0):
    upper_t, lower_t = [], []
    softmaxed_tensor_t = F.softmax(out_t, dim=-1)
    txt_result = softmaxed_tensor_t
    
    for i in range(txt_result.size(0)):
        for j in range(i+1, txt_result.size(0)):
            if torch.any(txt_result[i][j] > thr):
                upper_t.append([i,j])

    for i in range(txt_result.size(0)):
        for j in range(0, i):
            if torch.any(txt_result[i][j] > thr):
                lower_t.append([i,j])

    all_t = upper_t + lower_t 
    result_con_list_t = [] 
    for index_pair in all_t:
        row1, row2 = index_pair
        max_indices = [int(torch.argmax(biatten_t2i[row1])), int(torch.argmax(biatten_t2i[row2]))]
        result_con_list_t.append(max_indices)
        
    for pair_id in range(len(all_t)):
        sub1 = all_t[pair_id]
        sub2 = result_con_list_t[pair_id]
        out_im[sub2[0]][sub2[1]] = out_im[sub2[0]][sub2[1]] + lamda * out_t[sub1[0]][sub1[1]]
        
    return out_im


def it_guide_iht(out_im, out_t, thr, biatten_i2t, lamda = 0):
    upper_im, lower_im = [], []
    softmaxed_tensor_im = F.softmax(out_im, dim=-1)
    img_result = softmaxed_tensor_im
    
    for i in range(img_result.size(0)):
        for j in range(i+1, img_result.size(0)):
            if torch.any(img_result[i][j] > thr):
                upper_im.append([i,j])

    for i in range(img_result.size(0)):
        for j in range(0, i):
            if torch.any(img_result[i][j] > thr):
                lower_im.append([i,j])

    all_im = upper_im + lower_im 
    result_con_list_im = []
    for index_pair in all_im:
        row1, row2 = index_pair
        max_indices = [int(torch.argmax(biatten_i2t[row1])), int(torch.argmax(biatten_i2t[row2]))]
        result_con_list_im.append(max_indices)
        
    for pair_id in range(len(all_im)):
        sub1 = all_im[pair_id]   
        sub2 = result_con_list_im[pair_id] 
        out_t[sub2[0]][sub2[1]] = out_t[sub2[0]][sub2[1]] + lamda * out_im[sub1[0]][sub1[1]]
        

    return out_t
