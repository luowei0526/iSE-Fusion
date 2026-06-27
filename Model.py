import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from transformers import AutoModel
from torch.utils.checkpoint import checkpoint
import math
import sys


model = AutoModel.from_pretrained("/mnt/data/lly/LW/multi_se/DNABERT2_Model", trust_remote_code=True)


class GCN(nn.Module):
    def __init__(self, K=3, d=3, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=2, cnn_dim=64, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=768, dropout_rate=0.2, pnode_nn=True, fnode_nn=True):
        super(GCN, self).__init__()
        self.pnode_dim = d
        self.pnode_num = 4 ** (2 * K)
        self.fnode_num = 4 ** K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout_rate
        self.pnode_nn = pnode_nn
        self.fnode_nn = fnode_nn

        self.pnode_d = nn.Linear(self.pnode_num * self.pnode_dim, self.pnode_num * self.node_hidden_dim)
        self.fnode_d = nn.Linear(self.fnode_num, self.fnode_num * self.node_hidden_dim)
        
        self.gconvs_1 = nn.ModuleList()
        self.gconvs_2 = nn.ModuleList()
        
        if self.pnode_nn:
            pnode_dim_temp = self.node_hidden_dim
        else:
            pnode_dim_temp = self.pnode_dim
        
        if self.fnode_nn:
            fnode_dim_temp = self.node_hidden_dim
        else:
            fnode_dim_temp = 1
        
        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gconvs_1.append(pyg_nn.SAGEConv((fnode_dim_temp, pnode_dim_temp), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, fnode_dim_temp), self.gcn_dim))
            else:                                   
                self.gconvs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
        
        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.convs = nn.ModuleList()
        for l in range(self.cnn_layer_num):
            if l == 0:
                self.convs.append(nn.Conv1d(in_channels=self.gcn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
            else:
                self.convs.append(nn.Conv1d(in_channels=self.cnn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))


    def forward(self, x_src, x_dst, edge_index):
        x_f = x_src
        x_p = x_dst
        edge_index_forward = edge_index[:,::2]
        edge_index_backward = edge_index[[1, 0], :][:,1::2]
         
        # transfer primary nodes
        if self.pnode_nn:
            x_p = torch.reshape(x_p, (-1, self.pnode_num * self.pnode_dim))
            x_p = self.pnode_d(x_p)
            x_p = torch.reshape(x_p, (-1, self.node_hidden_dim))
        else:
            x_p = torch.reshape(x_p, (-1, self.pnode_dim))
        
        # transfer feature nodes
        if self.fnode_nn:
            x_f = torch.reshape(x_f, (-1, self.fnode_num))
            x_f = self.fnode_d(x_f)
            x_f = torch.reshape(x_f, (-1, self.node_hidden_dim))
        else:
            x_f = torch.reshape(x_f, (-1, 1))

        for i in range(self.gcn_layer_num):
            x_p = self.gconvs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.gelu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.gconvs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.gelu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.gcn_layer_num - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, self.pnode_num))

        for i in range(self.cnn_layer_num):
            x = self.convs[i](x)
            x = F.gelu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ↑ torch.Size([16, 64, 4075])

        x = x.permute(0, 2, 1)  # torch.Size([16, 4075, 64])

        return x
    

class DNABERT2(nn.Module):
    def __init__(self):
        super(DNABERT2, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert(input_ids, token_type_ids, attention_mask)[0]  # torch.Size([16, 750, 768])
        cls = self.dropout(hidden_states[:, 0:1, :])   # torch.Size([16, 1, 768])
        
        return cls, hidden_states
    

def build_alibi_bias(max_seq_len, num_heads, device):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes_power_of_2(2 * closest_power_of_2)[: n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(num_heads), device=device)
    pos = torch.arange(max_seq_len, device=device)
    rel_pos = pos[None, :] - pos[:, None]
    rel_pos = -rel_pos.abs().float()  # 负距离
    alibi = slopes[:, None, None] * rel_pos[None, :, :]  # [H, L, L]
    return alibi


class ALiBiSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, alibi_bias=None):
        B, L, D = x.shape
        H = self.nhead

        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        if alibi_bias is not None:
            attn_scores = attn_scores + alibi_bias[None, :, :, :]  # 加上 ALiBi 偏置

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # [B, H, L, d]
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(output)


class ALiBiTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = ALiBiSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.res_scale1 = nn.Parameter(torch.ones(1))
        self.res_scale2 = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu

    def forward(self, x, alibi_bias=None):
        attn_output = self.self_attn(x, alibi_bias)

        x = self.norm1(x + self.res_scale1 * attn_output)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.res_scale2 * x2)

        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, target_len, d_model=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, target_len, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.mean_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.proj(x)
        B = x.size(0)
        query = self.query.expand(B, -1, -1)

        pooled_attn, _ = self.attn(query, x, x)
        pooled_mean = self.mean_proj(x.mean(dim=1, keepdim=True))

        output = self.norm(self.dropout(pooled_attn + pooled_mean + query))
        return output


class Shape(nn.Module):
    def __init__(self, max_seq_len=750, nhead=8):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.nhead = nhead

        self.pre_cnn = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU()
        )

        self.AttPooling_input = AttentionPooling(input_dim=256, target_len=max_seq_len)
        self.encoder_layers = nn.ModuleList([
            ALiBiTransformerEncoderLayer(
                d_model=768, nhead=nhead, dim_feedforward=3072, dropout=0.1
            )
            for _ in range(4)
        ])
        self.AttPooling_output = AttentionPooling(input_dim=768, target_len=1)

        self.register_buffer("alibi_bias", build_alibi_bias(max_seq_len, nhead, device=torch.device("cuda")))


    def forward(self, x):  # 输入 x: [B, 3000, 5]
        x = x.permute(0, 2, 1)
        x = self.pre_cnn(x)
        x = x.permute(0, 2, 1)  # [B, 3000, 256]
        
        x = self.AttPooling_input(x)  # [B, 750, 768]
        seq_len = x.size(1)

        alibi = self.alibi_bias[:, :seq_len, :seq_len]

        def custom_forward(layer, x, alibi):
            return layer(x, alibi_bias=alibi)

        for layer in self.encoder_layers:
            x.requires_grad_(True)
            x = checkpoint(custom_forward, layer, x, alibi)

        x = self.AttPooling_output(x)  # [B, 1, 768]

        return x
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=8, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, 
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, branch1, branch2):
        attn_output, attn_weights = self.cross_attn(query=branch1, key=branch2, value=branch2)
        x = self.norm1(attn_output + branch1)
        fused = self.norm2(self.ffn(x) + x)
        return fused, attn_weights


class BiCrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, num_layers=2):
        super(BiCrossAttention, self).__init__()
        self.g2b_layers = nn.ModuleList([CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)])
        self.b2g_layers = nn.ModuleList([CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)])

    def forward(self, gcn_out, bert_out):
        g, b = gcn_out, bert_out
        for i in range(len(self.g2b_layers)):
            g, _ = self.g2b_layers[i](g, b)
            b, _ = self.b2g_layers[i](b, g)
        return g, b
    

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.GCN = GCN()
        self.BERT = DNABERT2()
        self.Shape = Shape()

        self.AttPool = AttentionPooling(input_dim=64, target_len=1)

        self.align_gcn = nn.Linear(768, 256)
        self.align_bert = nn.Linear(768, 256)
        self.align_shape = nn.Linear(768, 256)
        self.temperature1 = nn.Parameter(torch.tensor(0.2))
        self.temperature2 = nn.Parameter(torch.tensor(0.2))

        self.crossatt1 = BiCrossAttention()
        self.crossatt2 = BiCrossAttention()

        self.gate1 = nn.Sequential(
                    nn.Linear(768 * 2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768),
                    nn.Sigmoid())
        
        self.gate2 = nn.Sequential(
                    nn.Linear(768 * 2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768),
                    nn.Sigmoid())
        
        self.gate3 = nn.Sequential(
                    nn.Linear(768 * 2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768),
                    nn.Sigmoid())
        
        self.gate4 = nn.Sequential(
                    nn.Linear(768 * 2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768),
                    nn.Sigmoid())
        
        self.classifier = nn.Sequential(
                          nn.LayerNorm(1536),
                          nn.Linear(1536, 768),
                          nn.LayerNorm(768),
                          nn.GELU(),
                          nn.Dropout(0.1),
                          nn.Linear(768, 2))
                                                                                         
    def forward(self, x_src, x_dst, edge_index, input_ids, token_type_ids, attention_mask, shape_features):
        gcn_out = self.AttPool(self.GCN(x_src, x_dst, edge_index))  # [16, 1, 768]
        shape = self.Shape(shape_features)  # [16, 1, 768]
        cls, hidden_states = self.BERT(input_ids, token_type_ids, attention_mask)  # cls:[16, 1, 768] hidden_states:[16, 750, 768]

        # 对齐投影
        g_proj = F.normalize(self.align_gcn(gcn_out.squeeze(1)), dim=-1)  # [B, 256]
        b_proj = F.normalize(self.align_bert(cls.squeeze(1)), dim=-1)     # [B, 256]
        s_proj = F.normalize(self.align_shape(shape.squeeze(1)), dim=-1)  # [B, 256]

        fusion_out_1, fusion_out_2 = self.crossatt1(gcn_out, cls)
        fusion_out_3, fusion_out_4 = self.crossatt2(cls, shape)

        gate1 = self.gate1(torch.cat((fusion_out_1, fusion_out_2), dim=-1))
        gate2 = self.gate2(torch.cat((fusion_out_2, fusion_out_1), dim=-1))

        gate3 = self.gate3(torch.cat((fusion_out_3, fusion_out_4), dim=-1))
        gate4 = self.gate4(torch.cat((fusion_out_4, fusion_out_3), dim=-1))

        fusion_out_gb = gate1 * fusion_out_1 + gate2 * fusion_out_2  # [16, 1, 768]
        fusion_out_gb = F.layer_norm(fusion_out_gb + fusion_out_1 + fusion_out_2, (fusion_out_gb.shape[-1],))  # [16, 1, 768]

        fusion_out_bs = gate3 * fusion_out_3 + gate4 * fusion_out_4  # [16, 1, 768]
        fusion_out_bs = F.layer_norm(fusion_out_bs + fusion_out_3 + fusion_out_4, (fusion_out_bs.shape[-1],))  # [16, 1, 768]

        fusion_out = torch.cat((fusion_out_gb, fusion_out_bs), dim=1)  # [16, 2, 768]
        fusion_out = fusion_out.flatten(start_dim=1)  # [16, 1536]
        output = self.classifier(fusion_out)

        return output, g_proj, b_proj, s_proj
    