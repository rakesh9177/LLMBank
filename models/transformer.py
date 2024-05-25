import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        assert input_dim % num_heads == 0
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)

        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value, mask = None):
        b = query.shape[0]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = q.view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        dot_product_scores = torch.matmul(q,k.transponse(-2, -1)) / torch.sqrt(self.dim_per_head)

        if mask is not None:
            dot_product_scores = dot_product_scores.masked_fill(mask==0, dim=-1)

        attention = F.softmax(dot_product_scores, dim=-1) @ v

        out = attention.transpose(1,2).contiguous()
        out = out.view(b,-1, self.input_dim)
        out = self.out(out)

        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=10000):
        self.input_dim = input_dim
        self.max_len = max_len
    def forward(self, x):
        seq_len = x.shape[1]
        input_dim = x.shape[2]

        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = 1/ torch.pow(self.max_len, torch.arange(0, input_dim, 2) /input_dim)
        pe = torch.zeros(seq_len, input_dim)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        x = x + pe
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, ff_dim, dropout):
        self.linear_W1 = nn.Linear(input_dim, ff_dim)
        self.linear_W2 = nn.Linear(ff_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_W2(self.relu(self.linear_W1(x)))
    

class TransformerEncodeCell(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout):
        self.mha = MultiHeadAttention(input_dim, num_heads)
        self.droput1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.ff = FeedForwardNetwork(input_dim, ff_dim)
        self.droput2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(input_dim)

    def forward(self, x, mask = None):
        z = self.mha(x, x, x, mask)
        z = self.droput1(z)
        z = self.layernorm1(z+x)
        y = self.ff(z)
        y = self.droput2(y)
        y = self.layernorm2(y+z)

        return y
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_cells, droput = 0.1):
        self.norm = nn.LayerNorm(input_dim)
        self.cells = nn.ModuleList([
            TransformerEncodeCell(input_dim, num_heads, ff_dim, droput)
            for _ in range(num_cells)
        ])
    def forward(self, x, mask = None):
        for encoder_cell in self.cells:
            x = encoder_cell(x)
        x = self.norm(x)
        return x
    

class TransformerDecoderCell(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, droput = 0.1):
        self.mha_e = MultiHeadAttention(input_dim, input_dim)
        self.mha_d = MultiHeadAttention(input_dim, input_dim)
        self.ff = FeedForwardNetwork(input_dim, ff_dim, droput)
        self.dropout1 = nn.Dropout(droput)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(droput)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.dropout3 = nn.Dropout(droput)
        self.layernorm3 = nn.LayerNorm(input_dim)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        d = self.mha_d(x, x, x, tgt_mask)
        d = self.dropout1(d)
        d = self.layernorm1(d+x)

        f = self.mha_e(d, encoder_output, encoder_output, src_mask)
        f = self.dropout2(f)
        f = self.layernorm2(f + d)

        g = self.ff(f)
        g = self.dropout3(g)
        g = self.layernorm3(g + f)

        return g
    
class TranformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_cells, dropout = 0.1):
        self.norm = nn.LayerNorm(input_dim)
        self.cells = nn.ModuleList([
            TransformerDecoderCell(input_dim, num_heads, ff_dim, dropout)
            for _ in range(num_cells)
        ])
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for decoder_cell in self.cells:
            x = decoder_cell(x, encoder_out, src_mask, tgt_mask)
        x = self.norm(x)
        return x
    
    
    







        