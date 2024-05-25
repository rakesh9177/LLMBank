import torch
import torch.nn as nn
import transformer

'''
According to BERT paper, below are the parameters used

L = number of encoder blocks = num_cells
H = hidden size = ff_dim 
A = number of self_attention heds = num_heads 

Bert_base = (L = 12, H = 768, A = 12) => Total param = 110M
BERT_large = (L = 24, H = 1024, A = 16) => Total param = 340M
'''
class BERTEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_cells, droput = 0.1):
        self.norm = nn.LayerNorm(input_dim)
        self.cells = nn.ModuleList([
            transformer.TransformerEncodeCell(input_dim, num_heads, ff_dim, droput)
            for _ in range(num_cells)
        ])
    def forward(self, x, mask = None):
        if mask is None:
            mask = torch.rand_like(x) < 0.15
        for encoder_cell in self.cells:
            x = encoder_cell(x, mask)
        x = self.norm(x)
        return x
    
class BERTNSP(nn.Module):
    def __init__(self,x):
        '''
        Work in progress
        '''
        pass

