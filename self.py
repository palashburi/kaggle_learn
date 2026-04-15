import os
import math
import time
import inspect
from dataclass import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example , iterate_examples


#VISULAISING X FOR UNDERSTADING , X CONTAINS A LIST OF TOKENS REPRESENTED
# BY EMBEDDING AND THOSE TOKENS REPRESENT A CERTAIN SEQUENCE IN THAT BATCH

# x shape: [B, T, 768]
#         |  |   |
#         |  |   └── each token is a 768-dim vector
#         |  └────── T tokens in the sequence
#         └───────── B sequences in the batch

# So for the sentence "the cat sat on" with T=4:
# x = [
#   [0.2, -1.3, ..., 0.8],   ← "the"  (768 numbers)
#   [0.5,  0.1, ..., 0.3],   ← "cat"  (768 numbers)
#   [0.1,  0.9, ..., -0.2],  ← "sat"  (768 numbers)
#   [0.8, -0.4, ..., 0.6],   ← "on"   (768 numbers)
# ]

class CasualAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        #key,query,value projections for all heads , bt in a batch
        self.c_attn = nn.Linear(config.n_embd , 3*config.n_embd)

        # this layer how do you look at it :
        # You can mentally think of this W as three matrices stacked vertically:
        # W [2304 × 768] = | Wq [768 × 768] |
        #          | Wk [768 × 768] |
        #          | Wv [768 × 768] |

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT =1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self,x):
        B,T,C = x.size() #batch size , sequence length , embedding dimentionality(n_embd)

        # nh is the numebr fo heads
        # hs is the head size 
        # C is the number of channels = nh* hs 

        qkv = self.c_attn(x)       #dim 2
        # q = x @ Wq.T    # [B, T, 768]
        # k = x @ Wk.T    # [B, T, 768]
        # v = x @ Wv.T    # [B, T, 768]
        q,k,v = qkv.split(self.n_embd, dim = 2)

        k = k.view(B,T,self.n_head , C// self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head , C// self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head , C// self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C) #re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd , config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT =1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x+ self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768


class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size , config.n_embd),
            wpe = nn.Embedding(config.block_size , config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emdb),
        ))

        self.lm_head = nn.Linear(config.n_embd , config.vocab_size , bias = False)
        # self.transformer.wte.weight = self.lm_head.weight
        # self.apply(self._init_weights)

    

    




                         

    
