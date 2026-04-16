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

    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
model = GPT.from_pretrained('gpt2')
print("didn't crash aya!")
    




                         

    
