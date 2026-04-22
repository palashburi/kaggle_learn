import os
import math
import time
import inspect
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example , iterate_examples


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

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T

        with open('input.txt' , 'r') as f:
           text =  f.read()
        enc = tiktoken.get_encoding('gpt2')
        token2 = enc.encode(text)
        self.tokens = torch.tensor(token2)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position =0

    def next_batch(self):
        B,T = self.B , self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T
        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position =0
        return x,y
       

class CasualAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
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
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd , config.vocab_size , bias = False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2* self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight , mean =0.0 , std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight , mean =0.0 , std=0.02)  

# Only two linear layers in the whole model have this flag set:
# in CausalAttention
# self.c_proj.NANOGPT_SCALE_INIT = 1

# # in MLP
# self.c_proj.NANOGPT_SCALE_INIT = 1
# These are the layers that feed directly into the residual stream via x = x + attn(...) and x = x + mlp(...). They get a smaller std to stop the residual stream from growing too large across 12 blocks.
# normal std:  0.02
# scaled std:  0.02 × (24)^-0.5  =  0.02 × 0.204  =  0.00408

        
    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size , f"Cannot forward length {T} of T greater than block size"

        pos = torch.arange(0,T,dtype = torch.long ,device = idx.device)
        token_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = token_emb + pos_emb

        for block in self.transformer.h :
            x = block(x)
            
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x) #(B,T,vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self,weight_decay, learnr , device):
        param_dict = {np:p for np,p in self.named_parameters()}
        param_dict = {np:p for np,p in param_dict.items() if p.requires_grad}

        decay_params = [p for np,p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for np,p in param_dict.items() if p.dim() < 2 ]

        optim_groups = [
            {'params' : decay_params, 'weight_decay' :weight_decay},
            {'params' : nondecay_params, 'weight_decay' : 0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
      
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with { num_nondecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learnr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
 

        
# this forwards pass explained 

# ## token_emb = self.transformer.wte(idx)

# ---

# `idx` comes in as shape `[B, T]` — a 2D grid of integers:

# ```
# idx = [[15496,  11,  314,  716],    ← sequence 0
#        [15496,  11,  314,  257],    ← sequence 1
#        [15496,  11,  314, 3303],    ← sequence 2
#        [15496,  11,  314, 2746],    ← sequence 3
#        [15496,  11,  314,   11]]    ← sequence 4

# shape [5, 4]  →  B=5, T=4
# ```

# `wte.weight` is the frozen table sitting in memory:

# ```
# wte.weight [50257, 768]:
# row 0      → [0.12, -0.3,  ..., 0.8]
# row 11     → [0.44,  0.1,  ..., -0.2]   ← ","
# row 257    → [-0.1,  0.9,  ..., 0.5]    ← "a"
# row 314    → [0.22, -0.5,  ..., 0.3]    ← " I"
# row 716    → [0.61,  0.2,  ..., -0.4]   ← " am"
# row 15496  → [0.88, -1.3,  ..., 0.7]    ← "Hello"
# ...
# ```

# PyTorch goes through every integer in `idx` and fetches its row:

# ```
# idx[0][0] = 15496  →  wte.weight[15496]  →  768 numbers  →  token_emb[0][0]
# idx[0][1] = 11     →  wte.weight[11]     →  768 numbers  →  token_emb[0][1]
# idx[0][2] = 314    →  wte.weight[314]    →  768 numbers  →  token_emb[0][2]
# idx[0][3] = 716    →  wte.weight[716]    →  768 numbers  →  token_emb[0][3]

# idx[1][0] = 15496  →  wte.weight[15496]  →  768 numbers  →  token_emb[1][0]
# idx[1][1] = 11     →  wte.weight[11]     →  768 numbers  →  token_emb[1][1]
# ...and so on for all B×T positions
# ```

# Output shape `[B, T, 768]` — same grid as idx but every integer replaced by its 768-number row.

# ---

# ## pos_emb = self.transformer.wpe(pos)

# ---

# `pos` is a 1D counter created fresh this forward pass:

# ```
# pos = [0, 1, 2, 3]    shape [T]   ← just [0, 1, 2, ... T-1]
# ```

# `wpe.weight` is the frozen position table:

# ```
# wpe.weight [1024, 768]:
# row 0   → [0.01, -0.2, ..., 0.5]   ← "I am at position 0"
# row 1   → [0.33,  0.7, ..., -0.1]  ← "I am at position 1"
# row 2   → [-0.2,  0.4, ..., 0.8]   ← "I am at position 2"
# row 3   → [0.55, -0.3, ..., 0.2]   ← "I am at position 3"
# ...
# row 1023 → [...]
# ```

# PyTorch goes through every integer in `pos` and fetches its row:

# ```
# pos[0] = 0  →  wpe.weight[0]  →  768 numbers  →  pos_emb[0]
# pos[1] = 1  →  wpe.weight[1]  →  768 numbers  →  pos_emb[1]
# pos[2] = 2  →  wpe.weight[2]  →  768 numbers  →  pos_emb[2]
# pos[3] = 3  →  wpe.weight[3]  →  768 numbers  →  pos_emb[3]
# ```

# Output shape `[T, 768]` — NOT `[B, T, 768]` because pos is 1D, same positions apply to all sequences.

# ---

# ## Key difference between the two

# ```
# idx  [B, T]    →   token_emb  [B, T, 768]   2D input → 3D output
# pos  [T]       →   pos_emb    [T, 768]       1D input → 2D output
# ```

# `pos_emb` is only `[T, 768]` not `[B, T, 768]` because every sequence shares the exact same positions — sequence 0 and sequence 4 both have a token at position 0, position 1 etc. No need to repeat it B times. PyTorch handles the addition automatically via broadcasting:

# ```
# token_emb  [B, T, 768]
# pos_emb       [T, 768]   ← PyTorch broadcasts this across B dimension
# ─────────────────────
# x          [B, T, 768]   ← every sequence gets same position vectors added
# ```
    
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
    
num_return_sequences = 5
max_length = 30
    
# model = GPT.from_pretrained('gpt2')

# model.to('mps')
# # model.eval()

 

# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hi, I am a langage model")
# tokens = torch.tensor(tokens, dtype = torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
# x = tokens.to('mps')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:

#     with torch.no_grad():
#         logits = model(x) 
#         logits = logits[:,-1,:]
#         probs = F.softmax(logits,dim=-1)
#         topkprobs, topk_indices = torch.topk(probs,50,  dim=-1) # probs vlaue , vocab ids repectively [5,50] , 5 is the number of sequences
#         # topk_probs[0]   = [0.4,  0.2,  0.15, 0.08, ...]   top 50 probs
#         # topk_indices[0] = [314,  716,  257,  2746, ...]    their token ids
#         ix = torch.multinomial(topkprobs,1) # pick one  out of 50 , and the indice and get its vocab id like a list of vocabs for the the number of sentences 
#         xcol = torch.gather(topk_indices,-1,ix) 
#         x = torch.cat((x,xcol),dim= 1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">",decoded)

# torch.manual_seed(1337)
# torch.mps.manual_seed(1337)

# total_batch_size = 4096
# B=4
# T=256
# assert total_batch_size %(B*T) == 0
# grad_accum_steps = total_batch_size//(B*T)
# print(f"total desired batch size: {total_batch_size}")
# print(f"grad_accum_steps : {grad_accum_steps}")

# train_loader = DataLoaderLite(B=B,T=T)

# torch.set_float32_matmul_precision('high')

# model = GPT(GPTConfig())
# model.to('mps')
# # model = torch.compile(model)

# max_lr = 3e-4
# min_lr = max_lr*0.1
# warmup_steps = 10
# max_steps= 820

# def get_lr(it):
#     if it < warmup_steps:
#         return max_lr * (it+1) / warmup_steps
    
#     if it > max_steps:
#         return min_lr
    
#     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
#     return min_lr + coeff*(max_lr-min_lr)



# optimizer = model.configure_optimizers(weight_decay = 0.1 , learnr=  3e-4 , device = 'mps')

# for step in range(max_steps):
#     t0 = time.time()
#     optimizer.zero_grad()
#     for micro_step in range(grad_accum_steps):
#         x,y = train_loader.next_batch()
#         x,y = x.to('mps'),y.to('mps')
#         with torch.autocast(device_type="mps", dtype=torch.bfloat16):
#             logits,loss = model(x,y)
#         loss = loss/grad_accum_steps
#         # loss_accum += loss.detach()
#         loss.backward()
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     optimizer.step()
#     torch.mps.synchronize()
#     t1 = time.time()
#     dt = (t1-t0)*1000
#     token_persec = (train_loader.B * train_loader.T*grad_accum_steps)/(t1-t0)
#     print(f"step {step:4d} , loss : {loss.item():.6f} ,norm : {norm :.4f}, dt {dt*1000: 2f} time , tokens per sec {token_persec} ")


# # after your training loop, before sys.exit()
# torch.save(model.state_dict(), 'shakespeare_gpt10_.pt')
# print("model saved to shakespeare_gpt10_.pt")

# load saved weights
# model = GPT(GPTConfig())
# model.load_state_dict(torch.load('shakespeare_gpt10_.pt', map_location='mps'))
# model.to('mps')
# model.eval()
# print("model loaded")

# # then run generation
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("ROMEO: What light through yonder")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(5, 1)
# x = tokens.to('mps')

# torch.manual_seed(42)
# torch.mps.manual_seed(42)

# while x.size(1) < 100:
#     with torch.no_grad():
#         with torch.autocast(device_type="mps", dtype=torch.bfloat16):
#             logits, loss = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# for i in range(5):
#     tokens = x[i, :100].tolist()
#     decoded = enc.decode(tokens)
#     print(f"\n> {decoded}")
#     print("-" * 40)

import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
from datasets import load_dataset
import random
import itertools
# ── paste your GPT classes here (CausalAttention, MLP, Block, GPTConfig, GPT) ──

enc = tiktoken.get_encoding('gpt2')

# ─────────────────────────────────────────
# DATASET SETUP
# ─────────────────────────────────────────


print("loading datasets...")
ds_finemath  = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", streaming=True, split="train")
ds_textbooks = load_dataset("open-phi/textbooks",      streaming=True,   split="train")
ds_stories   = load_dataset("roneneldan/TinyStories",  streaming=True,   split="train")
print("done\n")

stories_iter   = itertools.cycle(ds_stories)
math_iter      = itertools.cycle(ds_finemath)
textbooks_iter = itertools.cycle(ds_textbooks)

def get_next_text():
    global stories_iter, math_iter, textbooks_iter
    """
    mix ratio:
      50% textbooks  ← reasoning structure
      30% stories    ← language fluency
      20% math       ← math patterns
    """
    r = random.random()
    if r < 0.50:
            ex = next(textbooks_iter)
            return ex.get('textbook', '') or ex.get('text', '')
    elif r < 0.75:
            ex = next(stories_iter)
            return ex.get('text', '')
    else:
            ex = next(math_iter)
            # format math as question + reasoning + answer
            q = ex.get('question', '')
            a = ex.get('answer', '')
            return f"Question: {q}\nAnswer: {a}"
    


# ─────────────────────────────────────────
# STREAMING DATA LOADER
# ─────────────────────────────────────────

class StreamingDataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.token_buffer = []
        self.total_tokens_seen = 0

    def _fill_buffer(self, needed):
        while len(self.token_buffer) < needed:
            text = get_next_text()
            if text and len(text.strip()) > 0:
                tokens = enc.encode(text)
                self.token_buffer.extend(tokens)

    def next_batch(self):
        needed = self.B * self.T + 1
        self._fill_buffer(needed)

        buf = torch.tensor(
            self.token_buffer[:needed],
            dtype=torch.long
        )
        self.token_buffer = self.token_buffer[self.B * self.T:]  # advance

        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.total_tokens_seen += self.B * self.T
        return x, y


# ─────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, step, loss, is_best=False):
    checkpoint = {
        'step':                step,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':                loss,
        'total_tokens_seen':   train_loader.total_tokens_seen,
    }
    # always save latest
    path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    torch.save(checkpoint, path)

    # save every 500 steps permanently
    if step % 500 == 0:
        path = os.path.join(CHECKPOINT_DIR, f'step_{step:05d}.pt')
        torch.save(checkpoint, path)
        print(f"  checkpoint saved → {path}")

    # save best loss separately
    if is_best:
        path = os.path.join(CHECKPOINT_DIR, 'best.pt')
        torch.save(checkpoint, path)
        print(f"  best model saved → {path}")


def load_checkpoint(model, optimizer, path):
    print(f"loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location='mps')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step  = checkpoint['step']
    loss  = checkpoint['loss']
    print(f"  resumed from step {step}, loss {loss:.4f}")
    return step, loss


# ─────────────────────────────────────────
# TRAINING SETUP
# ─────────────────────────────────────────

torch.manual_seed(1337)
torch.mps.manual_seed(1337)

device = 'mps'

B = 4
T = 256
total_batch_size = 4096
grad_accum_steps = total_batch_size // (B * T)

max_lr       = 3e-4
min_lr       = max_lr * 0.1
warmup_steps = 50
max_steps    = 10000         # ~8M tokens seen across all three datasets

CHECKPOINT_EVERY = 100      # save every 100 steps
RESUME_FROM      = None     # set to 'checkpoints/latest.pt' to resume

train_loader = StreamingDataLoader(B=B, T=T)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


model = GPT(GPTConfig())
model.to(device)

optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learnr=max_lr,
    device=device
)

# ── resume from checkpoint if available ──
start_step = 0
best_loss  = float('inf')

if RESUME_FROM and os.path.exists(RESUME_FROM):
    start_step, _ = load_checkpoint(model, optimizer, RESUME_FROM)
elif os.path.exists('checkpoints1/latest.pt'):
    answer = input("found existing checkpoint. resume? (y/n): ")
    if answer.lower() == 'y':
        start_step, _ = load_checkpoint(model, optimizer, 'checkpoints1/latest.pt')


# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────

print(f"\nstarting training from step {start_step}")
print(f"mix: 50% textbooks | 30% stories | 20% math")
print(f"grad_accum_steps: {grad_accum_steps}")
print(f"effective batch size: {total_batch_size} tokens\n")

for step in range(start_step, max_steps):
    t0 = time.time()

    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.mps.synchronize()

    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (B * T * grad_accum_steps) / (t1 - t0)
    current_loss = loss_accum.item()

    print(f"step {step:5d} | loss: {current_loss:.4f} | "
          f"norm: {norm:.4f} | lr: {lr:.2e} | "
          f"dt: {dt:.0f}ms | tok/s: {tokens_per_sec:.0f} | "
          f"tokens seen: {train_loader.total_tokens_seen:,}")

    # ── checkpointing ──
    is_best = current_loss < best_loss
    if is_best:
        best_loss = current_loss

    if step % CHECKPOINT_EVERY == 0 or is_best:
        save_checkpoint(model, optimizer, step, current_loss, is_best)

# ── final save ──
save_checkpoint(model, optimizer, max_steps, current_loss)
torch.save(model.state_dict(), 'phi_math_pretrained.pt')
print(f"\ntraining complete. final model saved to phi_math_pretrained.pt")
print(f"total tokens seen: {train_loader.total_tokens_seen:,}")
print(f"best loss achieved: {best_loss:.4f}")
# import sys; sys.exit(0)














                         

    
