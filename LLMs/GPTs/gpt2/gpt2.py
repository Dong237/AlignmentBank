import time
import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
from loader import DataLoaderChinese

@dataclass
class GPTConfig():
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 768
    vocab_size: int = 50257
    window_size: int = 1024
    dropout: int = 0.1
    batch_size: int = 512

class GPT2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.c_attn = nn.Linear(config.d_model, 3*config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.window_size, config.window_size)).view(
                1, 1, config.window_size, config.window_size
                )
            )

        # NOTE Residual weight scaling at initialization
        self.RESID_SCALING = 1
    
    def forward(self, x):

        B, T, C = x.shape

        qkv = self.c_attn(x)           # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # (B, T, C)

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1,2) # (B, nh, T, d_k)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1,2) # (B, nh, T, d_k)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1,2) # (B, nh, T, d_v)

        attn = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(C // self.config.n_heads, dtype=torch.float))  # (B, nh, T, T)
        masked_attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))     # (B, nh, T, T)
        attn_weights = F.softmax(masked_attn, dim=-1)               # (B, nh, T, T)
        # apply dropout
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)                       # (B, nh, T, d_v)

        # NOTE Accelerate Option 4: Flash Attention
        # context = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        context = context.transpose(1,2).contiguous().view(B, T, C)   # (B, T, C)
        context = self.c_proj(context)                                # (B, T, C)
        context = self.resid_dropout(context)                         # (B, T, C)

        return context


class GPT2MLP(nn.Module):

    def __init__(self, config):

        super().__init__()
        
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj = nn.Linear(4*config.d_model, config.d_model)
        self.act = nn.GELU(approximate='tanh')    # NOTE pay attention to how to use
        self.dropout = nn.Dropout(config.dropout)

        # NOTE Residual weight scaling at initialization
        self.RESID_SCALING = 1
    
    def forward(self, x):
        
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = GPT2MLP(config)
    
    def forward(self, x):

        x = x + self.attn(self.ln_1(x))  # NOTE pre-normalization
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.window_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),  # NOTE adidtional normalization
        ))

        # NOTE lm_head is in charge of dime aligning and info integration, bias should be False.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # NOTE Weight sharing used in the original transformer paper
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "RESID_SCALING"):
                std *= torch.sqrt(2 * self.config.n_layers) 
            nn.init.normal_(module, mean=0.0, std=std)
            if module.bias:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, inp, target=None):
        
        B, T = inp.shape
        pos = torch.arange(0, T, dtype=torch.long, device=inp.device)
        pe = self.transformer.wpe(pos)
        te = self.transformer.wte(inp)

        x = te + pe
        # NOTE here is the dropout
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            # NOTE pay attention to how to adapt the dimension of tensors
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=GPTConfig()):

        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_name_or_path)
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
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """This method is given by Claude"""
        # Clone input_ids to avoid modifying the original tensor
        input_ids = input_ids.clone()
        
        # Get batch size and current sequence length
        B, T = input_ids.shape
        
        # Maximum allowed sequence length
        max_seq_length = min(self.config.window_size, T + max_new_tokens)
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # If the sequence is too long, truncate it to fit within the model's context window
            if input_ids.shape[1] >= self.config.window_size:
                # Take the last window_size tokens
                input_ids = input_ids[:, -self.config.window_size:]
            
            # Forward pass to get logits
            logits, _ = self.forward(input_ids)
            
            # Focus only on the last token's predictions
            logits = logits[:, -1, :] # Shape: (B, vocab_size)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits below the top-k threshold to -inf
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)
            
            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # If we've reached the maximum sequence length, stop generating
            if input_ids.shape[1] >= max_seq_length:
                break
        
        return input_ids

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # NOTE this way of initializing AdamW is rare to see.
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    

if __name__ == "__main__":
    # NOTE A100 while paper: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available else "cpu")

    data_path = "/data/repos/learning/LLManager/chinese-poetry-collection/train.csv"
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    batch_size = 16
    seq_length = 128
    gradient_accum_step = total_batch_size // (batch_size * seq_length)

    data_loader = DataLoaderChinese(data_path, batch_size, seq_length)

    # NOTE Accelerate Option 5: replace ugly numbers with nice nunmbers (e.g., 2en)
    config = GPTConfig(vocab_size=len(data_loader.vocab))
    # model = GPT2.from_pretrained("/data/repos/huggingface/gpt2")
    # NOTE Acceleration Option 1: using TF32 here
    torch.set_float32_matmul_precision("high")
    model = GPT2(config).to(device)

    # NOTE Acceleration Option 3: complie the model
    # model = torch.complie(model)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # Training Setup: Weight Decay
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    for step in range(50):
        t0 = time.time()
        optimizer.zero_grad()

        # Training Setup: Gradient Accumulation
        for micro_step in range(gradient_accum_step):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # NOTE Acceleration Option 2: using BF16 here
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                # import code; code.interact(local=locals())
            loss = loss / gradient_accum_step
            loss_accum += loss.detach()
            loss.backward()
        
        # Training Setup: Gradient clipping
        norm = nn.utils.cli_grad_norm_(model.parameters(), 1.0)

        # Training Setup: Learning rate Scheduling
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_per_sec = (x.shape[0] * x.shape[1]) / dt
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")