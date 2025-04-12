import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig():
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 768
    vocab_size: int = 50257
    window_size: int = 1024
    dropout: float = 0.1
    batch_size: int = 512

class GPT2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.window_size, config.window_size)).view(1, 1, config.window_size, config.window_size))

        self.RESID_SCALING = 1

    def forward(self, x, past_kv=None):
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)  # (B, nh, T, d_k)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)  # (B, nh, T, d_k)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)  # (B, nh, T, d_v)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, nh, T_total, d_k)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)  # return updated cache

        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32, device=x.device))  # (B, nh, T, T_total)
        mask = self.mask[:, :, T - 1:T, :k.size(2)]  # (1, 1, 1, T_total)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, nh, T, d_v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        context = self.c_proj(context)
        context = self.resid_dropout(context)
        return context, present_kv

class GPT2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.act = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(config.dropout)
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

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            wpe=nn.Embedding(config.window_size, config.d_model),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, inp, target=None, past_kv=None):
        B, T = inp.shape
        device = inp.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        te = self.transformer.wte(inp)  # (B, T, C)
        pe = self.transformer.wpe(pos)  # (1, T, C)
        x = te + pe
        x = self.transformer.drop(x)

        new_past_kv = [] if past_kv is not None else None

        for i, block in enumerate(self.transformer.h):
            past = past_kv[i] if past_kv is not None else None
            x, present = block(x, past)
            if new_past_kv is not None:
                new_past_kv.append(present)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss, new_past_kv

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        B, T = input_ids.shape
        device = input_ids.device
        past_kv = None
        generated = input_ids

        for _ in range(max_new_tokens):
            x = generated[:, -1:]  # (B, 1)
            logits, _, past_kv = self.forward(x, past_kv=past_kv)  # get logits and cache
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_vals = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_vals, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated
