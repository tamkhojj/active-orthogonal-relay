"""CPU stub transformer mirroring the Hugging Face decoder layout used by the
recovery experiments: model.model.layers[l].self_attn.o_proj and .mlp.down_proj,
with model.model.embed_tokens.

The stub plants a minimal refusal self-repair mechanism so the offline
self-tests produce meaningful numbers. Harmful tokens inject a component along a
rank-R generator subspace at the embedding, and mid-to-late MLPs rewrite that
component from the residual stream, so ablating it at an early layer is partially
restored downstream. It exercises the code paths without a real language model,
modelling the geometry the experiments measure rather than language itself."""
from __future__ import annotations
import torch
import torch.nn as nn


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, h):                      # h: [B,T,d]
        ctx = h.mean(dim=1, keepdim=True).expand_as(h)   # cheap "attention"
        return self.o_proj(0.1 * ctx)


class _MLP(nn.Module):
    def __init__(self, d, C=None, Gw=None, repair_gain=0.0):
        super().__init__()
        self.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)
        self.act = nn.GELU()
        # Faithful RANK-m self-repair: READ m orthogonal 'harmfulness context'
        # directions C [d,m] (which survive ablation of the generator subspace)
        # and WRITE the matching m generator directions Gw [d,m]. The refusal
        # signal is re-synthesised from context, so the recovery operator
        # context->generators has rank m.
        self.register_buffer("C", C if C is not None else torch.zeros(d, 1))
        self.register_buffer("Gw", Gw if Gw is not None else torch.zeros(d, 1))
        self.repair_gain = repair_gain

    def forward(self, h):                      # h: [B,T,d]
        base = self.down_proj(self.act(self.up_proj(h))) * 0.01
        ctx = h @ self.C                       # [B,T,m]  project onto contexts
        repair = self.repair_gain * (ctx @ self.Gw.T)   # [B,T,d] rank-m write
        return base + repair


class _Block(nn.Module):
    def __init__(self, d, C=None, Gw=None, repair_gain=0.0):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, C, Gw, repair_gain)

    def forward(self, h):
        h = h + self.self_attn(h)
        h = h + self.mlp(h)
        return (h,)                            # tuple like HF decoder layers


class _Inner(nn.Module):
    def __init__(self, vocab, d, n_layers, C, Gw, repair_layers, repair_gain):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            _Block(d,
                   C if l in repair_layers else None,
                   Gw if l in repair_layers else None,
                   repair_gain if l in repair_layers else 0.0)
            for l in range(n_layers)
        ])

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)[0]
        return h


class _Config:
    def __init__(self, vocab, d, n_layers):
        self.vocab_size = vocab
        self.hidden_size = d
        self.num_hidden_layers = n_layers


class TinyTransformer(nn.Module):
    def __init__(self, vocab=64, d=96, n_layers=10, gen_rank=3,
                 repair_layers=(5, 6, 7), repair_gain=0.5, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = _Config(vocab, d, n_layers)
        # planted generator subspace G [d, gen_rank] (orthonormal) and an EQUAL
        # number of orthogonal 'harmfulness context' directions C [d, gen_rank]
        # that survive ablation of G. The planted recovery operator maps C -> G
        # with rank = gen_rank, so Exp3 (operator rank) and Exp4 (ablation knee)
        # should both identify k* = gen_rank.
        basis = torch.randn(d, 2 * gen_rank, generator=g)
        Q, _ = torch.linalg.qr(basis)
        self.register_buffer("G", Q[:, :gen_rank])              # generators
        self.register_buffer("C", Q[:, gen_rank:2 * gen_rank])   # contexts (⊥ G)
        self.register_buffer("Gw", self.G.clone())               # write each g_j from c_j
        self.model = _Inner(vocab, d, n_layers, self.C, self.Gw,
                            set(repair_layers), repair_gain)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.gen_rank = gen_rank
        # plant: token ids 1..gen_rank are distinct 'harmful context markers',
        # each carrying the shared refusal codeword PLUS a DIFFERENT context
        # direction c_j. A harmful prompt ends in one marker, so the context
        # (and hence the regenerated generator) VARIES across prompts along all
        # gen_rank directions -> the recovery operator is genuinely rank gen_rank.
        with torch.no_grad():
            self.model.embed_tokens.weight.mul_(0.2)
            codeword = (self.G @ torch.ones(gen_rank)) * 1.5
            for j in range(gen_rank):
                self.model.embed_tokens.weight[1 + j] += codeword + self.C[:, j] * 2.5

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids)
        return type("Out", (), {"logits": self.lm_head(h)})()
