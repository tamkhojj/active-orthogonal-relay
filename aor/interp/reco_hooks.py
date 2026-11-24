"""Re-entrant forward-hook engine for activation ablation.

Supported modes:
  - writes      : ablation on o_proj and down_proj (and embed_tokens when
                  ablate_embed=True);
  - mlp_writes  : ablate only mlp.down_proj writes;
  - attn_writes : ablate only self_attn.o_proj writes;
  - full_resid  : pre-block, pre-MLP, and post-block residual ablation;
  - resid       : post-block output only."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torch


def get_blocks(model):
    return model.model.layers


def get_attn_out(block):
    return block.self_attn.o_proj


def get_mlp_out(block):
    return block.mlp.down_proj


def _get_embed_tokens(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens
    return None


def _get_pre_mlp_norm(block):
    if hasattr(block, "pre_feedforward_layernorm"):      # Gemma-2
        return block.pre_feedforward_layernorm
    if hasattr(block, "post_attention_layernorm"):       # Llama, Qwen2/2.5, Mistral
        return block.post_attention_layernorm
    return None


def _pick(h: torch.Tensor, token_pos: Union[int, torch.Tensor]) -> torch.Tensor:
    if isinstance(token_pos, torch.Tensor):
        B = h.shape[0]
        idx = token_pos.to(h.device).long()
        return h[torch.arange(B, device=h.device), idx, :]
    return h[:, token_pos, :]


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, new_h):
    if isinstance(out, tuple):
        return (new_h,) + tuple(out[1:])
    return new_h


class RecoveryHooks:
    def __init__(self, model,
                 blocks_fn=get_blocks, attn_fn=get_attn_out, mlp_fn=get_mlp_out):
        self.model = model
        self.blocks = list(blocks_fn(model))
        self.attn_mods = [attn_fn(b) for b in self.blocks]
        self.mlp_mods = [mlp_fn(b) for b in self.blocks]
        self.pre_mlp_norms = [_get_pre_mlp_norm(b) for b in self.blocks]
        self.embed_mod = _get_embed_tokens(model)
        self.n_layers = len(self.blocks)
        self._handles: List[object] = []
        self._store: Dict[str, Dict[int, torch.Tensor]] = {}
        self.force_stats: Dict[Tuple[int, str], Dict[str, float]] = {}

    # ---- capture (100% compatible with Exp2-Exp5) --------------------------- #
    def _cap_block(self, idx, token_pos):
        def hook(_m, _i, out):
            self._store["resid"][idx] = _pick(_hidden(out), token_pos).detach().float().cpu()
        return hook

    def _cap_linear(self, bucket, idx, token_pos):
        def hook(_m, _i, out):
            self._store[bucket][idx] = _pick(out, token_pos).detach().float().cpu()
        return hook

    # ---- intervene -------------------------------------------------------- #
    def _init_force_stat(self, key: Tuple[int, str]):
        if key not in self.force_stats:
            self.force_stats[key] = {
                "sum_delta_norm": 0.0,
                "sum_h_norm": 0.0,
                "count": 0.0,
                "nan_or_inf": 0.0,
            }

    def _ablate_pre_hook(self, U: torch.Tensor, positions: str, alpha: float = 1.0):
        Uf = U.float()
        if Uf.ndim == 1:
            Uf = Uf.view(-1, 1)

        def hook(_m, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return args
            h = args[0]
            Ud = Uf.to(dtype=h.dtype, device=h.device)
            if positions == "all":
                h2 = h - alpha * ((h @ Ud) @ Ud.T)
            else:
                h2 = h.clone()
                last = h[:, -1, :]
                h2[:, -1, :] = last - alpha * ((last @ Ud) @ Ud.T)
            return (h2,) + tuple(args[1:])

        return hook

    def _ablate_hook(self, U: torch.Tensor, positions: str,
                     layer_idx: Optional[int] = None,
                     sublayer_tag: str = "resid",
                     track_force: bool = False, alpha: float = 1.0):
        Uf = U.float()
        if Uf.ndim == 1:
            Uf = Uf.view(-1, 1)

        def hook(_m, _i, out):
            h = _hidden(out)
            Ud = Uf.to(dtype=h.dtype, device=h.device)
            if positions == "all":
                coeff = h @ Ud
                delta = alpha * (coeff @ Ud.T)
                h2 = h - delta
            else:
                h2 = h.clone()
                last = h[:, -1, :]
                delta_last = alpha * ((last @ Ud) @ Ud.T)
                h2[:, -1, :] = last - delta_last
                delta = delta_last

            if track_force and layer_idx is not None:
                with torch.no_grad():
                    key = (int(layer_idx), sublayer_tag)
                    self._init_force_stat(key)
                    if positions == "all":
                        d_last = delta[:, -1, :].float()
                        h_last = h[:, -1, :].float()
                    else:
                        d_last = delta.float()
                        h_last = h[:, -1, :].float()
                    dn = d_last.norm(dim=-1)
                    hn = h_last.norm(dim=-1).clamp_min(1e-8)
                    st = self.force_stats[key]
                    st["sum_delta_norm"] += float(dn.sum().item())
                    st["sum_h_norm"] += float(hn.sum().item())
                    st["count"] += float(dn.numel())
                    if torch.isnan(h2).any() or torch.isinf(h2).any():
                        st["nan_or_inf"] += 1.0
            return _rewrap(out, h2)

        return hook

    @staticmethod
    def _add_hook(vec: torch.Tensor, coeff: float, positions: str):
        vf = vec.float().squeeze()

        def hook(_m, _i, out):
            h = _hidden(out)
            v = (coeff * vf).to(dtype=h.dtype, device=h.device)
            if positions == "all":
                h2 = h + v
            else:
                h2 = h.clone()
                h2[:, -1, :] = h[:, -1, :] + v
            return _rewrap(out, h2)

        return hook

    def summarize_force(self) -> Dict[str, object]:
        per_layer_norm = [0.0] * self.n_layers
        per_layer_hnorm = [0.0] * self.n_layers
        embed_force = 0.0
        any_nan = False

        for (l, _tag), st in self.force_stats.items():
            cnt = max(st["count"], 1.0)
            mean_dn = st["sum_delta_norm"] / cnt
            mean_hn = st["sum_h_norm"] / cnt
            if l == -1:
                embed_force += mean_dn
            elif 0 <= l < self.n_layers:
                per_layer_norm[l] += mean_dn
                per_layer_hnorm[l] += mean_hn
            if st["nan_or_inf"] > 0:
                any_nan = True

        per_layer_ratio = [
            round(dn / hn, 5) if hn > 1e-8 else 0.0
            for dn, hn in zip(per_layer_norm, per_layer_hnorm)
        ]
        per_layer_norm_r = [round(x, 4) for x in per_layer_norm]
        active_ratios = [r for r in per_layer_ratio if r > 0]
        return {
            "per_layer_force_norm": per_layer_norm_r,
            "per_layer_force_ratio": per_layer_ratio,
            "embed_force_norm": round(embed_force, 4),
            "total_force_norm": round(sum(per_layer_norm) + embed_force, 4),
            "mean_active_ratio": round(sum(active_ratios) / max(1, len(active_ratios)), 5),
            "nan_or_inf": any_nan,
        }

    # ---- public API ------------------------------------------------------- #
    @contextmanager
    def intervene(self, ablate_U: Optional[torch.Tensor] = None,
                  ablate_layers: Optional[Iterable[int]] = None,
                  ablate_positions: str = "all",
                  ablate_by_layer: Optional[dict] = None,
                  ablate_mode: str = "resid",
                  ablate_embed: bool = False,
                  alpha: float = 1.0,
                  track_force: bool = False,
                  add_vec: Optional[torch.Tensor] = None,
                  add_coeff: float = 0.0,
                  add_layers: Optional[Iterable[int]] = None,
                  add_positions: str = "all"):
        abl = set(int(x) for x in ablate_layers) if ablate_layers is not None else set()
        addl = set(int(x) for x in add_layers) if add_layers is not None else set()
        byl = {int(k): v for k, v in ablate_by_layer.items()} if ablate_by_layer else {}
        if track_force:
            self.force_stats = {}
        local_handles = []
        try:
            if ablate_embed and self.embed_mod is not None:
                embed_U = ablate_U
                if embed_U is None and byl:
                    embed_U = byl[min(byl.keys())]
                if embed_U is not None:
                    local_handles.append(self.embed_mod.register_forward_hook(
                        self._ablate_hook(embed_U, ablate_positions, -1, "embed", track_force, alpha)
                    ))

            for l, blk in enumerate(self.blocks):
                target_U = byl[l] if l in byl else (ablate_U if (ablate_U is not None and l in abl) else None)
                if target_U is not None:
                    if ablate_mode == "writes":
                        h_a = self.attn_mods[l].register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "attn", track_force, alpha))
                        h_m = self.mlp_mods[l].register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "mlp", track_force, alpha))
                        local_handles.extend([h_a, h_m])
                    elif ablate_mode == "mlp_writes":
                        h_m = self.mlp_mods[l].register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "mlp", track_force, alpha))
                        local_handles.append(h_m)
                    elif ablate_mode == "attn_writes":
                        h_a = self.attn_mods[l].register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "attn", track_force, alpha))
                        local_handles.append(h_a)
                    elif ablate_mode == "full_resid":
                        local_handles.append(blk.register_forward_pre_hook(
                            self._ablate_pre_hook(target_U, ablate_positions, alpha)))
                        mid_norm = self.pre_mlp_norms[l]
                        if mid_norm is not None:
                            local_handles.append(mid_norm.register_forward_pre_hook(
                                self._ablate_pre_hook(target_U, ablate_positions, alpha)))
                        else:
                            local_handles.append(self.attn_mods[l].register_forward_hook(
                                self._ablate_hook(target_U, ablate_positions, None, "attn", False, alpha)))
                        local_handles.append(blk.register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "resid", track_force, alpha)))
                    else:
                        local_handles.append(blk.register_forward_hook(
                            self._ablate_hook(target_U, ablate_positions, l, "resid", track_force, alpha)))
                if add_vec is not None and l in addl:
                    local_handles.append(blk.register_forward_hook(
                        self._add_hook(add_vec, add_coeff, add_positions)))
            self._handles.extend(local_handles)
            yield self
        finally:
            for h in local_handles:
                try:
                    h.remove()
                except Exception:
                    pass
                if h in self._handles:
                    self._handles.remove(h)

    @contextmanager
    def capture(self, token_pos: Union[int, torch.Tensor] = -1,
                ablate_U: Optional[torch.Tensor] = None,
                ablate_layers: Optional[Iterable[int]] = None,
                ablate_positions: str = "all",
                ablate_by_layer: Optional[dict] = None,
                add_vec: Optional[torch.Tensor] = None,
                add_coeff: float = 0.0,
                add_layers: Optional[Iterable[int]] = None,
                add_positions: str = "all"):
        self._store = {"resid": {}, "attn": {}, "mlp": {}}
        abl = set(int(x) for x in ablate_layers) if ablate_layers is not None else set()
        addl = set(int(x) for x in add_layers) if add_layers is not None else set()
        byl = {int(k): v for k, v in ablate_by_layer.items()} if ablate_by_layer else {}
        local_handles = []
        try:
            for l, blk in enumerate(self.blocks):
                target_U = byl[l] if l in byl else (ablate_U if (ablate_U is not None and l in abl) else None)
                if target_U is not None:
                    local_handles.append(blk.register_forward_hook(
                        self._ablate_hook(target_U, ablate_positions)))
                if add_vec is not None and l in addl:
                    local_handles.append(blk.register_forward_hook(
                        self._add_hook(add_vec, add_coeff, add_positions)))
            for l, blk in enumerate(self.blocks):
                local_handles.append(blk.register_forward_hook(self._cap_block(l, token_pos)))
                local_handles.append(self.attn_mods[l].register_forward_hook(
                    self._cap_linear("attn", l, token_pos)))
                local_handles.append(self.mlp_mods[l].register_forward_hook(
                    self._cap_linear("mlp", l, token_pos)))
            self._handles.extend(local_handles)
            yield self
        finally:
            for h in local_handles:
                try:
                    h.remove()
                except Exception:
                    pass
                if h in self._handles:
                    self._handles.remove(h)

    def stacked(self, bucket: str) -> torch.Tensor:
        d = self._store[bucket]
        return torch.stack([d[l] for l in range(self.n_layers)], dim=0)

    def release(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self.model = None
        self.blocks = []
        self.attn_mods = []
        self.mlp_mods = []
        self.pre_mlp_norms = []
        self.embed_mod = None
        self._store = {}
        self.force_stats = {}


@torch.no_grad()
def forward_pass(model, input_ids, attention_mask=None):
    kw = {}
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    return model(input_ids, **kw)


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=1) - 1