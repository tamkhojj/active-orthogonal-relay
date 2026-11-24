"""Shared model backend for Exp2-Exp6.

Provides:
  1. standardized chat formatting with no system prompt, removing system-prompt
     confounders across Llama-3/3.1, Gemma-2, Qwen2/2.5, and Mistral;
  2. strict train/held-out separation with an explicit zero-overlap assertion;
  3. bfloat16 by default and mini-batched activation capture;
  4. automatic behavioral-peak and Causal-Basin discovery (ablate_mode="writes")."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import json
import torch

from aor.interp import recovery as R
from aor.interp import reco_hooks as H


def format_chat_no_system(text: str, model_name: str, tok=None) -> str:
    """Minimal chat template with NO system prompt (Arditi et al. Table 6 &
    AOR exp3_jailbreak_asr.py), ensuring exact BOS handling per architecture
    when tokenized with `add_special_tokens=False`."""
    name = (model_name or "").lower()
    if "llama-3" in name or "llama3" in name:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if "gemma" in name:
        bos = (getattr(tok, "bos_token", None) or "<bos>") if tok is not None else "<bos>"
        return f"{bos}<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    if "qwen" in name:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    if "mistral" in name:
        bos = (getattr(tok, "bos_token", None) or "<s>") if tok is not None else "<s>"
        return f"{bos}[INST] {text} [/INST]"
    if "llama-2" in name or "llama2" in name:
        bos = (getattr(tok, "bos_token", None) or "<s>") if tok is not None else "<s>"
        return f"{bos}[INST] {text} [/INST] "
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"User: {text}\nAssistant:"


def difference_in_means(Hh: torch.Tensor, Hb: torch.Tensor) -> torch.Tensor:
    d = Hh.float().mean(0) - Hb.float().mean(0)
    return d / (d.norm() + 1e-12)


def safe_eigh(C: torch.Tensor):
    Cs = (0.5 * (C + C.T)).float()
    if torch.cuda.is_available():
        try:
            w, V = torch.linalg.eigh(Cs.cuda())
            return w.cpu(), V.cpu()
        except Exception:
            pass
    try:
        import numpy as np
        w, V = np.linalg.eigh(Cs.cpu().numpy())
        return torch.from_numpy(w), torch.from_numpy(V)
    except Exception:
        pass
    w, V = torch.linalg.eigh(Cs.cpu())
    return w, V


def local_healing_subspace(Hh: torch.Tensor, Hb: torch.Tensor, k: int,
                           anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    d = Hh.shape[1]
    k = int(max(1, min(k, d)))
    a = anchor if anchor is not None else difference_in_means(Hh, Hb)
    a = (a / (a.norm() + 1e-12)).float()
    if k == 1:
        return a.view(d, 1)
    P = torch.eye(d) - torch.outer(a, a)
    Hh0 = (Hh.float() - Hh.float().mean(0)) @ P.T
    Hb0 = (Hb.float() - Hb.float().mean(0)) @ P.T
    Cres = (Hh0.T @ Hh0) / max(1, Hh0.shape[0]) - (Hb0.T @ Hb0) / max(1, Hb0.shape[0])
    w, V = safe_eigh(Cres)
    order = torch.argsort(w, descending=True)
    extra = V[:, order[: k - 1]]
    U = torch.cat([a.view(d, 1), extra], dim=1)
    return R.orthonormalize(U)


@dataclass
class Backend:
    model: object
    hooks: H.RecoveryHooks
    device: str
    d: int
    n_layers: int
    U: torch.Tensor
    r: torch.Tensor
    harmful_ids: torch.Tensor
    harmful_mask: Optional[torch.Tensor]
    harmful_pos: object
    benign_ids: torch.Tensor
    benign_mask: Optional[torch.Tensor]
    benign_pos: object
    l0: int
    peak: int
    repair_layers: List[int]
    basin_layers: List[int] = field(default_factory=list)
    tag: str = "model"
    model_name: str = ""
    is_stub: bool = False
    tok: object = None
    harmful_prompts: Optional[List[str]] = None
    benign_prompts: Optional[List[str]] = None
    test_harmful_prompts: Optional[List[str]] = None
    test_benign_prompts: Optional[List[str]] = None
    inject_scale: float = 1.0
    Hh_all: Optional[torch.Tensor] = None
    Hb_all: Optional[torch.Tensor] = None
    k_gen: int = 6
    basin_info: Dict[str, object] = field(default_factory=dict)
    use_system_prompt: bool = False

    def format_chat(self, prompt: str) -> str:
        if self.use_system_prompt and self.tok is not None and hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        return format_chat_no_system(prompt, self.model_name or self.tag, tok=self.tok)

    def layer_subspace(self, l: int, k: Optional[int] = None) -> torch.Tensor:
        k = k or self.k_gen
        return local_healing_subspace(self.Hh_all[l], self.Hb_all[l], k=k).to(torch.float32)

    def layer_dim_vector(self, l: int) -> torch.Tensor:
        return difference_in_means(self.Hh_all[l], self.Hb_all[l]).to(torch.float32)

    def generator_energy(self, resid_stack: torch.Tensor) -> torch.Tensor:
        L, B, d = resid_stack.shape
        U = self.U.to(device=resid_stack.device, dtype=resid_stack.dtype)
        coeffs, _ = R.project_coeffs(resid_stack.reshape(-1, d), U)
        return coeffs.pow(2).sum(1).reshape(L, B).mean(1)


def build_stub_backend(d=96, n_layers=10, gen_rank=3,
                       repair_layers=(5, 6, 7), seed=1, n=64) -> Backend:
    from aor.experiments._stub_model import TinyTransformer
    torch.manual_seed(seed)
    model = TinyTransformer(vocab=64, d=d, n_layers=n_layers, gen_rank=gen_rank,
                            repair_layers=repair_layers, repair_gain=0.5, seed=seed).eval()
    U = R.orthonormalize(model.G)
    r = U[:, 0].clone()
    T = 6
    gr = gen_rank
    harmful = torch.randint(gr + 2, 64, (n, T))
    harmful[:, -1] = torch.randint(1, gr + 1, (n,))
    benign = torch.randint(gr + 2, 64, (n, T))
    benign[:, -1] = gr + 1
    hooks = H.RecoveryHooks(model)
    with hooks.capture(token_pos=-1):
        H.forward_pass(model, harmful)
        Hh_all = hooks.stacked("resid")
    with hooks.capture(token_pos=-1):
        H.forward_pass(model, benign)
        Hb_all = hooks.stacked("resid")
    return Backend(model=model, hooks=hooks, device="cpu", d=d, n_layers=n_layers,
                   U=U, r=r, harmful_ids=harmful, harmful_mask=None, harmful_pos=-1,
                   benign_ids=benign, benign_mask=None, benign_pos=-1,
                   l0=2, peak=2, repair_layers=list(repair_layers),
                   basin_layers=[2, 3, 4], tag="stub", model_name="stub", is_stub=True,
                   Hh_all=Hh_all, Hb_all=Hb_all, k_gen=gen_rank)


def _load_prompts(path: str, limit: Optional[int] = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            val = item.get("instruction") or item.get("prompt") or item.get("goal") or item.get("text") or ""
            if val:
                out.append(str(val).strip())
    out = [x for x in out if x]
    return out[:limit] if limit is not None else out


@torch.no_grad()
def _capture_resid_batched(model, hooks, ids, mask, pos, batch_size: int = 16) -> torch.Tensor:
    N = ids.shape[0]
    chunks = []
    for i in range(0, N, batch_size):
        ids_b = ids[i:i + batch_size]
        mask_b = mask[i:i + batch_size] if mask is not None else None
        pos_b = pos[i:i + batch_size] if isinstance(pos, torch.Tensor) else pos
        with hooks.capture(token_pos=pos_b):
            H.forward_pass(model, ids_b, attention_mask=mask_b)
            chunks.append(hooks.stacked("resid"))
    return torch.cat(chunks, dim=1)


def _select_dtype(dtype):
    if dtype is not None:
        return dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_real_backend(model_name: str, harmful_path: str, benign_path: str,
                       test_harmful_path: Optional[str] = None,
                       test_benign_path: Optional[str] = None,
                       device: str = "cuda", dtype=None,
                       n_train: int = 128, n_test: int = 150, k_max: int = 6,
                       batch_size: int = 16,
                       ablate_mode: str = "writes",
                       use_system_prompt: bool = False,
                       basin: Optional[Sequence[int]] = None,
                       peak: Optional[int] = None) -> Backend:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aor.interp import basin as BA

    use_dtype = _select_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=use_dtype, device_map=device).eval()
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Load train + strictly disjoint held-out test sets
    all_h_train = _load_prompts(harmful_path)
    all_b_train = _load_prompts(benign_path)
    h_prompts = all_h_train[:n_train]
    b_prompts = all_b_train[:n_train]
    train_h_set = set(h_prompts)
    train_b_set = set(b_prompts)

    if test_harmful_path:
        raw_test_h = _load_prompts(test_harmful_path)
        test_h_prompts = [p for p in raw_test_h if p not in train_h_set][:n_test]
    else:
        test_h_prompts = [p for p in all_h_train[n_train:] if p not in train_h_set][:n_test]
    if not test_h_prompts:
        raise ValueError(
            "Zero held-out test harmful prompts available! Provide --test_harmful "
            "or ensure harmful_train.json has > n_train prompts."
        )

    if test_benign_path:
        raw_test_b = _load_prompts(test_benign_path)
        test_b_prompts = [p for p in raw_test_b if p not in train_b_set][:n_test]
    else:
        test_b_prompts = [p for p in all_b_train[n_train:] if p not in train_b_set][:n_test]
    if not test_b_prompts:
        raise ValueError(
            "Zero held-out test benign prompts available! Provide --test_benign "
            "or ensure harmless_train.json has > n_train prompts."
        )

    assert len(set(h_prompts) & set(test_h_prompts)) == 0, "DATA LEAKAGE: train/test harmful overlap!"
    assert len(set(b_prompts) & set(test_b_prompts)) == 0, "DATA LEAKAGE: train/test benign overlap!"

    def _fmt(p: str) -> str:
        if use_system_prompt and hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
        return format_chat_no_system(p, model_name, tok=tok)

    def tok_batch(prompts):
        chats = [_fmt(p) for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=512, add_special_tokens=False)
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        pos = H.last_token_index(mask)
        return ids, mask, pos

    h_ids, h_mask, h_pos = tok_batch(h_prompts)
    b_ids, b_mask, b_pos = tok_batch(b_prompts)

    hooks = H.RecoveryHooks(model)
    Hh = _capture_resid_batched(model, hooks, h_ids, h_mask, h_pos, batch_size=batch_size)
    Hb = _capture_resid_batched(model, hooks, b_ids, b_mask, b_pos, batch_size=batch_size)

    geo_basin, geo_peak, _ = BA.geometric_basin(Hh, Hb, peak=peak)
    init_peak = int(peak) if peak is not None else geo_peak
    init_basin = sorted(int(x) for x in basin) if basin is not None else geo_basin

    r = difference_in_means(Hh[init_peak], Hb[init_peak])
    U = local_healing_subspace(Hh[init_peak], Hb[init_peak], k=k_max, anchor=r)

    hi_tmp = max(init_basin)
    repair_layers = list(range(hi_tmp + 1, min(n_layers, hi_tmp + 10))) or [n_layers - 1]
    inj = float(torch.stack([(Hh[l].mean(0) - Hb[l].mean(0)).norm() for l in repair_layers]).mean())

    bk = Backend(
        model=model, hooks=hooks, device=str(model.device), d=d, n_layers=n_layers,
        U=U.float().cpu(), r=r.float().cpu(),
        harmful_ids=h_ids, harmful_mask=h_mask, harmful_pos=h_pos,
        benign_ids=b_ids, benign_mask=b_mask, benign_pos=b_pos,
        l0=min(init_basin), peak=init_peak, repair_layers=repair_layers,
        basin_layers=init_basin, tag=model_name.split("/")[-1], model_name=model_name,
        is_stub=False, tok=tok, harmful_prompts=h_prompts, benign_prompts=b_prompts,
        test_harmful_prompts=test_h_prompts, test_benign_prompts=test_b_prompts,
        inject_scale=max(inj, 1e-6), Hh_all=Hh, Hb_all=Hb, k_gen=k_max,
        use_system_prompt=use_system_prompt,
    )

    # Automatic Behavioral Peak + Causal Basin discovery under `ablate_mode`
    beh_info = BA.discover_basin_behavioral(
        bk, n_calib=min(32, len(h_prompts)), batch_size=batch_size,
        ablate_mode=ablate_mode,
    )
    bk.basin_info = beh_info
    if peak is None:
        bk.peak = int(beh_info["peak"])
        bk.r = difference_in_means(Hh[bk.peak], Hb[bk.peak]).float().cpu()
        bk.U = local_healing_subspace(Hh[bk.peak], Hb[bk.peak], k=k_max, anchor=bk.r).float().cpu()
    if basin is None:
        bk.basin_layers = sorted(int(x) for x in beh_info["basin"])
        bk.l0 = min(bk.basin_layers)
        hi = max(bk.basin_layers)
        bk.repair_layers = list(range(hi + 1, min(n_layers, hi + 10))) or [n_layers - 1]
        bk.inject_scale = max(float(torch.stack(
            [(Hh[l].mean(0) - Hb[l].mean(0)).norm() for l in bk.repair_layers]).mean()), 1e-6)

    print(f">> [DATA] train_h={len(h_prompts)}  test_h={len(test_h_prompts)} (0% overlap) | "
          f"train_b={len(b_prompts)}  test_b={len(test_b_prompts)} (0% overlap) | "
          f"dtype={use_dtype} | sys_prompt={use_system_prompt} | mode={ablate_mode}")
    print(f">> [BASIN] ref_tokens={beh_info.get('refusal_tokens')} | "
          f"beh_peak={beh_info['peak']} (induce_lo={beh_info.get('peak_induce_log_odds')}, "
          f"drop={beh_info.get('peak_global_drop')}, kl={beh_info.get('peak_benign_kl')}) | "
          f"beh_basin={beh_info['basin']} | geo_basin={beh_info['geometric_basin']} | "
          f"IoU={beh_info['basin_iou']:.3f}")
    return bk