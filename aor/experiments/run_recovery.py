"""End-to-end mechanistic and intervention orchestrator.

Runs Exp2, Exp3, Exp4, Exp5 (discrete Frenet-Serret geometry and torsional
self-repair coupling), and Exp6 (writes-mode ablation, no system prompt, with
StrongREJECT mean, StrongREJECT binary >= 0.5, and non-refusal readouts plus
utility and forced-response controls). --only_exp6 skips Exp2-Exp4 when checking Exp6
quickly; Exp5 still runs, since it is fast and logs the Frenet and basin
diagnostics alongside Exp6."""
import sys, os, json, argparse, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from aor.interp import reco_backend as B
from aor.experiments import exp2_recovery_curve as Exp2
from aor.experiments import exp3_recovery_operator as Exp3
from aor.experiments import exp4_identification as Exp4
from aor.experiments import exp5_rrg as Exp5
from aor.experiments import exp6_geora as Exp6
from aor.experiments import exp1_sufficiency as Exp1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None, help="Train harmful JSON")
    ap.add_argument("--benign", default=None, help="Train harmless JSON")
    ap.add_argument("--test_harmful", default=None, help="HELD-OUT test harmful JSON")
    ap.add_argument("--test_benign", default=None, help="HELD-OUT test benign JSON")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--exp6_ranks", default="1,2", help="Comma-separated ablation ranks for Exp6")
    ap.add_argument("--ablate_mode", default="writes", choices=["writes", "full_resid", "resid"],
                    help="Ablation hook site for basin sweep & Exp6 (default: writes = Arditi Eq.5 / AOR)")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="Generation length for Exp6 StrongREJECT evaluation (256 or 512)")
    ap.add_argument("--use_system_prompt", action="store_true",
                    help="Keep default model system prompt (off by default per Arditi Table 6 / AOR)")
    ap.add_argument("--only_exp6", action="store_true",
                    help="Skip Exp2-Exp4 and run only Exp5 + Exp6 for fast Exp6 verification")
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_gen", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--peak", type=int, default=None)
    ap.add_argument("--basin", default=None)
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--inject", default="basin", choices=["basin", "all", "repair"])
    ap.add_argument("--run_exp1", action="store_true")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    basin = None
    if a.basin:
        lo, hi = a.basin.split("-")
        basin = list(range(int(lo), int(hi) + 1))
    exp6_ranks = tuple(int(x.strip()) for x in a.exp6_ranks.split(",") if x.strip())

    if a.selftest or a.model is None:
        print(">> building stub backend (offline self-test)")
        bk = B.build_stub_backend()
    else:
        print(f">> loading real backend: {a.model}")
        bk = B.build_real_backend(
            a.model, a.harmful, a.benign,
            test_harmful_path=a.test_harmful, test_benign_path=a.test_benign,
            device=a.device, k_max=a.k_max, n_train=a.n_train, n_test=a.n_gen,
            batch_size=a.batch_size, ablate_mode=a.ablate_mode,
            use_system_prompt=a.use_system_prompt, peak=a.peak, basin=basin,
        )
    print(f">> peak(l*)={bk.peak}  l0={bk.l0}  basin={bk.basin_layers}  "
          f"repair_band={bk.repair_layers}  inject_scale={bk.inject_scale:.3f}")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)

    def outpath(name):
        return os.path.join(a.outdir, f"{name}_{bk.tag}.json") if a.outdir else None

    def run_exp4_no_leak():
        orig_h = bk.harmful_prompts
        try:
            if bk.test_harmful_prompts:
                bk.harmful_prompts = bk.test_harmful_prompts
            return Exp4.run(bk, gold=a.gold, n_gen=a.n_gen, out=outpath("exp4"))
        finally:
            bk.harmful_prompts = orig_h

    results = {}
    phaseA = []
    if not a.only_exp6:
        phaseA.extend([
            ("exp2", lambda: Exp2.run(bk, out=outpath("exp2"))),
            ("exp3", lambda: Exp3.run(bk, out=outpath("exp3"))),
            ("exp4", run_exp4_no_leak),
        ])
    phaseA.extend([
        ("exp5", lambda: Exp5.run(bk, out=outpath("exp5"), batch_size=a.batch_size)),
        ("exp6", lambda: Exp6.run(bk, n_gen=a.n_gen, max_new_tokens=a.max_new_tokens,
                                ranks=exp6_ranks, ablate_mode=a.ablate_mode,
                                batch_size=a.batch_size, out=outpath("exp6"))),
    ])
    if a.run_exp1:
        phaseA.append(("exp1", lambda: Exp1.run(bk, where=a.inject, n_gen=a.n_gen,
                                              out=outpath("exp1"))))

    for name, fn in phaseA:
        print(f"\n################  {name.upper()}  ################")
        try:
            results[name] = fn()
        except Exception:
            print(f"!! {name} failed:\n{traceback.format_exc()}")
            results[name] = {"error": traceback.format_exc()}

    # ---- PHASE B: Free target LLM, load StrongREJECT once, score Exp4 + Exp6 ----
    exp4_jobs = results.get("exp4", {}).get("_gold_jobs") if isinstance(results.get("exp4"), dict) else None
    exp6_jobs = results.get("exp6", {}).get("_gold_jobs") if isinstance(results.get("exp6"), dict) else None
    if a.gold and (exp4_jobs or exp6_jobs):
        print("\n################  GOLD StrongREJECT (2-phase, held-out test)  ################")
        try:
            import gc, torch as _torch
            from aor.interp import reco_judge as JD
            try:
                bk.hooks.release()
            except Exception:
                pass
            m = bk.model; bk.model = None; bk.hooks = None
            del m; gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            flat_p, flat_c, spans, idx = [], [], {}, 0
            for k, (p, c) in (exp4_jobs or {}).items():
                spans[("exp4", k)] = (idx, idx + len(c)); idx += len(c); flat_p += list(p); flat_c += list(c)
            for cond, (p, c) in (exp6_jobs or {}).items():
                spans[("exp6", cond)] = (idx, idx + len(c)); idx += len(c); flat_p += list(p); flat_c += list(c)

            mode, scores = JD.strongreject_scores_safe(flat_p, flat_c)
            print(f"   (judge mode: {mode}, total held-out completions scored: {len(flat_c)})")

            if exp4_jobs:
                asr_by_k = {k: JD.asr_from_scores(scores[aa:bb], continuous=True)
                            for (tag, k), (aa, bb) in spans.items() if tag == "exp4"}
                results["exp4"] = Exp4.finalize_gold(results["exp4"], asr_by_k, out=outpath("exp4"))
            if exp6_jobs:
                asr_by_cond = {}
                raw_scores_by_cond = {}
                raw_comps_by_cond = {}
                for (tag, cond), (aa, bb) in spans.items():
                    if tag != "exp6":
                        continue
                    seg_s = list(scores[aa:bb])
                    seg_c = list(flat_c[aa:bb])
                    asr_by_cond[cond] = JD.summarize_condition_scores(seg_s, seg_c, threshold=0.5)
                    raw_scores_by_cond[cond] = seg_s
                    raw_comps_by_cond[cond] = seg_c
                results["exp6"] = Exp6.finalize_gold(
                    results["exp6"], asr_by_cond,
                    raw_scores_by_cond=raw_scores_by_cond,
                    raw_comps_by_cond=raw_comps_by_cond,
                    out=outpath("exp6"),
                )
        except Exception:
            print(f"!! gold scoring failed:\n{traceback.format_exc()}")

    # ---- summary ----
    print("\n================  SUITE SUMMARY  ================")
    try:
        exp5 = results.get("exp5", {})
        print(f"  [BASIN] peak={exp5.get('peak')} basin={exp5.get('basin')} "
              f"IoU={exp5.get('basin_info', {}).get('basin_iou')}")
        print(f"  [FRENET] mean_tau={exp5.get('mean_torsion_tau')} | "
              f"0p R^2: geo={exp5.get('zero_param_geodesic_r2')} vs exp={exp5.get('zero_param_exp_drift_r2')} | "
              f"1p R^2: geo={exp5.get('geodesic_r2')} vs exp={exp5.get('exp_drift_r2')}")
        exp6 = results.get("exp6", {})
        if isinstance(exp6.get("gold_asr"), dict):
            g = exp6["gold_asr"]
            print("  [Exp6 SR_Mean] " + " | ".join(f"{k}={v['asr']:.1f}%" for k, v in g.items()))
            print("  [Exp6 NonRef ] " + " | ".join(f"{k}={v.get('non_refusal_asr', 0):.1f}%" for k, v in g.items()))
    except Exception:
        print(traceback.format_exc())

    if a.outdir:
        dump = {k: ({kk: vv for kk, vv in v.items() if kk != "_gold_jobs"}
                    if isinstance(v, dict) else v) for k, v in results.items()}
        json.dump(dump, open(os.path.join(a.outdir, f"summary_{bk.tag}.json"), "w"), indent=2)
    print("done.")


if __name__ == "__main__":
    main()