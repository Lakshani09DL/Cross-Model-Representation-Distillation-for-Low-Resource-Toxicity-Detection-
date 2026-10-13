"""
CMRD component ablation -- TAMIL (DravidianCodeMix) only.

    CE  ->  CE + NCE  ->  CE + KL  ->  CE + NCE + KL (full CMRD)

This driver does NOT copy the training code. It imports cmrd-llama.py as a module,
overrides its config globals for each variant, and calls its _main_inner(run_dir).
Each variant runs in its own subprocess, so the 4-bit Llama is fully freed from the GPU
between runs and one crash does not kill the others.

Usage
    python tamil_cmrd_ablation.py                       # run all variants for every seed in SEEDS
    python tamil_cmrd_ablation.py --root <existing dir> # resume: finished runs are skipped
    python tamil_cmrd_ablation.py --summarize <dir>     # rebuild the summary table only

Output (one directory per ablation):
    <ABL_ROOT>/ablation_<timestamp>/
        s42_ce/ s42_ce_nce/ s42_ce_kl/ s42_ce_nce_kl/   (train_log.txt, test_predictions.npz,
                                                         config.pt, lora.pt, diagnostics.npz, ...)
        summary.csv  summary.md
"""

import argparse
import datetime
import importlib.util
import os
import subprocess
import sys
import traceback

# ============================================================================
# CONFIG
# ============================================================================

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmrd-llama.py")
ABL_ROOT = "/home/cse_g3/Subhagya/notebooks/condlora/ablation_llama32_3b_ta"

SEEDS = [42]          # add more (e.g. [42, 43, 44]) for seed variance; same SPLIT_SEED is kept
MAX_TRAIN = None      # identical for every variant; set e.g. 12000 to cut Tamil training time

# Everything not listed here is taken unchanged from cmrd-llama.py
# (layer 18, tau 0.07, teacher tau 0.05, false-neg thresh 0.85, LoRA r16, 3 epochs, ...).
LAMBDA = (0.3, 0.1)   # same aux-loss schedule for every aligned variant
VARIANTS = {
    #  name          lambda start/end   W_NCE  W_REL(KL)
    "ce":         dict(lam=(0.0, 0.0), w_nce=0.0, w_rel=0.0),   # baseline: no heads, no BGE loss
    "ce_nce":     dict(lam=LAMBDA,     w_nce=1.0, w_rel=0.0),
    "ce_kl":      dict(lam=LAMBDA,     w_nce=0.0, w_rel=1.0),
    "ce_nce_kl":  dict(lam=LAMBDA,     w_nce=1.0, w_rel=1.0),   # full CMRD
}
ORDER = ["ce", "ce_nce", "ce_kl", "ce_nce_kl"]                  # baseline MUST run first
LABELS = {"ce": "CE only (baseline)", "ce_nce": "CE + NCE",
          "ce_kl": "CE + KL", "ce_nce_kl": "CE + NCE + KL (CMRD)"}
KEY_METRICS = ["Accuracy", "Macro F1", "F1 (offensive)"]


# ============================================================================
# WORKER: one variant, one seed (runs inside its own subprocess)
# ============================================================================

def load_cmrd():
    spec = importlib.util.spec_from_file_location("cmrd_llama", SCRIPT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # the __main__ guard keeps it from training on import
    return m


def run_worker(variant, seed, run_dir, baseline_preds):
    import torch

    m = load_cmrd()
    cfg = VARIANTS[variant]

    # --- config overrides (the original functions read these globals at call time) ---
    m.DATASET = "tamil"
    m.SAVE_ROOT = ABL_ROOT
    m.RANDOM_SEED = seed
    m.MAX_TRAIN = MAX_TRAIN
    m.ALIGN_TARGET_MODE = "real"
    m.UNLABELED_PATH = None
    m.ALIGN_LAMBDA_START, m.ALIGN_LAMBDA_END = cfg["lam"]
    m.ALIGN_W_NCE = cfg["w_nce"]
    m.ALIGN_W_REL = cfg["w_rel"]
    m.ALIGN_W_COS = 0.0
    m.BASELINE_PREDS_PATH = baseline_preds

    # --- keep data order / dropout identical to the baseline ---
    # The baseline never builds AlignHeads; in aligned runs their random init would consume
    # the global CPU RNG and change the DataLoader shuffle order. Build them inside a forked
    # RNG so every variant sees the same LoRA init, the same batches, the same dropout masks.
    Orig = m.AlignHeads

    class RNGNeutralAlignHeads(Orig):
        def __init__(self, *a, **k):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 10_000)
                super().__init__(*a, **k)

    m.AlignHeads = RNGNeutralAlignHeads

    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train_log.txt"), "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = m.Tee(orig_stdout, log)
    try:
        print(f"ABLATION VARIANT: {variant} ({LABELS[variant]}) | seed {seed} | "
              f"W_NCE={cfg['w_nce']} W_KL={cfg['w_rel']} lambda={cfg['lam']} | "
              f"baseline_preds={baseline_preds}")
        m._main_inner(run_dir)
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        sys.stdout = orig_stdout
        log.close()


# ============================================================================
# DRIVER
# ============================================================================

def is_done(run_dir):
    return all(os.path.exists(os.path.join(run_dir, f)) for f in ("test_predictions.npz", "config.pt"))


def launch(variant, seed, run_dir, baseline_preds):
    if is_done(run_dir):
        print(f"[skip] {os.path.basename(run_dir)} already finished")
        return True
    cmd = [sys.executable, os.path.abspath(__file__), "--variant", variant,
           "--seed", str(seed), "--run_dir", run_dir]
    if baseline_preds:
        cmd += ["--baseline_preds", baseline_preds]
    print(f"\n>>> {LABELS[variant]} | seed {seed}\n    {run_dir}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"!!! {variant} seed {seed} failed (exit {rc}); see its train_log.txt")
    return rc == 0


def run_all(root):
    os.makedirs(root, exist_ok=True)
    for seed in SEEDS:
        base_dir = os.path.join(root, f"s{seed}_ce")
        if not launch("ce", seed, base_dir, None):
            print(f"Baseline failed for seed {seed}; skipping its aligned variants.")
            continue
        base_preds = os.path.join(base_dir, "test_predictions.npz")
        for v in ORDER[1:]:
            launch(v, seed, os.path.join(root, f"s{seed}_{v}"), base_preds)
    summarize(root)


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def _ci(r, signed=False):
    f = "{:+.4f}" if signed else "{:.4f}"
    s = f"{f.format(r['point'])} [{f.format(r['lo'])}, {f.format(r['hi'])}]"
    if signed and (r["lo"] > 0 or r["hi"] < 0):
        s += " *"                                   # 95% CI of the paired diff excludes 0
    return s


def summarize(root):
    import numpy as np
    import pandas as pd
    import torch

    rows = []
    for seed in SEEDS:
        for v in ORDER:
            p = os.path.join(root, f"s{seed}_{v}", "config.pt")
            if not os.path.exists(p):
                continue
            c = torch.load(p, weights_only=False)
            boot = c["bootstrap"]["results"]
            diff = c["bootstrap"]["diff_vs_baseline"]
            row = {"variant": LABELS[v], "key": v, "seed": seed,
                   "best_val_macro_f1": float(c["best_val_macro_f1"])}
            for k in KEY_METRICS:
                row[k] = boot[k]["point"]
                row[f"{k} (95% CI)"] = _ci(boot[k])
                row[f"Δ{k} vs CE (95% CI)"] = _ci(diff[k], signed=True) if diff else "—"
            rows.append(row)

    if not rows:
        print("No finished runs found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "summary.csv"), index=False)

    md = ["# CMRD ablation — Tamil (DravidianCodeMix test set)", "",
          "Point estimate [10k-bootstrap 95% CI]. Δ = paired bootstrap difference vs the CE "
          "baseline of the same seed; * = CI excludes 0.", ""]
    for seed in SEEDS:
        d = df[df.seed == seed]
        if d.empty:
            continue
        md += [f"## Seed {seed}", ""]
        cols = (["variant", "best_val_macro_f1"] + [f"{k} (95% CI)" for k in KEY_METRICS]
                + [f"Δ{k} vs CE (95% CI)" for k in KEY_METRICS])
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "---|" * len(cols))
        for _, r in d.iterrows():
            md.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                        for c in cols) + " |")
        md.append("")

    if df.seed.nunique() > 1:
        md += ["## Across seeds (mean ± std of test point estimates)", "",
               "| variant | n seeds | " + " | ".join(KEY_METRICS) + " |",
               "|" + "---|" * (2 + len(KEY_METRICS))]
        for v in ORDER:
            d = df[df.key == v]
            if d.empty:
                continue
            cells = [f"{d[k].mean():.4f} ± {d[k].std(ddof=1) if len(d) > 1 else 0:.4f}"
                     for k in KEY_METRICS]
            md.append(f"| {LABELS[v]} | {len(d)} | " + " | ".join(cells) + " |")
        md.append("")

    text = "\n".join(md)
    with open(os.path.join(root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Summary written to {root}/summary.csv and summary.md")


# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))       # worker mode (internal)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run_dir")
    ap.add_argument("--baseline_preds")
    ap.add_argument("--root", help="resume into an existing ablation directory")
    ap.add_argument("--summarize", metavar="DIR", help="only rebuild summary for DIR")
    a = ap.parse_args()

    if a.variant:
        run_worker(a.variant, a.seed, a.run_dir, a.baseline_preds)
    elif a.summarize:
        summarize(a.summarize)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_all(a.root or os.path.join(ABL_ROOT, f"ablation_{ts}"))
