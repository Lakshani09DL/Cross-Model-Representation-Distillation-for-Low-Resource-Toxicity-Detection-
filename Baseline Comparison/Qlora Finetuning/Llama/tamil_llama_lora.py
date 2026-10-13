"""
Tamil code-mixed offensive-language classification.
Llama-3.2-3B-Instruct + LoRA -- plain LoRA fine-tuning, no BGE-M3 conditioning.

Converted from the Gemma 4 E2B LoRA script:
  - Gemma processor / multimodal model / tower pruning replaced by a plain
    AutoTokenizer + AutoModelForCausalLM (Llama is text-only)
  - LoRA target regex changed from `language_model.layers` to `model.layers`
  - LoRA config (rank, alpha, dropout, projection modules) unchanged
  - Data paths / download logic taken from the llama-lora-only notebook
  - EPOCHS = 2, per-epoch validation classification report, in-memory best
    checkpoint (by val macro-F1), single run log file -- all unchanged
  - Bootstrap on the test set (N_BOOTSTRAP = 10,000) unchanged: point estimate,
    bootstrap mean +/- std and 95% percentile CI for every classification_report
    quantity, all at 5 decimal digits
"""

import gc
import os
import logging
import urllib.request
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb_opt


# ============================================================================
# CONFIG -- edit these directly
# ============================================================================

# Data paths as in the llama-lora-only notebook: the DravidianCodeMix zip is
# downloaded from GitHub and extracted to ./dravidian_data if it isn't there yet.
DATA_DIR = "dravidian_data"
ZIP_PATH = "DravidianCodeMix-2020.zip"
ZIP_URL = (
    "https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset"
    "/raw/main/DravidianCodeMix-2020.zip"
)

TRAIN_PATH = os.path.join(DATA_DIR, "DravidianCodeMix", "tamil_offensive_full_train.csv")
DEV_PATH = os.path.join(DATA_DIR, "DravidianCodeMix", "tamil_offensive_full_dev.csv")
TEST_PATH = os.path.join(DATA_DIR, "DravidianCodeMix", "tamil_offensive_full_test.csv")

# Only output of this script: a run log written to this folder.
LOG_DIR = os.path.join("logs", "llama-tamil")

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

RANDOM_SEED = 42

# Memory fix: much smaller micro-batch, more accumulation steps
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16  # effective batch size = 64 (4 x 16)

MAX_LEN = 128
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 0.01
COMPUTE_DTYPE = torch.bfloat16

PATIENCE = 3

# --- Bootstrap settings ---
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42

# LoRA config -- same rank / alpha / dropout / projection modules as before;
# only the module-path prefix changes for Llama (`model.layers`).
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_REGEX = (
    r".*model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))"
)

LABEL_MAP = {
    "Not_offensive": 0,
    "Offensive_Untargetede": 1,
    "Offensive_Untargeted": 1,
    "Offensive_Targeted_Insult_Individual": 1,
    "Offensive_Targeted_Insult_Group": 1,
    "Offensive_Targeted_Insult_Other": 1,
}

# --- FLOPs accounting (analytical estimate, standard transformer FLOPs-per-token
# convention from Kaplan et al.) ---
#   forward pass                ~= 2 * N_params * n_tokens
#   forward + backward (train)  ~= 6 * N_params * n_tokens
# N_params here is `all_param` (every active parameter the forward pass touches).
# LoRA doesn't shrink this: the frozen base weights are still multiplied on every
# forward/backward pass -- LoRA only restricts which parameters get *stored
# gradients*, not the number of multiply-accumulate ops in the pass itself.
# Likewise 4-bit quantization changes memory/throughput, not the FLOP count.
FLOPS_PER_PARAM_FWD = 2
FLOPS_PER_PARAM_TRAIN = 6
 
# --- Inference cost benchmarking config ---
RUN_INFERENCE_BENCHMARK = True   # latency / throughput / memory / TFLOP-s on the test set
RUN_BATCH_SIZE_SWEEP = True      # repeat the benchmark across several batch sizes
BENCHMARK_WARMUP_BATCHES = 3     # excluded from timing (kernel/cache warm-up)
BENCHMARK_MAX_BATCHES = 20       # cap per batch-size setting, so the sweep stays quick
SWEEP_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
 
# Optional: your GPU's peak bf16/fp16 TFLOP/s (e.g. ~989 for H100 SXM, ~312 for A100 80GB
# SXM, ~65 for a T4). If set, the benchmark also reports MFU (Model FLOPs Utilization =
# achieved TFLOP/s / this peak) as a hardware-efficiency sanity check. Leave as None to skip.
GPU_PEAK_TFLOPS = None
 
 
def format_flops(flops):
    """Human-readable FLOPs as GFLOPs / TFLOPs / PFLOPs, whichever fits best."""
    if flops >= 1e15:
        return f"{flops / 1e15:.4f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.4f} TFLOPs"
    else:
        return f"{flops / 1e9:.4f} GFLOPs"


# ============================================================================
# LOGGING SETUP -- the single output artifact of this script
# ============================================================================

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"tamil_llama32_lora_{timestamp}.log")

    logger = logging.getLogger("tamil_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger, log_path


# ============================================================================
# DATA LOADING
# ============================================================================

def ensure_dataset(logger):
    """Download + extract DravidianCodeMix only if the split files are missing."""
    if all(os.path.exists(p) for p in (TRAIN_PATH, DEV_PATH, TEST_PATH)):
        return

    if not os.path.exists(ZIP_PATH):
        logger.info("Downloading DravidianCodeMix-2020.zip from GitHub...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        logger.info("Download complete.")

    if not os.path.exists(DATA_DIR):
        logger.info("Extracting archive...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(DATA_DIR)
        logger.info("Extraction complete.")

    for p in (TRAIN_PATH, DEV_PATH, TEST_PATH):
        assert os.path.exists(p), f"Missing expected file: {p}"


def load_and_clean(path, split_name, logger):
    """Load a tab-delimited Tamil offensive split and preprocess."""
    df = pd.read_csv(path, sep="\t", header=None, on_bad_lines="skip")

    df_clean = df[[0, 1]].copy()
    df_clean.columns = ["comment", "category"]
    df_clean = df_clean.dropna(subset=["comment", "category"]).reset_index(drop=True)

    df_filtered = df_clean[df_clean["category"] != "not-Tamil"].copy()
    df_filtered["label"] = df_filtered["category"].map(LABEL_MAP)

    unmapped = df_filtered[df_filtered["label"].isna()]["category"].unique()
    if len(unmapped) > 0:
        logger.info(f"  WARNING: Unmapped categories in {split_name}: {list(unmapped)}")

    df_filtered = df_filtered.dropna(subset=["label"]).copy()
    df_filtered["label"] = df_filtered["label"].astype(int)

    logger.info(f"--- {split_name} ---")
    logger.info(f"Raw rows: {len(df)} | After cleaning: {len(df_filtered)}")
    logger.info(f"Class distribution:\n{df_filtered['label'].value_counts()}")

    return df_filtered["comment"].astype(str).tolist(), df_filtered["label"].values


# ============================================================================
# DATASET
# ============================================================================

class TamilDataset(Dataset):
    """Plain text classification dataset -- no BGE embeddings attached."""

    def __init__(self, sentences, labels, tokenizer, bos_id, id0, id1, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.bos_id = bos_id
        self.id0 = id0
        self.id1 = id1
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = (
            'Classify the following text as either '
            '"not offensive" (0) or "offensive" (1). '
            'Answer with only the number.\n\n'
            f'Text: {self.sentences[idx]}\n'
            'Answer: '
        )
        text_ids = self.tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=self.max_len,
        ).input_ids

        if self.bos_id is not None:
            text_ids = [self.bos_id] + text_ids

        target_id = self.id1 if int(self.labels[idx]) == 1 else self.id0

        return {
            "input_ids": text_ids,
            "attention_mask": [1] * len(text_ids),
            "target_id": target_id,
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, target_ids = [], [], []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad_len + x["input_ids"])
            attention_mask.append([0] * pad_len + x["attention_mask"])
            target_ids.append(x["target_id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    return collate_fn


# ============================================================================
# FORWARD PASS HELPERS (plain LoRA -- standard HF forward)
# ============================================================================

def run_batch(batch, model, id0, id1, class_weights_tensor, return_preds=False):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    target_ids = batch["target_ids"].to(model.device)

    with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        last_logits = logits[:, -1, :]

    binary_logits = last_logits[:, [id0, id1]]
    binary_targets = (target_ids == id1).long()

    loss = F.cross_entropy(binary_logits.float(), binary_targets, weight=class_weights_tensor)

    if return_preds:
        preds = (last_logits.float()[:, id1] > last_logits.float()[:, id0]).long()
        return loss, preds.cpu().numpy(), binary_targets.cpu().numpy()

    return loss


def evaluate(loader, desc, model, id0, id1, class_weights_tensor):
    """Run a full pass, returning (avg_loss, macro_f1, accuracy, preds, targets)."""
    model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.no_grad():
        for batch in bar:
            loss, preds, targets = run_batch(
                batch, model, id0, id1, class_weights_tensor, return_preds=True,
            )
            total_loss += loss.item()
            all_preds.extend(preds)
            all_targets.extend(targets)
            bar.set_postfix(loss=f"{loss.item():.4f}")
            del loss

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, macro_f1, acc, all_preds, all_targets


# ============================================================================
# INFERENCE COST BENCHMARKING
# ============================================================================
# Note: this model does a single forward pass per example (last-token logits
# for a 0/1 classification), not autoregressive generation -- so the usual
# generation-serving metrics (time-to-first-token, time-per-output-token) do
# not apply here. What matters for this kind of workload is per-batch/
# per-sample latency, throughput, memory footprint, and how those scale with
# batch size.
 
def _try_get_gpu_power_w():
    """Best-effort instantaneous GPU power draw in Watts via pynvml. Returns
    None (silently) if pynvml isn't installed or there's no NVIDIA GPU --
    power/energy reporting is a nice-to-have, not a hard requirement."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        return power_mw / 1000.0
    except Exception:
        return None
 
 
def benchmark_inference(loader, model, id0, id1, class_weights_tensor, all_param, logger,
                         n_warmup=BENCHMARK_WARMUP_BATCHES, n_batches=None):
    """
    Wall-clock inference benchmarking: per-batch latency (mean/std/p50/p90/p99),
    throughput (samples/sec, tokens/sec), peak GPU memory, achieved compute
    rate (TFLOP/s, via the same 2N-per-token FLOPs convention used elsewhere
    in this script), optional MFU vs. GPU_PEAK_TFLOPS, and a best-effort
    power/energy estimate. Uses CUDA events for accurate GPU-side timing and
    excludes a warm-up phase so first-call kernel/cache effects don't bias
    the numbers.
    """
    model.eval()
    device = model.device
 
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
 
    batches = list(loader)
    if n_batches is not None:
        batches = batches[:n_batches]
    n_warmup = min(n_warmup, max(0, len(batches) - 1))
 
    with torch.no_grad():
        for batch in batches[:n_warmup]:
            run_batch(batch, model, id0, id1, class_weights_tensor, return_preds=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
 
    timed_batches = batches[n_warmup:]
    per_batch_ms, per_batch_samples, per_batch_tokens, power_samples = [], [], [], []
 
    with torch.no_grad():
        for batch in tqdm(timed_batches, desc="Inference benchmark", leave=False, unit="batch"):
            n_samples = batch["input_ids"].shape[0]
            n_tok = batch["attention_mask"].sum().item()
 
            power_w = _try_get_gpu_power_w()
            if power_w is not None:
                power_samples.append(power_w)
 
            if torch.cuda.is_available():
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run_batch(batch, model, id0, id1, class_weights_tensor, return_preds=True)
                end_evt.record()
                torch.cuda.synchronize()
                elapsed_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.perf_counter()
                run_batch(batch, model, id0, id1, class_weights_tensor, return_preds=True)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
 
            per_batch_ms.append(elapsed_ms)
            per_batch_samples.append(n_samples)
            per_batch_tokens.append(n_tok)
 
    per_batch_ms = np.array(per_batch_ms)
    per_batch_samples = np.array(per_batch_samples)
    per_batch_tokens = np.array(per_batch_tokens)
 
    total_ms = per_batch_ms.sum()
    total_samples = int(per_batch_samples.sum())
    total_tokens = int(per_batch_tokens.sum())
 
    per_sample_ms = per_batch_ms / per_batch_samples
    p50, p90, p99 = np.percentile(per_batch_ms, [50, 90, 99])
 
    throughput_samples_per_sec = total_samples / (total_ms / 1000.0)
    throughput_tokens_per_sec = total_tokens / (total_ms / 1000.0)
 
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else None
 
    inference_flops = FLOPS_PER_PARAM_FWD * all_param * total_tokens
    achieved_tflops_per_sec = (inference_flops / (total_ms / 1000.0)) / 1e12
 
    avg_power_w = float(np.mean(power_samples)) if power_samples else None
    energy_j = avg_power_w * (total_ms / 1000.0) if avg_power_w is not None else None
    energy_wh = energy_j / 3600.0 if energy_j is not None else None
 
    logger.info(f"Batches benchmarked: {len(timed_batches)} (warm-up batches excluded: {n_warmup})")
    logger.info(f"Total samples: {total_samples:,} | Total tokens: {total_tokens:,}")
    logger.info(f"Total wall-clock time: {total_ms / 1000.0:.4f} s")
    logger.info(f"Mean batch latency: {per_batch_ms.mean():.3f} ms | Std: {per_batch_ms.std():.3f} ms")
    logger.info(f"Batch latency p50 / p90 / p99: {p50:.3f} / {p90:.3f} / {p99:.3f} ms")
    logger.info(f"Mean per-sample latency: {per_sample_ms.mean():.3f} ms")
    logger.info(
        f"Throughput: {throughput_samples_per_sec:.2f} samples/sec | "
        f"{throughput_tokens_per_sec:.2f} tokens/sec"
    )
    if peak_mem_gb is not None:
        logger.info(f"Peak GPU memory during inference: {peak_mem_gb:.3f} GB")
    logger.info(f"Inference FLOPs (this benchmark run): {format_flops(inference_flops)}")
    logger.info(f"Achieved compute rate: {achieved_tflops_per_sec:.4f} TFLOP/s")
    if GPU_PEAK_TFLOPS:
        mfu = achieved_tflops_per_sec / GPU_PEAK_TFLOPS * 100
        logger.info(f"MFU (Model FLOPs Utilization, vs. {GPU_PEAK_TFLOPS} TFLOP/s peak): {mfu:.2f}%")
    if avg_power_w is not None:
        logger.info(f"Avg GPU power draw (sampled, best-effort): {avg_power_w:.2f} W")
        logger.info(f"Estimated energy for this benchmark: {energy_j:.2f} J ({energy_wh:.6f} Wh)")
    else:
        logger.info("GPU power/energy sampling unavailable (pynvml not installed, or no NVIDIA GPU) -- skipped.")
 
    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_ms": float(total_ms),
        "mean_batch_ms": float(per_batch_ms.mean()),
        "std_batch_ms": float(per_batch_ms.std()),
        "p50_ms": float(p50), "p90_ms": float(p90), "p99_ms": float(p99),
        "mean_per_sample_ms": float(per_sample_ms.mean()),
        "throughput_samples_per_sec": float(throughput_samples_per_sec),
        "throughput_tokens_per_sec": float(throughput_tokens_per_sec),
        "peak_mem_gb": peak_mem_gb,
        "inference_flops": float(inference_flops),
        "achieved_tflops_per_sec": float(achieved_tflops_per_sec),
        "avg_power_w": avg_power_w,
        "energy_j": energy_j,
        "energy_wh": energy_wh,
    }
 
 
def benchmark_batch_size_sweep(test_dataset, collate_fn, model, id0, id1, class_weights_tensor,
                                all_param, logger, batch_sizes=SWEEP_BATCH_SIZES,
                                max_batches=BENCHMARK_MAX_BATCHES):
    """
    Sweep across batch sizes to see how inference latency/throughput/memory
    scale -- the key input for picking a serving batch size. Runs at most
    `max_batches` per setting (capped by dataset size) to keep this quick.
    """
    logger.info("=" * 80)
    logger.info("BATCH SIZE SWEEP (inference latency / throughput vs. batch size)")
    logger.info("=" * 80)
 
    results = []
    for bs in batch_sizes:
        loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
        n_batches = min(max_batches, len(loader))
        if n_batches <= 1:
            logger.info(f"Batch size {bs}: not enough batches to benchmark, skipping.")
            continue
        logger.info(f"-- Batch size {bs} ({n_batches} batches) --")
        stats = benchmark_inference(
            loader, model, id0, id1, class_weights_tensor, all_param, logger,
            n_warmup=min(2, n_batches - 1), n_batches=n_batches,
        )
        stats["batch_size"] = bs
        results.append(stats)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    logger.info("-" * 80)
    logger.info("Batch size sweep summary:")
    logger.info(f"{'Batch size':<12}{'Throughput (samp/s)':<24}{'Mean latency (ms)':<22}{'Peak mem (GB)':<16}")
    for r in results:
        mem_str = f"{r['peak_mem_gb']:.3f}" if r["peak_mem_gb"] is not None else "n/a"
        logger.info(
            f"{r['batch_size']:<12}{r['throughput_samples_per_sec']:<24.2f}"
            f"{r['mean_batch_ms']:<22.3f}{mem_str:<16}"
        )
 
    return results


# ============================================================================
# BOOTSTRAP CLASSIFICATION REPORT
# ============================================================================

# Metric keys in the same shape as a 2-class classification_report:
#   per-class precision/recall/f1/support for class 0 and class 1,
#   accuracy, and macro-avg / weighted-avg precision/recall/f1.
_METRIC_KEYS = [
    "precision_0", "recall_0", "f1_0", "support_0",
    "precision_1", "recall_1", "f1_1", "support_1",
    "accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]


def _report_row(y_true, y_pred):
    """Compute one full set of classification-report-style metrics."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0
    )
    return {
        "precision_0": p[0], "recall_0": r[0], "f1_0": f[0], "support_0": s[0],
        "precision_1": p[1], "recall_1": r[1], "f1_1": f[1], "support_1": s[1],
        "accuracy": acc,
        "macro_precision": pm, "macro_recall": rm, "macro_f1": fm,
        "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": fw,
    }


def bootstrap_classification_report(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Nonparametric bootstrap over the test set predictions.

    Resamples (y_true, y_pred) pairs WITH replacement n_boot times (same
    size as the original test set each time), recomputes the full set of
    classification-report metrics on each resample, and returns:
      - point:   the metrics computed once on the full (un-resampled) test set
      - dists:   dict of key -> np.array of length n_boot (the bootstrap distribution)
      - mean:    dict of key -> bootstrap mean
      - std:     dict of key -> bootstrap standard deviation
      - ci_lo/ci_hi: dict of key -> 2.5th / 97.5th percentile (95% percentile CI),
                      reported alongside mean +/- std for reference
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    assert n == len(y_pred), "y_true and y_pred must be the same length"

    rng = np.random.default_rng(seed)

    dists = {k: np.empty(n_boot, dtype=np.float64) for k in _METRIC_KEYS}

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        row = _report_row(y_true[idx], y_pred[idx])
        for k in _METRIC_KEYS:
            dists[k][i] = row[k]

    point = _report_row(y_true, y_pred)
    mean = {k: float(np.mean(dists[k])) for k in _METRIC_KEYS}
    std = {k: float(np.std(dists[k], ddof=1)) for k in _METRIC_KEYS}
    ci_lo = {k: float(np.percentile(dists[k], 2.5)) for k in _METRIC_KEYS}
    ci_hi = {k: float(np.percentile(dists[k], 97.5)) for k in _METRIC_KEYS}

    return {
        "point": point,
        "dists": dists,
        "mean": mean,
        "std": std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def format_bootstrap_report(boot_result, n_boot=N_BOOTSTRAP, digits=5):
    """
    Render the bootstrap result as a classification_report-shaped table,
    with every value at `digits` decimal places, each cell shown as:
        point_estimate  (bootstrap_mean +/- bootstrap_std)  [95% CI: lo, hi]
    """
    point = boot_result["point"]
    mean = boot_result["mean"]
    std = boot_result["std"]
    ci_lo = boot_result["ci_lo"]
    ci_hi = boot_result["ci_hi"]

    def fmt(k):
        return (
            f"{point[k]:.{digits}f}  "
            f"({mean[k]:.{digits}f} +/- {std[k]:.{digits}f})  "
            f"[95% CI: {ci_lo[k]:.{digits}f}, {ci_hi[k]:.{digits}f}]"
        )

    lines = []
    lines.append(f"Bootstrap resamples: N = {n_boot}")
    lines.append(
        "Each cell = point_estimate_on_full_test_set  "
        "(bootstrap_mean +/- bootstrap_std)  [95% percentile CI]"
    )
    lines.append("-" * 100)
    header = f"{'':<14}{'precision':<40}{'recall':<40}{'f1-score':<40}{'support':<20}"
    lines.append(header)

    lines.append(
        f"{'Non-toxic':<14}"
        f"{fmt('precision_0'):<40}"
        f"{fmt('recall_0'):<40}"
        f"{fmt('f1_0'):<40}"
        f"{fmt('support_0'):<20}"
    )
    lines.append(
        f"{'Toxic':<14}"
        f"{fmt('precision_1'):<40}"
        f"{fmt('recall_1'):<40}"
        f"{fmt('f1_1'):<40}"
        f"{fmt('support_1'):<20}"
    )
    lines.append("")
    lines.append(f"{'accuracy':<14}{fmt('accuracy')}")
    lines.append(
        f"{'macro avg':<14}"
        f"{fmt('macro_precision'):<40}"
        f"{fmt('macro_recall'):<40}"
        f"{fmt('macro_f1'):<40}"
    )
    lines.append(
        f"{'weighted avg':<14}"
        f"{fmt('weighted_precision'):<40}"
        f"{fmt('weighted_recall'):<40}"
        f"{fmt('weighted_f1'):<40}"
    )
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

# def main():
#     logger, log_path = setup_logger()
#     logger.info("=" * 80)
#     logger.info("TAMIL LLAMA-3.2-3B LORA -- normal LoRA run (no BGE conditioning)")
#     logger.info(f"Log file: {log_path}")
#     logger.info("=" * 80)

#     torch.manual_seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(RANDOM_SEED)

#     # Llama is gated: log in if a token is provided (otherwise cached credentials are used)
#     hf_token = "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"
#     if hf_token:
#         from huggingface_hub import login
#         login(token=hf_token)

#     # ---------------- STAGE 1: DATA ----------------
#     ensure_dataset(logger)

#     X_train, y_train = load_and_clean(TRAIN_PATH, "TRAIN", logger)
#     X_val, y_val = load_and_clean(DEV_PATH, "DEV", logger)
#     X_test, y_test = load_and_clean(TEST_PATH, "TEST", logger)

#     logger.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

#     # ---------------- STAGE 2: LOAD LLAMA-3.2-3B-INSTRUCT ----------------
#     logger.info("=" * 80)
#     logger.info("STAGE 2: LLAMA-3.2-3B-INSTRUCT + LoRA")
#     logger.info("=" * 80)

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=COMPUTE_DTYPE,
#         bnb_4bit_use_double_quant=True,
#     )

#     logger.info("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     logger.info("Loading Llama-3.2-3B-Instruct...")
#     model = AutoModelForCausalLM.from_pretrained(
#         LLAMA_MODEL,
#         quantization_config=bnb_config,
#         device_map={"": 0},
#         dtype=COMPUTE_DTYPE,
#     )

#     logger.info(f"Model loaded: {type(model).__name__}")
#     if torch.cuda.is_available():
#         logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

#     # (No vision/audio towers to prune -- Llama is a text-only checkpoint.)
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # Gradient checkpointing
#     logger.info("Enabling gradient checkpointing...")
#     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#     model.enable_input_require_grads()

#     norm_upcast_count = 0
#     for name, module in model.named_modules():
#         cls_name = type(module).__name__.lower()
#         if "norm" in cls_name and hasattr(module, "weight") and module.weight is not None:
#             module.to(torch.float32)
#             norm_upcast_count += 1
#     logger.info(f"Upcast {norm_upcast_count} norm layers to fp32")

#     # Token IDs for "0" / "1" answer classification
#     id0 = tokenizer("0", add_special_tokens=False).input_ids[0]
#     id1 = tokenizer("1", add_special_tokens=False).input_ids[0]
#     bos_id = tokenizer.bos_token_id
#     logger.info(f"Token '0': {id0}")
#     logger.info(f"Token '1': {id1}")
#     logger.info(f"BOS: {bos_id}")

#     # LoRA -- same rank, alpha, dropout, projection modules
#     logger.info("Injecting LoRA (attention + MLP only)...")
#     lora_config = LoraConfig(
#         r=LORA_R,
#         lora_alpha=LORA_ALPHA,
#         lora_dropout=LORA_DROPOUT,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=LORA_TARGET_REGEX,
#     )
#     model = get_peft_model(model, lora_config)

#     # print_trainable_parameters() writes straight to stdout -- capture it too
#     trainable_params, all_param = 0, 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     logger.info(
#         f"trainable params: {trainable_params:,} || all params: {all_param:,} "
#         f"|| trainable%: {100 * trainable_params / all_param:.4f}"
#     )

#     lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
#     non_lora_leaks = [n for n in lora_param_names if "lora" not in n]
#     logger.info(f"Trainable LoRA params: {len(lora_param_names)}")
#     logger.info(f"Trainable params that are not LoRA: {len(non_lora_leaks)}")
#     if non_lora_leaks:
#         logger.info(f"  WARNING -- these should not be here: {non_lora_leaks[:5]}")

#     # ---------------- DATASETS / LOADERS ----------------
#     collate_fn = make_collate_fn(tokenizer)

#     train_dataset = TamilDataset(X_train, y_train, tokenizer, bos_id, id0, id1, MAX_LEN)
#     val_dataset = TamilDataset(X_val, y_val, tokenizer, bos_id, id0, id1, MAX_LEN)

#     train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#     class_counts = np.bincount(y_train)
#     class_weights_tensor = torch.tensor(
#         [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32, device=model.device,
#     )
#     class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()
#     logger.info(f"Class weights: {class_weights_tensor}")

#     # ---------------- OPTIMIZER (LoRA params only) ----------------
#     lora_params = [p for p in model.parameters() if p.requires_grad]

#     optimizer = bnb_opt.optim.AdamW8bit(
#         [{"params": lora_params, "lr": LR}],
#         weight_decay=WEIGHT_DECAY,
#     )

#     logger.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}")

#     # ---------------- GRADIENT FLOW CHECK ----------------
#     model.train()

#     optimizer.zero_grad(set_to_none=True)
#     probe_batch = next(iter(train_loader))
#     probe_loss = run_batch(probe_batch, model, id0, id1, class_weights_tensor)
#     probe_loss.backward()

#     lora_grad_norms = [p.grad.norm().item() for p in lora_params if p.grad is not None]
#     logger.info(f"LoRA params with grad: {len(lora_grad_norms)} / {len(lora_params)}")
#     if lora_grad_norms:
#         logger.info(f"Mean LoRA grad norm: {np.mean(lora_grad_norms):.6f}")
#     else:
#         logger.info("  (none)")

#     optimizer.zero_grad(set_to_none=True)
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # ---------------- TRAINING LOOP ----------------
#     # Best checkpoint kept in memory only (by val macro-F1) so the final
#     # test pass uses the best epoch's weights. Nothing is written to disk --
#     # the log file is the only output of this script.
#     best_val_macro_f1 = -np.inf
#     best_state = None
#     epochs_no_improve = 0

#     epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

#     for epoch in epoch_bar:
#         # ---- TRAIN ----
#         model.train()

#         total_train_loss = 0.0
#         optimizer.zero_grad(set_to_none=True)

#         train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False, unit="batch")

#         for step, batch in enumerate(train_bar):
#             loss = run_batch(batch, model, id0, id1, class_weights_tensor)
#             scaled_loss = loss / GRAD_ACCUM_STEPS
#             scaled_loss.backward()

#             total_train_loss += loss.item()

#             if (step + 1) % GRAD_ACCUM_STEPS == 0:
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             train_bar.set_postfix(loss=f"{loss.item():.4f}")
#             del loss, scaled_loss

#         if len(train_loader) % GRAD_ACCUM_STEPS != 0:
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)

#         train_loss = total_train_loss / len(train_loader)

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # ---- VALIDATION ----
#         val_loss, val_macro_f1, val_acc, val_preds, val_targets = evaluate(
#             val_loader, f"Epoch {epoch + 1}/{EPOCHS} [val]",
#             model, id0, id1, class_weights_tensor,
#         )

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         epoch_bar.set_postfix(
#             train_loss=f"{train_loss:.4f}",
#             val_loss=f"{val_loss:.4f}",
#             val_macro_f1=f"{val_macro_f1:.4f}",
#         )

#         logger.info(
#             f"Epoch {epoch + 1}/{EPOCHS} | "
#             f"Train loss: {train_loss:.4f} | "
#             f"Val loss: {val_loss:.4f} | "
#             f"Val macro-F1: {val_macro_f1:.4f} | "
#             f"Val acc: {val_acc:.4f}"
#         )

#         # ---- CLASSIFICATION REPORT (every epoch) ----
#         report = classification_report(
#             val_targets, val_preds, target_names=["Non-toxic", "Toxic"], digits=5
#         )
#         logger.info(f"Validation classification report -- epoch {epoch + 1}:\n{report}")

#         # ---- TRACK BEST (in memory only, by macro-F1) ----
#         if val_macro_f1 > best_val_macro_f1:
#             best_val_macro_f1 = val_macro_f1
#             best_state = {
#                 k: v.detach().cpu().clone()
#                 for k, v in model.state_dict().items()
#                 if "lora" in k
#             }
#             epochs_no_improve = 0
#             logger.info(f"New best validation macro-F1: {best_val_macro_f1:.4f}")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= PATIENCE:
#                 logger.info("Early stopping.")
#                 break

#     # ---------------- RELOAD BEST CHECKPOINT (in memory, by macro-F1) ----------------
#     if best_state is not None:
#         logger.info(f"Reloading best in-memory checkpoint (val macro-F1: {best_val_macro_f1:.4f}) before test...")
#         model.load_state_dict(best_state, strict=False)
#     else:
#         logger.info("No improved checkpoint was ever recorded -- using final epoch weights.")

#     # ---------------- TEST ----------------
#     test_dataset = TamilDataset(X_test, y_test, tokenizer, bos_id, id0, id1, MAX_LEN)
#     test_loader = DataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#     logger.info("=" * 80)
#     logger.info("TEST EVALUATION")
#     logger.info("=" * 80)

#     _, test_macro_f1, test_acc, test_preds, test_targets = evaluate(
#         test_loader, "Evaluating", model, id0, id1, class_weights_tensor,
#     )

#     test_preds = np.array(test_preds)
#     test_targets = np.array(test_targets)

#     acc = accuracy_score(test_targets, test_preds)
#     f1 = f1_score(test_targets, test_preds)

#     logger.info("=" * 80)
#     logger.info("FINAL RESULTS")
#     logger.info("=" * 80)
#     logger.info(f"Accuracy:   {acc:.5f}")
#     logger.info(f"F1 (Toxic): {f1:.5f}")
#     logger.info(f"Macro-F1:   {test_macro_f1:.5f}")

#     report = classification_report(
#         test_targets, test_preds, target_names=["Non-toxic", "Toxic"], digits=5
#     )
#     logger.info(f"Test classification report (point estimate, 5 digits):\n{report}")

#     # ---------------- BOOTSTRAP CI ON TEST SET ----------------
#     logger.info("=" * 80)
#     logger.info(f"BOOTSTRAP RESAMPLING ON TEST SET (N = {N_BOOTSTRAP})")
#     logger.info("=" * 80)

#     boot_result = bootstrap_classification_report(
#         test_targets, test_preds, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
#     )

#     boot_report_str = format_bootstrap_report(boot_result, n_boot=N_BOOTSTRAP, digits=5)
#     logger.info(f"Bootstrap classification report (mean +/- std, 5 digits):\n{boot_report_str}")

#     # Also dump every single metric explicitly, one per line, at 5 digits,
#     # so nothing is left implicit in the table formatting above.
#     logger.info("-" * 80)
#     logger.info("Bootstrap metrics, one per line (point | mean +/- std | 95% CI), 5 digits:")
#     label_map = {
#         "precision_0": "Non-toxic precision",
#         "recall_0": "Non-toxic recall",
#         "f1_0": "Non-toxic f1-score",
#         "support_0": "Non-toxic support",
#         "precision_1": "Toxic precision",
#         "recall_1": "Toxic recall",
#         "f1_1": "Toxic f1-score",
#         "support_1": "Toxic support",
#         "accuracy": "Accuracy",
#         "macro_precision": "Macro avg precision",
#         "macro_recall": "Macro avg recall",
#         "macro_f1": "Macro avg f1-score",
#         "weighted_precision": "Weighted avg precision",
#         "weighted_recall": "Weighted avg recall",
#         "weighted_f1": "Weighted avg f1-score",
#     }
#     for key, label in label_map.items():
#         pt = boot_result["point"][key]
#         mn = boot_result["mean"][key]
#         sd = boot_result["std"][key]
#         lo = boot_result["ci_lo"][key]
#         hi = boot_result["ci_hi"][key]
#         logger.info(
#             f"{label:<24}: point={pt:.5f} | mean +/- std = {mn:.5f} +/- {sd:.5f} "
#             f"| 95% CI = [{lo:.5f}, {hi:.5f}]"
#         )

#     logger.info(f"Run complete. Log saved to: {log_path}")


def main():
    logger, log_path = setup_logger()
    logger.info("=" * 80)
    logger.info("TAMIL LLAMA-3.2-3B LORA -- normal LoRA run (no BGE conditioning)")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)
 
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
 
    # Llama is gated: log in if a token is provided (otherwise cached credentials are used)
    hf_token = "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"
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
 
    # ---------------- STAGE 1: DATA ----------------
    ensure_dataset(logger)
 
    X_train, y_train = load_and_clean(TRAIN_PATH, "TRAIN", logger)
    X_val, y_val = load_and_clean(DEV_PATH, "DEV", logger)
    X_test, y_test = load_and_clean(TEST_PATH, "TEST", logger)
 
    logger.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
 
    # ---------------- STAGE 2: LOAD LLAMA-3.2-3B-INSTRUCT ----------------
    logger.info("=" * 80)
    logger.info("STAGE 2: LLAMA-3.2-3B-INSTRUCT + LoRA")
    logger.info("=" * 80)
 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
 
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    logger.info("Loading Llama-3.2-3B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=COMPUTE_DTYPE,
    )
 
    logger.info(f"Model loaded: {type(model).__name__}")
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
 
    # (No vision/audio towers to prune -- Llama is a text-only checkpoint.)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # Gradient checkpointing
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
 
    norm_upcast_count = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__.lower()
        if "norm" in cls_name and hasattr(module, "weight") and module.weight is not None:
            module.to(torch.float32)
            norm_upcast_count += 1
    logger.info(f"Upcast {norm_upcast_count} norm layers to fp32")
 
    # Token IDs for "0" / "1" answer classification
    id0 = tokenizer("0", add_special_tokens=False).input_ids[0]
    id1 = tokenizer("1", add_special_tokens=False).input_ids[0]
    bos_id = tokenizer.bos_token_id
    logger.info(f"Token '0': {id0}")
    logger.info(f"Token '1': {id1}")
    logger.info(f"BOS: {bos_id}")
 
    # LoRA -- same rank, alpha, dropout, projection modules
    logger.info("Injecting LoRA (attention + MLP only)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_config)
 
    # print_trainable_parameters() writes straight to stdout -- capture it too
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} "
        f"|| trainable%: {100 * trainable_params / all_param:.4f}"
    )
 
    lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora_leaks = [n for n in lora_param_names if "lora" not in n]
    logger.info(f"Trainable LoRA params: {len(lora_param_names)}")
    logger.info(f"Trainable params that are not LoRA: {len(non_lora_leaks)}")
    if non_lora_leaks:
        logger.info(f"  WARNING -- these should not be here: {non_lora_leaks[:5]}")
 
    # ---------------- DATASETS / LOADERS ----------------
    collate_fn = make_collate_fn(tokenizer)
 
    train_dataset = TamilDataset(X_train, y_train, tokenizer, bos_id, id0, id1, MAX_LEN)
    val_dataset = TamilDataset(X_val, y_val, tokenizer, bos_id, id0, id1, MAX_LEN)
 
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
 
    class_counts = np.bincount(y_train)
    class_weights_tensor = torch.tensor(
        [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32, device=model.device,
    )
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()
    logger.info(f"Class weights: {class_weights_tensor}")
 
    # ---------------- OPTIMIZER (LoRA params only) ----------------
    lora_params = [p for p in model.parameters() if p.requires_grad]
 
    optimizer = bnb_opt.optim.AdamW8bit(
        [{"params": lora_params, "lr": LR}],
        weight_decay=WEIGHT_DECAY,
    )
 
    logger.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
 
    # ---------------- GRADIENT FLOW CHECK ----------------
    model.train()
 
    optimizer.zero_grad(set_to_none=True)
    probe_batch = next(iter(train_loader))
    probe_loss = run_batch(probe_batch, model, id0, id1, class_weights_tensor)
    probe_loss.backward()
 
    lora_grad_norms = [p.grad.norm().item() for p in lora_params if p.grad is not None]
    logger.info(f"LoRA params with grad: {len(lora_grad_norms)} / {len(lora_params)}")
    if lora_grad_norms:
        logger.info(f"Mean LoRA grad norm: {np.mean(lora_grad_norms):.6f}")
    else:
        logger.info("  (none)")
 
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # ---------------- TRAINING LOOP ----------------
    # Best checkpoint kept in memory only (by val macro-F1) so the final
    # test pass uses the best epoch's weights. Nothing is written to disk --
    # the log file is the only output of this script.
    best_val_macro_f1 = -np.inf
    best_state = None
    epochs_no_improve = 0
 
    # FLOPs accumulators (analytical estimate, see FLOPS_PER_PARAM_* above)
    total_train_tokens = 0
    total_train_flops = 0.0
    total_eval_tokens = 0
    total_eval_flops = 0.0
 
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
 
    for epoch in epoch_bar:
        # ---- TRAIN ----
        model.train()
 
        total_train_loss = 0.0
        epoch_train_tokens = 0
        optimizer.zero_grad(set_to_none=True)
 
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False, unit="batch")
 
        for step, batch in enumerate(train_bar):
            loss = run_batch(batch, model, id0, id1, class_weights_tensor)
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
 
            total_train_loss += loss.item()
            epoch_train_tokens += batch["attention_mask"].sum().item()
 
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
 
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            del loss, scaled_loss
 
        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
 
        train_loss = total_train_loss / len(train_loader)
 
        epoch_train_flops = FLOPS_PER_PARAM_TRAIN * all_param * epoch_train_tokens
        total_train_tokens += epoch_train_tokens
        total_train_flops += epoch_train_flops
 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        # ---- VALIDATION ----
        val_loss, val_macro_f1, val_acc, val_preds, val_targets, val_tokens = evaluate(
            val_loader, f"Epoch {epoch + 1}/{EPOCHS} [val]",
            model, id0, id1, class_weights_tensor,
        )
 
        val_flops = FLOPS_PER_PARAM_FWD * all_param * val_tokens
        total_eval_tokens += val_tokens
        total_eval_flops += val_flops
 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_macro_f1=f"{val_macro_f1:.4f}",
        )
 
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val macro-F1: {val_macro_f1:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train tokens: {epoch_train_tokens:,} | Train FLOPs: {format_flops(epoch_train_flops)} | "
            f"Val tokens: {val_tokens:,} | Val FLOPs: {format_flops(val_flops)}"
        )
 
        # ---- CLASSIFICATION REPORT (every epoch) ----
        report = classification_report(
            val_targets, val_preds, target_names=["Non-toxic", "Toxic"], digits=5
        )
        logger.info(f"Validation classification report -- epoch {epoch + 1}:\n{report}")
 
        # ---- TRACK BEST (in memory only, by macro-F1) ----
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if "lora" in k
            }
            epochs_no_improve = 0
            logger.info(f"New best validation macro-F1: {best_val_macro_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info("Early stopping.")
                break
 
    # ---------------- RELOAD BEST CHECKPOINT (in memory, by macro-F1) ----------------
    if best_state is not None:
        logger.info(f"Reloading best in-memory checkpoint (val macro-F1: {best_val_macro_f1:.4f}) before test...")
        model.load_state_dict(best_state, strict=False)
    else:
        logger.info("No improved checkpoint was ever recorded -- using final epoch weights.")
 
    # ---------------- TEST ----------------
    test_dataset = TamilDataset(X_test, y_test, tokenizer, bos_id, id0, id1, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
 
    logger.info("=" * 80)
    logger.info("TEST EVALUATION")
    logger.info("=" * 80)
 
    _, test_macro_f1, test_acc, test_preds, test_targets, test_tokens = evaluate(
        test_loader, "Evaluating", model, id0, id1, class_weights_tensor,
    )
 
    test_flops = FLOPS_PER_PARAM_FWD * all_param * test_tokens
    total_eval_tokens += test_tokens
    total_eval_flops += test_flops
 
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
 
    acc = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds)
 
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:   {acc:.5f}")
    logger.info(f"F1 (Toxic): {f1:.5f}")
    logger.info(f"Macro-F1:   {test_macro_f1:.5f}")
 
    report = classification_report(
        test_targets, test_preds, target_names=["Non-toxic", "Toxic"], digits=5
    )
    logger.info(f"Test classification report (point estimate, 5 digits):\n{report}")
 
    # ---------------- BOOTSTRAP CI ON TEST SET ----------------
    logger.info("=" * 80)
    logger.info(f"BOOTSTRAP RESAMPLING ON TEST SET (N = {N_BOOTSTRAP})")
    logger.info("=" * 80)
 
    boot_result = bootstrap_classification_report(
        test_targets, test_preds, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
 
    boot_report_str = format_bootstrap_report(boot_result, n_boot=N_BOOTSTRAP, digits=5)
    logger.info(f"Bootstrap classification report (mean +/- std, 5 digits):\n{boot_report_str}")
 
    # Also dump every single metric explicitly, one per line, at 5 digits,
    # so nothing is left implicit in the table formatting above.
    logger.info("-" * 80)
    logger.info("Bootstrap metrics, one per line (point | mean +/- std | 95% CI), 5 digits:")
    label_map = {
        "precision_0": "Non-toxic precision",
        "recall_0": "Non-toxic recall",
        "f1_0": "Non-toxic f1-score",
        "support_0": "Non-toxic support",
        "precision_1": "Toxic precision",
        "recall_1": "Toxic recall",
        "f1_1": "Toxic f1-score",
        "support_1": "Toxic support",
        "accuracy": "Accuracy",
        "macro_precision": "Macro avg precision",
        "macro_recall": "Macro avg recall",
        "macro_f1": "Macro avg f1-score",
        "weighted_precision": "Weighted avg precision",
        "weighted_recall": "Weighted avg recall",
        "weighted_f1": "Weighted avg f1-score",
    }
    for key, label in label_map.items():
        pt = boot_result["point"][key]
        mn = boot_result["mean"][key]
        sd = boot_result["std"][key]
        lo = boot_result["ci_lo"][key]
        hi = boot_result["ci_hi"][key]
        logger.info(
            f"{label:<24}: point={pt:.5f} | mean +/- std = {mn:.5f} +/- {sd:.5f} "
            f"| 95% CI = [{lo:.5f}, {hi:.5f}]"
        )
 
    # ---------------- INFERENCE COST BENCHMARKING ----------------
    if RUN_INFERENCE_BENCHMARK:
        logger.info("=" * 80)
        logger.info("INFERENCE COST BENCHMARKING (test-set batch size, real-world latency/throughput)")
        logger.info("=" * 80)
        benchmark_inference(
            test_loader, model, id0, id1, class_weights_tensor, all_param, logger,
            n_warmup=BENCHMARK_WARMUP_BATCHES, n_batches=BENCHMARK_MAX_BATCHES,
        )
 
    if RUN_BATCH_SIZE_SWEEP:
        benchmark_batch_size_sweep(
            test_dataset, collate_fn, model, id0, id1, class_weights_tensor, all_param, logger,
        )
 
    # ---------------- COMPUTATIONAL COST (FLOPs) ----------------
    logger.info("=" * 80)
    logger.info("COMPUTATIONAL COST (analytical FLOPs estimate)")
    logger.info("=" * 80)
    logger.info(f"Active parameters (forward pass width): {all_param:,}")
    logger.info(f"Trainable (LoRA) parameters: {trainable_params:,}")
    logger.info(f"Total training tokens processed (all epochs run): {total_train_tokens:,}")
    logger.info(f"Total eval/inference tokens processed (val, all epochs + final test): {total_eval_tokens:,}")
    logger.info(f"Training FLOPs (fwd+bwd, {FLOPS_PER_PARAM_TRAIN}N convention): {format_flops(total_train_flops)}")
    logger.info(f"Eval/inference FLOPs (fwd only, {FLOPS_PER_PARAM_FWD}N convention): {format_flops(total_eval_flops)}")
    logger.info(f"Total FLOPs (train + eval): {format_flops(total_train_flops + total_eval_flops)}")
    logger.info(
        "Note: analytical estimate (Kaplan et al. 2N forward / 6N forward+backward convention). "
        "Treats all active (frozen + LoRA) parameters as compute cost per token, since the frozen "
        "base weights are still multiplied on every pass and 4-bit quantization reduces memory "
        "footprint, not the number of multiply-accumulate operations. Bootstrap resampling adds "
        "no model compute and is excluded."
    )
 
    logger.info(f"Run complete. Log saved to: {log_path}")
 

if __name__ == "__main__":
    main()
