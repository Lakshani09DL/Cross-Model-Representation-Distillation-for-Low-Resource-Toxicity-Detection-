"""
Offensive-language classification: Llama 3.2 3B + LoRA with a Qwen3-Embedding-as-TEACHER
representation-alignment auxiliary loss.  NOTHING is injected into Llama.

Datasets (DATASET switch): "sinhala" = random 80/10/10 split of one CSV (SPLIT_SEED);
"tamil" = DravidianCodeMix official train/dev/test (dev is the validation set, "not-Tamil"
rows dropped, every offensive sub-category -> positive class).

Training:
    h_l   = masked-mean pool of layer-l hidden states over the SENTENCE tokens only
            (prompt template and padding excluded)
    z     = g_l(h_l)            g_l = LayerNorm -> Linear(d_model -> 1024)   [train-time only]
    t     = Qwen3-Embedding-0.6B embedding of the sentence (cached; centered + L2-normalised)
    loss  = CE(last-token 0/1 logits)  +  lambda(step) * L_align(z, t)

    L_align = W_NCE * InfoNCE(z vs the whole bank of train targets, false negatives masked)
            + W_REL * KL( softmax(t . bank / tau_t) || softmax(z . bank / tau) )   (self masked)
            + W_COS * (1 - cos(z, t))

Inference: plain Llama 3.2 3B + LoRA. The embedding model and the heads are not used.
Labels / PSM information never enter the alignment loss (it is label-free).
Optional UNLABELED_PATH adds extra label-free text that only receives the alignment loss.

Llama-specific plumbing (ported from the Gemma 4 E2B version):
  - Plain LlamaForCausalLM: model.model IS the text backbone (28 layers, d_model 3072).
    No nested language_model, no vision tower to prune.
  - No per-layer-input (PLE) mechanism, so the forward is just embed_tokens -> backbone -> lm_head.
  - One LoRA regex covers every layer (self_attn + mlp at every layer). GQA k_proj/v_proj
    (3072->1024) are handled automatically.
  - lm_head is applied to the LAST token only (vocab is 128k).
  - hidden_states[l + 1] is the output of layer l. For the LAST layer (27) HF returns it
    AFTER the final RMSNorm; for every other layer it is the raw residual stream.

Diagnostics (all evaluation-only, run before and after training):
    * linear CKA between pooled Llama layer states and teacher targets, for EVERY layer
    * linear probe (logistic regression) for the offensive class on pooled states at the aligned layers
    * held-out alignment quality: mean cosine and recall@1 of g(h_l) against its own teacher target

Every run writes to its own timestamped directory (log, predictions, checkpoints, diagnostics).

Baseline: set ALIGN_LAMBDA_START = ALIGN_LAMBDA_END = 0.0 -> plain Llama+LoRA with the SAME
data pipeline, seed and split. Use its test_predictions.npz as BASELINE_PREDS_PATH for the
paired bootstrap in the aligned run.
"""

import datetime
import gc
import math
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.nn as bnb_nn
import bitsandbytes as bnb_opt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re


# ============================================================================
# CONFIG -- edit these directly
# ============================================================================

DATASET = "tamil"           # "sinhala" | "tamil"

# # Sinhala: one CSV (columns Hate_speech, Sentence), split randomly with SPLIT_SEED
# DATA_PATH = "/home/cse_g3/Subhagya/notebooks/data/sentence-level-annotation.csv"
# # Tamil: official DravidianCodeMix tab-separated splits, no header: col0 = comment, col1 = category
# TAMIL_TRAIN_PATH = "/home/cse_g3/Subhagya/notebooks/data/dravidian_data/DravidianCodeMix/tamil_offensive_full_train.csv"
# TAMIL_DEV_PATH = "/home/cse_g3/Subhagya/notebooks/data/dravidian_data/DravidianCodeMix/tamil_offensive_full_dev.csv"
# TAMIL_TEST_PATH = "/home/cse_g3/Subhagya/notebooks/data/dravidian_data/DravidianCodeMix/tamil_offensive_full_test.csv"

DATA_PATH = "/home/cse_g2/Raleesa/Notebooks/sentence-level-annotation.csv"
# Tamil: official DravidianCodeMix tab-separated splits, no header: col0 = comment, col1 = category
TAMIL_TRAIN_PATH = "/home/cse_g2/Raleesa/Notebooks/tamil_offensive_full_train.csv"
TAMIL_DEV_PATH = "/home/cse_g2/Raleesa/Notebooks/tamil_offensive_full_dev.csv"
TAMIL_TEST_PATH = "/home/cse_g2/Raleesa/Notebooks/tamil_offensive_full_test.csv"

SAVE_ROOT = os.path.join("/home/cse_g2/Raleesa/Notebooks/condlora",
                         "align_llama32_3b_q3emb_" + {"sinhala": "si", "tamil": "ta"}[DATASET])

# Optional stratified cap on the labeled TRAIN set (Tamil train is ~3x Sinhala). Use the SAME
# value for the baseline and the aligned run.
MAX_TRAIN = None

# Optional extra label-free text (alignment loss only). .txt = one sentence per line,
# .csv = column UNLABELED_COLUMN. Anything that also appears in the labeled data is dropped.
UNLABELED_PATH = None
UNLABELED_COLUMN = "Sentence"
UNLABELED_MAX = 20000
UNLAB_PER_STEP = 1            # unlabeled micro-batches per labeled micro-batch

# Embedding teacher (Qwen3-Embedding-0.6B)
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"   # teacher (1024-d output, last-token pooling)
EMB_DIM = 1024
EMB_BATCH_SIZE = 32
CENTER_TARGETS = True         # subtract train-mean of teacher vectors, then re-normalise

# Where to align (fractions of depth; layer = round(frac * (num_layers - 1)))
# 28 layers: 0.65 -> layer 18.
ALIGN_LAYER_FRACS = [0.65]
ALIGN_LAYERS = None           # explicit 0-based layer indices override the fractions, e.g. [16, 20]
HEAD_MLP = False              # False: LN -> Linear.  True: LN -> Linear -> GELU -> Linear
HEAD_LR = 1e-3
HEAD_WD = 0.05

# Alignment loss
ALIGN_W_NCE = 1.0
ALIGN_W_REL = 1.0
ALIGN_W_COS = 0.0
ALIGN_TAU = 0.07              # student temperature (InfoNCE and relational)
ALIGN_REL_TAU_T = 0.05        # teacher temperature for the relational term
FALSE_NEG_THRESH = 0.85       # mask InfoNCE negatives whose (centered) target cosine exceeds this
ALIGN_LAMBDA_START = 0.3      # 0 / 0 -> pure baseline (no alignment, no heads)
ALIGN_LAMBDA_END = 0.1        # linearly annealed over all optimizer steps
ALIGN_ONLY_STEPS = 0          # optimizer steps at the start with CE switched off (alignment warm-up)

# Teacher-target control: "shuffle" gives every training sentence ANOTHER sentence's teacher vector
# (same lambda, same everything else). Val/test targets stay real, so val recall@1 should fall to chance.
ALIGN_TARGET_MODE = "real"    # "real" | "shuffle"

# Llama / LoRA
LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # replace with your actual repo id
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_LAYERS = "all"    # "all" | "later_half" | "upto_align"
# Llama has one block type at every layer, so a single regex covers the stack.
# Group 1 is the layer index.
LORA_TARGET_REGEX = (
    r".*model\.layers\.(\d+)\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)

# Training
SPLIT_SEED = 42               # Sinhala random split + probe subsample; do NOT change between runs
RANDOM_SEED = 42              # training seed (LoRA init, shuffling, dropout); change for extra seeds
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16         # effective batch = 64
EVAL_BATCH_SIZE = 16
MAX_SENT_TOKENS = 100         # the SENTENCE is truncated to this; template is never cut
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.01
COMPUTE_DTYPE = torch.bfloat16
PATIENCE = 3
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Diagnostics
RUN_CKA = True
RUN_PROBE = True
PROBE_C = 0.1
PROBE_MAX_TRAIN = 10000       # random subset of train used to fit the probe (same subset pre/post)

# Bootstrap
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
BASELINE_PREDS_PATH = None    # test_predictions.npz of the lambda=0 run for a paired diff

LABEL_MAP = {"Not offensive": 0, "Abusive": 1, "Hate-Inducing": 1}          # Sinhala
TAMIL_LABEL_MAP = {                                                          # Tamil (DravidianCodeMix)
    "Not_offensive": 0,
    "Offensive_Untargetede": 1,
    "Offensive_Untargeted": 1,
    "Offensive_Targeted_Insult_Individual": 1,
    "Offensive_Targeted_Insult_Group": 1,
    "Offensive_Targeted_Insult_Other": 1,
}
NEG = -1e4                    # finite "minus infinity" for masked logits

PROMPT_PREFIX = ('Classify the following text as either "not offensive" (0) or '
                 '"offensive" (1). Answer with only the number.\n\nText: ')
PROMPT_SUFFIX = "\nAnswer: "


# ============================================================================
# TOKENISATION WITH A SENTENCE-TOKEN MASK
# ============================================================================

def build_example(tokenizer, sentence, bos_id):
    """Prompt tokens + a mask marking the tokens that belong to the SENTENCE only.

    Only the sentence is truncated (to MAX_SENT_TOKENS), so the 'Answer:' suffix is
    always present. Tokenisation of the full prompt is unchanged (one pass over the whole
    string); offsets are used to find which tokens overlap the sentence characters.
    """
    sent = str(sentence)
    enc_s = tokenizer(sent, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc_s["input_ids"]) > MAX_SENT_TOKENS:
        cut = enc_s["offset_mapping"][MAX_SENT_TOKENS - 1][1]
        sent = sent[:cut]

    text = PROMPT_PREFIX + sent + PROMPT_SUFFIX
    s0, s1 = len(PROMPT_PREFIX), len(PROMPT_PREFIX) + len(sent)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    sent_mask = [1 if (b > a and a < s1 and b > s0) else 0
                 for a, b in enc["offset_mapping"]]

    if bos_id is not None:                                # Llama 3 BOS = <|begin_of_text|>
        ids = [bos_id] + ids
        sent_mask = [0] + sent_mask
    return ids, sent_mask


class TextDataset(Dataset):
    """labels may be None (unlabeled); targets are the centered/normalised teacher vectors;
    bank_idx (position of the example's target in the training bank) may be None."""

    def __init__(self, sentences, labels, targets, bank_idx, tokenizer, bos_id, id0, id1):
        self.sentences = sentences
        self.labels = labels
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.bank_idx = bank_idx
        self.tokenizer = tokenizer
        self.bos_id = bos_id
        self.id0, self.id1 = id0, id1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        ids, sm = build_example(self.tokenizer, self.sentences[i], self.bos_id)
        if self.labels is None:
            tid = -1
        else:
            tid = self.id1 if int(self.labels[i]) == 1 else self.id0
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "sent_mask": sm,
            "target_id": tid,
            "tgt": self.targets[i],
            "bank_idx": -1 if self.bank_idx is None else int(self.bank_idx[i]),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        L = max(len(x["input_ids"]) for x in batch)
        ids, am, sm = [], [], []
        for x in batch:                                  # left padding
            p = L - len(x["input_ids"])
            ids.append([tokenizer.pad_token_id] * p + x["input_ids"])
            am.append([0] * p + x["attention_mask"])
            sm.append([0] * p + x["sent_mask"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "sent_mask": torch.tensor(sm, dtype=torch.long),
            "target_ids": torch.tensor([x["target_id"] for x in batch], dtype=torch.long),
            "tgt": torch.stack([x["tgt"] for x in batch]),
            "bank_idx": torch.tensor([x["bank_idx"] for x in batch], dtype=torch.long),
        }
    return collate_fn


def cycle(loader):
    while True:
        for b in loader:
            yield b


# ============================================================================
# LoRA (plain)
# ============================================================================

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        base_out = self.base_layer(x)
        xd = self.dropout(x).to(self.lora_A.dtype)
        delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + (delta * self.scaling).to(base_out.dtype)


def inject_lora(root, regex, target_layers, r, alpha, dropout, device):
    pattern = re.compile(regex)
    replaced, skipped = [], 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, (nn.Linear, bnb_nn.Linear4bit)):
            continue
        m = pattern.match(name)
        if not m:
            continue
        if int(m.group(1)) not in target_layers:
            skipped += 1
            continue
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child, LoRALinear(module, r, alpha, dropout).to(device))
        replaced.append(name)
    return replaced, skipped


def is_lora_key(name):
    return "lora_A" in name or "lora_B" in name


# ============================================================================
# ALIGNMENT HEADS + LOSSES
# ============================================================================

class AlignHeads(nn.Module):
    """One head per aligned layer; used at training time only."""

    def __init__(self, d_model, d_out, n_heads, mlp):
        super().__init__()

        def mk():
            if mlp:
                return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                     nn.GELU(), nn.Linear(d_model, d_out))
            return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_out))

        self.heads = nn.ModuleList([mk() for _ in range(n_heads)])

    def forward(self, pooled_list):
        return [h(p.float()) for h, p in zip(self.heads, pooled_list)]


def pool_masked(h, mask):
    """h: (B, T, D); mask: (B, T) -> (B, D) float32 mean over masked tokens."""
    m = mask.to(torch.float32).unsqueeze(-1)
    return (h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)


def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def compute_align(zs, bank_idx, bank):
    """zs: list (one per aligned layer) of (B, D) head outputs.
    bank: (N, D) centered+normalised training targets; bank_idx: (B,) own position in bank."""
    tgt = bank[bank_idx]
    B = tgt.shape[0]
    ar = torch.arange(B, device=tgt.device)

    sim_t = tgt @ bank.t()                                           # (B, N)
    self_mask = torch.zeros_like(sim_t, dtype=torch.bool)
    self_mask[ar, bank_idx] = True
    fn_mask = (sim_t > FALSE_NEG_THRESH) & ~self_mask               # likely false negatives

    p_t = None
    if ALIGN_W_REL > 0:
        p_t = F.softmax(sim_t.masked_fill(self_mask, NEG) / ALIGN_REL_TAU_T, dim=-1)

    comp = {"nce": 0.0, "rel": 0.0, "cos": 0.0}
    for z in zs:
        zn = F.normalize(z.float(), dim=-1)
        logits = (zn @ bank.t()) / ALIGN_TAU
        if ALIGN_W_NCE > 0:
            comp["nce"] = comp["nce"] + F.cross_entropy(logits.masked_fill(fn_mask, NEG), bank_idx)
        if ALIGN_W_REL > 0:
            logp = F.log_softmax(logits.masked_fill(self_mask, NEG), dim=-1)
            comp["rel"] = comp["rel"] + F.kl_div(logp, p_t, reduction="batchmean")
        comp["cos"] = comp["cos"] + (1.0 - (zn * tgt).sum(-1)).mean()

    n = len(zs)
    comp = {k: v / n for k, v in comp.items()}
    comp["total"] = ALIGN_W_NCE * comp["nce"] + ALIGN_W_REL * comp["rel"] + ALIGN_W_COS * comp["cos"]
    return comp


# ============================================================================
# FORWARD HELPERS
# ============================================================================

def run_batch(batch, model, language_model, need_logits=True, hidden_layers=None):
    """Returns dict(logits, pooled, valid). pooled[i] is the sentence-token mean of the
    output of 0-based layer hidden_layers[i] (hidden_states[l + 1])."""
    dev = model.device
    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)
    sent_mask = batch["sent_mask"].to(dev)
    want_hidden = hidden_layers is not None

    with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        inputs_embeds = language_model.embed_tokens(input_ids)
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=want_hidden,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :]) if need_logits else None

    pooled = None
    if want_hidden:
        hs = outputs.hidden_states
        pooled = [pool_masked(hs[l + 1], sent_mask) for l in hidden_layers]
    return {"logits": logits, "pooled": pooled, "valid": sent_mask.sum(1) > 0}


def ce_from_logits(logits, target_ids, id0, id1, class_weights):
    bl = logits[:, [id0, id1]].float()
    bt = (target_ids == id1).long()
    return F.cross_entropy(bl, bt, weight=class_weights), bt


@torch.no_grad()
def evaluate(loader, desc, model, language_model, heads, align_layers, id0, id1, cw):
    """Returns (avg_ce, macro_f1, acc, preds, targets, align_metrics or None)."""
    model.eval()
    if heads is not None:
        heads.eval()
    total_ce, preds, targets = 0.0, [], []
    z_all = [[] for _ in align_layers] if heads is not None else None
    t_all = []

    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        out = run_batch(batch, model, language_model,
                        hidden_layers=align_layers if heads is not None else None)
        tid = batch["target_ids"].to(model.device)
        ce, bt = ce_from_logits(out["logits"], tid, id0, id1, cw)
        total_ce += float(ce)
        lg = out["logits"].float()
        preds.extend((lg[:, id1] > lg[:, id0]).long().cpu().numpy())
        targets.extend(bt.cpu().numpy())

        if heads is not None:
            zs = heads(out["pooled"])
            v = out["valid"]
            for i, z in enumerate(zs):
                z_all[i].append(F.normalize(z[v].float(), dim=-1).cpu())
            t_all.append(batch["tgt"].to(model.device)[v].cpu())

    preds, targets = np.array(preds), np.array(targets)
    avg_ce = total_ce / len(loader)
    macro_f1 = f1_score(targets, preds, average="macro")
    acc = accuracy_score(targets, preds)

    align_metrics = None
    if heads is not None and t_all:
        T = torch.cat(t_all)
        n = T.shape[0]
        align_metrics = {}
        for i, l in enumerate(align_layers):
            Z = torch.cat(z_all[i])
            sim = Z @ T.t()
            align_metrics[f"cos_L{l}"] = float((Z * T).sum(-1).mean())
            align_metrics[f"recall@1_L{l}"] = float((sim.argmax(1) == torch.arange(n)).float().mean())
        align_metrics["chance_recall@1"] = 1.0 / n
    return avg_ce, macro_f1, acc, preds, targets, align_metrics


# ============================================================================
# DIAGNOSTICS: CKA + linear probe
# ============================================================================

def linear_cka(X, Y):
    X = X.astype(np.float64) - X.astype(np.float64).mean(0, keepdims=True)
    Y = Y.astype(np.float64) - Y.astype(np.float64).mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y) ** 2
    xtx = np.linalg.norm(X.T @ X)
    yty = np.linalg.norm(Y.T @ Y)
    return float(xty / (xtx * yty + 1e-12))


@torch.no_grad()
def extract_pooled(loader, layer_ids, desc, model, language_model, id1):
    """-> feats (N, len(layer_ids), d_model) float32, tgt (N, D), y (N,) in {0,1}."""
    model.eval()
    feats, tg, ys = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        out = run_batch(batch, model, language_model, need_logits=False, hidden_layers=layer_ids)
        feats.append(torch.stack(out["pooled"], dim=1).cpu())
        tg.append(batch["tgt"])
        ys.append((batch["target_ids"] == id1).long())
    return torch.cat(feats).numpy(), torch.cat(tg).numpy(), torch.cat(ys).numpy()


def probe_scores(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=PROBE_C, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict(sc.transform(Xte))
    return (f1_score(yte, p, average="macro", zero_division=0),
            f1_score(yte, p, pos_label=1, zero_division=0))


# ============================================================================
# BOOTSTRAP + LOGGING
# ============================================================================

METRICS = {
    "Accuracy": accuracy_score,
    "Macro F1": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
    "Macro Precision": lambda t, p: precision_score(t, p, average="macro", zero_division=0),
    "Macro Recall": lambda t, p: recall_score(t, p, average="macro", zero_division=0),
    "F1 (offensive)": lambda t, p: f1_score(t, p, pos_label=1, zero_division=0),
    "F1 (not offensive)": lambda t, p: f1_score(t, p, pos_label=0, zero_division=0),
}


def _summarize(point, samples, ci):
    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "boot_mean": float(np.mean(samples)), "boot_std": float(np.std(samples, ddof=1))}


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    s = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, fn in METRICS.items():
            s[k].append(fn(y_true[idx], y_pred[idx]))
    return {k: _summarize(fn(y_true, y_pred), s[k], ci) for k, fn in METRICS.items()}


def paired_bootstrap_diff(y_true, pa, pb, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    s = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, fn in METRICS.items():
            s[k].append(fn(y_true[idx], pa[idx]) - fn(y_true[idx], pb[idx]))
    return {k: _summarize(fn(y_true, pa) - fn(y_true, pb), s[k], ci) for k, fn in METRICS.items()}


def print_ci_table(results, title):
    print(title)
    for name, r in results.items():
        print(f"  {name:<20} {r['point']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]  "
              f"mean +- std: {r['boot_mean']:.4f} +- {r['boot_std']:.4f}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _load_tamil_split(path, split_name):
    """Same parsing as the earlier Tamil CondLoRA script: tab-delimited, no header,
    default quoting, bad lines skipped, 'not-Tamil' rows dropped, all offensive
    sub-categories -> 1."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tamil {split_name} file not found: {path}")
    df = pd.read_csv(path, sep="\t", header=None, on_bad_lines="skip")
    df_clean = df[[0, 1]].copy()
    df_clean.columns = ["comment", "category"]
    df_clean = df_clean.dropna(subset=["comment", "category"]).reset_index(drop=True)

    df_f = df_clean[df_clean["category"] != "not-Tamil"].copy()
    df_f["label"] = df_f["category"].map(TAMIL_LABEL_MAP)
    unmapped = df_f[df_f["label"].isna()]["category"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: Unmapped categories in {split_name}: {list(unmapped)}")
    df_f = df_f.dropna(subset=["label"]).copy()
    df_f["label"] = df_f["label"].astype(int)

    print(f"\n--- {split_name} ---")
    print(f"Raw rows: {len(df)} | After cleaning: {len(df_f)}")
    print(f"Class distribution:\n{df_f['label'].value_counts()}")
    return df_f["comment"].astype(str).tolist(), df_f["label"].values


def load_splits():
    """-> X_tr, y_tr, X_val, y_val, X_test, y_test (lists + int arrays)."""
    if DATASET == "sinhala":
        df = pd.read_csv(DATA_PATH)
        df["label"] = df["Hate_speech"].map(LABEL_MAP)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        sentences = df["Sentence"].astype(str).tolist()
        labels = df["label"].values
        print(f"Total examples: {len(sentences)} | class distribution: {np.bincount(labels)}")
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=labels)
        train_idx, val_idx = train_test_split(
            np.arange(len(y_train)), test_size=VAL_SIZE, stratify=y_train, random_state=SPLIT_SEED)
        X_arr = np.array(X_train, dtype=object)
        return (X_arr[train_idx].tolist(), y_train[train_idx],
                X_arr[val_idx].tolist(), y_train[val_idx],
                list(X_test), np.asarray(y_test))
    if DATASET == "tamil":
        X_tr, y_tr = _load_tamil_split(TAMIL_TRAIN_PATH, "TRAIN")
        X_val, y_val = _load_tamil_split(TAMIL_DEV_PATH, "DEV")
        X_test, y_test = _load_tamil_split(TAMIL_TEST_PATH, "TEST")
        return X_tr, y_tr, X_val, y_val, X_test, y_test
    raise ValueError(DATASET)


def center_norm(emb, mu):
    e = emb - mu if CENTER_TARGETS else emb
    return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def load_unlabeled(path, forbidden):
    if path.endswith(".csv"):
        texts = pd.read_csv(path)[UNLABELED_COLUMN].astype(str).tolist()
    else:
        with open(path, encoding="utf-8") as f:
            texts = [ln.rstrip("\n") for ln in f]
    seen, out = set(), []
    for t in texts:
        t = t.strip()
        if t and t not in forbidden and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= UNLABELED_MAX:
            break
    return out


# ============================================================================
# MAIN
# ============================================================================

def _main_inner(run_dir):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    use_align = max(ALIGN_LAMBDA_START, ALIGN_LAMBDA_END) > 0
    print(f"Run dir: {run_dir}")
    print(f"dataset={DATASET} | seed={RANDOM_SEED} | target_mode={ALIGN_TARGET_MODE} | max_train={MAX_TRAIN}")
    print(f"use_align={use_align} | lambda {ALIGN_LAMBDA_START}->{ALIGN_LAMBDA_END} | "
          f"W_NCE={ALIGN_W_NCE} W_REL={ALIGN_W_REL} W_COS={ALIGN_W_COS} | tau={ALIGN_TAU} | "
          f"layer_fracs={ALIGN_LAYER_FRACS} layers_override={ALIGN_LAYERS} | head_mlp={HEAD_MLP} | "
          f"unlabeled={UNLABELED_PATH} | center_targets={CENTER_TARGETS}")

    # ---------------- DATA ----------------
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_splits()
    y_tr, y_val, y_test = np.asarray(y_tr), np.asarray(y_val), np.asarray(y_test)
    all_labeled = set(map(str.strip, X_tr + X_val + X_test))

    if MAX_TRAIN and len(X_tr) > MAX_TRAIN:
        keep, _ = train_test_split(np.arange(len(X_tr)), train_size=MAX_TRAIN,
                                   stratify=y_tr, random_state=SPLIT_SEED)
        keep = np.sort(keep)
        X_tr, y_tr = [X_tr[i] for i in keep], y_tr[keep]
        print(f"MAX_TRAIN={MAX_TRAIN}: using a stratified subset of the training set")
    print(f"Train: {len(X_tr)} {np.bincount(y_tr).tolist()} | Val: {len(X_val)} "
          f"{np.bincount(y_val).tolist()} | Test: {len(X_test)} {np.bincount(y_test).tolist()}")

    X_unl = []
    if UNLABELED_PATH:
        X_unl = load_unlabeled(UNLABELED_PATH, all_labeled)
        print(f"Unlabeled sentences kept (labeled/test overlap removed): {len(X_unl)}")

    # ---------------- EMBEDDING TEACHER TARGETS (cached, then freed) ----------------
    print("\nEncoding with Qwen3-Embedding-0.6B...")
    # Qwen3-Embedding pools the LAST token, so it needs left padding. Sentences are
    # encoded WITHOUT an instruction prompt (instructions are only for queries).
    teacher = SentenceTransformer(EMB_MODEL, model_kwargs={"torch_dtype": torch.float16},
                                  tokenizer_kwargs={"padding_side": "left"})
    teacher.max_seq_length = 256
    enc = lambda xs: teacher.encode(xs, batch_size=EMB_BATCH_SIZE, show_progress_bar=True,
                                    normalize_embeddings=True)
    emb_tr, emb_val, emb_test = enc(X_tr), enc(X_val), enc(X_test)
    emb_unl = enc(X_unl) if X_unl else np.zeros((0, EMB_DIM), dtype=np.float32)
    assert emb_tr.shape[1] == EMB_DIM
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mu = emb_tr.mean(0, keepdims=True)                              # train-split mean only
    T_tr = center_norm(emb_tr, mu)
    T_val = center_norm(emb_val, mu)
    T_test = center_norm(emb_test, mu)
    T_unl = center_norm(emb_unl, mu) if X_unl else np.zeros((0, EMB_DIM), dtype=np.float32)

    T_tr_bank, T_unl_bank = T_tr, T_unl
    if ALIGN_TARGET_MODE == "shuffle":
        rng = np.random.default_rng(RANDOM_SEED)
        T_tr_bank = T_tr[rng.permutation(len(T_tr))]
        if len(T_unl):
            T_unl_bank = T_unl[rng.permutation(len(T_unl))]
        print("TARGET CONTROL: training sentences are aligned to OTHER sentences' teacher vectors")
    elif ALIGN_TARGET_MODE != "real":
        raise ValueError(ALIGN_TARGET_MODE)

    # ---------------- LLAMA ----------------
    print("\n" + "=" * 80 + "\nLLAMA 3.2 3B + LoRA (Qwen3-Embedding as training-time teacher)\n" + "=" * 80)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, quantization_config=bnb_config, device_map={"": 0}, dtype=COMPUTE_DTYPE)
    print("Model loaded:", type(model).__name__)

    # Plain LlamaForCausalLM: model.model IS the text stack (embed_tokens, layers, norm).
    language_model = model.model
    text_config = model.config
    num_layers = text_config.num_hidden_layers
    d_model = text_config.hidden_size
    print("Backbone type:", type(language_model).__name__)

    if not hasattr(language_model, "embed_tokens"):
        raise RuntimeError("Llama model does not expose `embed_tokens` -- check module path.")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for _, module in model.named_modules():
        if "norm" in type(module).__name__.lower() and getattr(module, "weight", None) is not None:
            module.to(torch.float32)

    id0 = tokenizer("0", add_special_tokens=False).input_ids[0]
    id1 = tokenizer("1", add_special_tokens=False).input_ids[0]
    bos_id = tokenizer.bos_token_id
    print(f"Token '0': {id0} | Token '1': {id1} | BOS: {bos_id} | layers: {num_layers} | d_model: {d_model}")

    # aligned layers
    if ALIGN_LAYERS is not None:
        align_layers = sorted(set(int(l) for l in ALIGN_LAYERS))
    else:
        align_layers = sorted(set(int(round(f * (num_layers - 1))) for f in ALIGN_LAYER_FRACS))
    assert all(0 <= l < num_layers for l in align_layers)
    print(f"Aligned layers (0-based): {align_layers}")

    # LoRA
    if LORA_TARGET_LAYERS == "all":
        target_layers = set(range(num_layers))
    elif LORA_TARGET_LAYERS == "later_half":
        target_layers = set(range(num_layers // 2, num_layers))
    elif LORA_TARGET_LAYERS == "upto_align":
        target_layers = set(range(max(align_layers) + 1))
    else:
        raise ValueError(LORA_TARGET_LAYERS)
    for p in model.parameters():
        p.requires_grad = False
    replaced, skipped = inject_lora(model, LORA_TARGET_REGEX, target_layers,
                                    LORA_R, LORA_ALPHA, LORA_DROPOUT, model.device)
    print(f"LoRA on layers {min(target_layers)}-{max(target_layers)}: replaced {len(replaced)} linears "
          f"(r={LORA_R}), skipped {skipped}")
    print(f"  self_attn: {sum('self_attn' in n for n in replaced)} | "
          f"mlp: {sum('.mlp.' in n for n in replaced)}")

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and is_lora_key(n)]
    assert len(lora_params) == sum(p.requires_grad for p in model.parameters()), \
        "Unexpected trainable params outside LoRA."
    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")

    heads, bank = None, None
    if use_align:
        heads = AlignHeads(d_model, EMB_DIM, len(align_layers), HEAD_MLP).to(model.device)
        print(f"Align-head params (train-time only): {sum(p.numel() for p in heads.parameters()):,}")
        bank = torch.tensor(np.concatenate([T_tr_bank, T_unl_bank], axis=0),
                            dtype=torch.float32, device=model.device)
        print(f"Target bank: {tuple(bank.shape)} ({len(T_tr)} labeled + {len(T_unl)} unlabeled)")

    # ---------------- DATASETS / LOADERS ----------------
    collate = make_collate_fn(tokenizer)
    n_tr = len(X_tr)
    ds_tr = TextDataset(X_tr, y_tr, T_tr, np.arange(n_tr), tokenizer, bos_id, id0, id1)
    ds_val = TextDataset(X_val, y_val, T_val, None, tokenizer, bos_id, id0, id1)
    ds_test = TextDataset(X_test, y_test, T_test, None, tokenizer, bos_id, id0, id1)

    train_loader = DataLoader(ds_tr, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate)

    unl_iter = None
    if use_align and X_unl:
        ds_unl = TextDataset(X_unl, None, T_unl, n_tr + np.arange(len(X_unl)),
                             tokenizer, bos_id, id0, id1)
        unl_iter = cycle(DataLoader(ds_unl, batch_size=MICRO_BATCH_SIZE, shuffle=True,
                                    collate_fn=collate))

    cc = np.bincount(y_tr)
    cw = torch.tensor([1.0 / cc[0], 1.0 / cc[1]], dtype=torch.float32, device=model.device)
    cw = cw / cw.sum()
    print("Class weights:", cw)

    # ---------------- OPTIMIZER ----------------
    groups = [{"params": lora_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
    if use_align:
        groups.append({"params": list(heads.parameters()), "lr": HEAD_LR, "weight_decay": HEAD_WD})
    optimizer = bnb_opt.optim.AdamW8bit(groups)

    # ---------------- DIAGNOSTICS (eval-only; labels used only by the probe) ----------------
    probe_idx = np.arange(len(ds_tr))
    if len(ds_tr) > PROBE_MAX_TRAIN:
        probe_idx = np.sort(np.random.default_rng(SPLIT_SEED).choice(
            len(ds_tr), PROBE_MAX_TRAIN, replace=False))

    def run_diagnostics(stage):
        res = {}
        eval_loader_kw = dict(batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate)
        te_loader = DataLoader(ds_test, **eval_loader_kw)
        all_layers = list(range(num_layers))
        print(f"\n--- Diagnostics [{stage}] ---")
        Fte, Tte, yte = extract_pooled(te_loader, all_layers, f"[{stage}] test feats",
                                       model, language_model, id1)
        if RUN_CKA:
            cka = np.array([linear_cka(Fte[:, i, :], Tte) for i in range(num_layers)])
            res["cka"] = cka
            print("Linear CKA(pooled Llama layer, teacher target) on test, by layer:")
            print("  " + " ".join(f"L{i}:{c:.3f}{'*' if i in align_layers else ''}"
                                  for i, c in enumerate(cka)))
        if RUN_PROBE:
            tr_loader = DataLoader(torch.utils.data.Subset(ds_tr, probe_idx.tolist()), **eval_loader_kw)
            Ftr, _, ytr = extract_pooled(tr_loader, align_layers, f"[{stage}] train feats",
                                         model, language_model, id1)
            for j, l in enumerate(align_layers):
                mf1, tf1 = probe_scores(Ftr[:, j, :], ytr, Fte[:, l, :], yte)
                res[f"probe_macro_f1_L{l}"] = mf1
                res[f"probe_toxic_f1_L{l}"] = tf1
                print(f"Linear probe @ L{l}: macro-F1 {mf1:.4f} | offensive F1 {tf1:.4f}")
            del Ftr
        del Fte
        gc.collect()
        torch.cuda.empty_cache()
        return res

    diag_pre = run_diagnostics("pre-training (base Llama)")

    # ---------------- GRADIENT FLOW CHECK ----------------
    model.train()
    if heads is not None:
        heads.train()
    optimizer.zero_grad(set_to_none=True)
    pb = next(iter(train_loader))
    po = run_batch(pb, model, language_model, hidden_layers=align_layers if use_align else None)
    assert po["logits"] is not None
    if use_align:
        assert po["pooled"][0].shape[-1] == d_model, "d_model mismatch"
        print(f"Sentence tokens in probe batch: {po['valid'].sum().item()}/{len(pb['input_ids'])} rows valid, "
              f"mean sentence tokens/row {pb['sent_mask'].sum(1).float().mean():.1f} "
              f"(of {pb['attention_mask'].sum(1).float().mean():.1f} total)")
    tid = pb["target_ids"].to(model.device)
    ce0, _ = ce_from_logits(po["logits"], tid, id0, id1, cw)
    l0 = ce0
    if use_align:
        al0 = compute_align([z[po["valid"]] for z in heads(po["pooled"])],
                            pb["bank_idx"].to(model.device)[po["valid"]], bank)
        l0 = l0 + ALIGN_LAMBDA_START * al0["total"]
        print(f"Step-0 losses: CE {float(ce0):.4f} | nce {_f(al0['nce']):.3f} "
              f"rel {_f(al0['rel']):.3f} cos {_f(al0['cos']):.3f}")
    l0.backward()
    gn = defaultdict(list)
    for n, p in model.named_parameters():
        if p.grad is not None:
            gn["A" if "lora_A" in n else "B"].append(p.grad.float().norm().item())
    if heads is not None:
        gn["head"] = [p.grad.float().norm().item() for p in heads.parameters() if p.grad is not None]
    print("Grad norms at step 0: " + " | ".join(f"{k}={np.mean(v):.2e}" for k, v in gn.items()))
    optimizer.zero_grad(set_to_none=True)
    del po, l0, ce0
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- TRAINING ----------------
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_opt_steps = EPOCHS * steps_per_epoch

    def lam_at(step):
        frac = min(1.0, step / max(1, total_opt_steps))
        return ALIGN_LAMBDA_START + (ALIGN_LAMBDA_END - ALIGN_LAMBDA_START) * frac

    best_val_macro_f1, best_state, no_improve = -np.inf, None, 0
    opt_step = 0

    for epoch in range(EPOCHS):
        model.train()
        if heads is not None:
            heads.train()
        optimizer.zero_grad(set_to_none=True)
        sums = defaultdict(float)
        n_mb = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False, unit="batch")
        for step, batch in enumerate(bar):
            lam = lam_at(opt_step) if use_align else 0.0
            ce_w = 0.0 if opt_step < ALIGN_ONLY_STEPS else 1.0

            out = run_batch(batch, model, language_model,
                            hidden_layers=align_layers if use_align else None)
            tid = batch["target_ids"].to(model.device)
            ce, _ = ce_from_logits(out["logits"], tid, id0, id1, cw)
            loss = ce_w * ce

            if use_align:
                v = out["valid"]
                if v.any():
                    zs = [z[v] for z in heads(out["pooled"])]
                    al = compute_align(zs, batch["bank_idx"].to(model.device)[v], bank)
                    loss = loss + lam * al["total"]
                    for k in ("nce", "rel", "cos"):
                        sums[k] += _f(al[k])
            (loss / GRAD_ACCUM_STEPS).backward()
            sums["ce"] += float(ce)
            n_mb += 1
            del out, loss, ce

            # label-free extra text: alignment loss only (separate backward keeps memory low)
            if unl_iter is not None and lam > 0:
                for _ in range(UNLAB_PER_STEP):
                    ub = next(unl_iter)
                    uo = run_batch(ub, model, language_model, need_logits=False,
                                   hidden_layers=align_layers)
                    v = uo["valid"]
                    if v.any():
                        zs = [z[v] for z in heads(uo["pooled"])]
                        al_u = compute_align(zs, ub["bank_idx"].to(model.device)[v], bank)
                        ((lam * al_u["total"]) / GRAD_ACCUM_STEPS).backward()
                        sums["unl_nce"] += _f(al_u["nce"])
                    del uo

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
            bar.set_postfix(ce=f"{sums['ce'] / n_mb:.4f}")

        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

        gc.collect()
        torch.cuda.empty_cache()

        val_ce, val_f1, val_acc, _, _, val_align = evaluate(
            val_loader, f"Epoch {epoch + 1}/{EPOCHS} [val]",
            model, language_model, heads, align_layers, id0, id1, cw)

        msg = (f"\nEpoch {epoch + 1}/{EPOCHS} | lambda {lam_at(opt_step) if use_align else 0:.3f} | "
               f"Train CE {sums['ce'] / n_mb:.4f}")
        if use_align:
            msg += (f" | train align: nce {sums['nce'] / n_mb:.3f} rel {sums['rel'] / n_mb:.3f} "
                    f"cos {sums['cos'] / n_mb:.3f}")
        msg += f"\n          Val CE {val_ce:.4f} | Val macro-F1 {val_f1:.4f} | Val acc {val_acc:.4f}"
        if val_align:
            msg += "\n          Val alignment: " + " ".join(f"{k}={v:.4f}" for k, v in val_align.items())
        print(msg)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_state = {
                "lora": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                         if is_lora_key(k)},
                "heads": ({k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
                          if heads is not None else None),
            }
            no_improve = 0
            print(f"✓ New best validation macro-F1: {best_val_macro_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    # ---------------- RELOAD BEST + TEST ----------------
    if best_state is not None:
        print(f"\nReloading best checkpoint (val macro-F1 {best_val_macro_f1:.4f})...")
        model.load_state_dict(best_state["lora"], strict=False)
        if heads is not None and best_state["heads"] is not None:
            heads.load_state_dict(best_state["heads"])

    print("\n" + "=" * 80 + "\nTEST EVALUATION (plain Llama + LoRA; embedding model and heads not used for predictions)\n"
          + "=" * 80)
    _, test_f1, test_acc, pred, target, test_align = evaluate(
        test_loader, "Evaluating", model, language_model, heads, align_layers, id0, id1, cw)
    f1_tox = f1_score(target, pred)

    diag_post = run_diagnostics("post-training (best checkpoint)")

    if RUN_CKA:
        d = diag_post["cka"] - diag_pre["cka"]
        print("\nCKA change (post - pre) by layer:")
        print("  " + " ".join(f"L{i}:{x:+.3f}{'*' if i in align_layers else ''}" for i, x in enumerate(d)))
    if RUN_PROBE:
        print("\nProbe change at aligned layers (post - pre):")
        for l in align_layers:
            print(f"  L{l}: macro-F1 {diag_post[f'probe_macro_f1_L{l}'] - diag_pre[f'probe_macro_f1_L{l}']:+.4f} | "
                  f"offensive F1 {diag_post[f'probe_toxic_f1_L{l}'] - diag_pre[f'probe_toxic_f1_L{l}']:+.4f}")

    print(f"\nComputing {N_BOOTSTRAP:,}-sample bootstrap CIs ({CI_LEVEL:.0%})...")
    ci_results = bootstrap_ci(target, pred)
    diff_results = None
    if BASELINE_PREDS_PATH is not None:
        base = np.load(BASELINE_PREDS_PATH)
        assert np.array_equal(base["targets"], target), "Baseline test split differs."
        diff_results = paired_bootstrap_diff(target, pred, base["preds"])

    print("\n" + "=" * 80 + "\nFINAL RESULTS\n" + "=" * 80)
    print(f"Accuracy:   {accuracy_score(target, pred):.4f}")
    print(f"F1 (Toxic): {f1_tox:.4f}")
    print(f"Macro-F1:   {test_f1:.4f}\n")
    print_ci_table(ci_results, f"Bootstrap {CI_LEVEL:.0%} CI ({N_BOOTSTRAP:,} resamples, n_test={len(target)}):")
    if diff_results is not None:
        print()
        print_ci_table(diff_results, "Paired difference vs baseline (this run − baseline):")
    if test_align:
        print("\nTest alignment quality: " + " ".join(f"{k}={v:.4f}" for k, v in test_align.items()))
    print("\nClassification report:")
    print(classification_report(target, pred, target_names=["Non-toxic", "Toxic"]))

    # ---------------- SAVE ----------------
    torch.save({k: v for k, v in model.state_dict().items() if is_lora_key(k)},
               os.path.join(run_dir, "lora.pt"))
    if heads is not None:
        torch.save(heads.state_dict(), os.path.join(run_dir, "align_heads.pt"))
    np.savez(os.path.join(run_dir, "test_predictions.npz"), preds=pred, targets=target)
    np.savez(os.path.join(run_dir, "diagnostics.npz"),
             **{f"pre_{k}": np.asarray(v) for k, v in diag_pre.items()},
             **{f"post_{k}": np.asarray(v) for k, v in diag_post.items()})
    torch.save({
        "base_model": LLAMA_MODEL, "teacher_model": EMB_MODEL, "method": "lora_embedding_alignment_teacher",
        "dataset": DATASET, "seed": RANDOM_SEED, "split_seed": SPLIT_SEED,
        "target_mode": ALIGN_TARGET_MODE, "max_train": MAX_TRAIN,
        "use_align": use_align, "align_layers": align_layers,
        "lambda": [ALIGN_LAMBDA_START, ALIGN_LAMBDA_END], "align_only_steps": ALIGN_ONLY_STEPS,
        "w_nce": ALIGN_W_NCE, "w_rel": ALIGN_W_REL, "w_cos": ALIGN_W_COS,
        "tau": ALIGN_TAU, "rel_tau_t": ALIGN_REL_TAU_T, "false_neg_thresh": FALSE_NEG_THRESH,
        "center_targets": CENTER_TARGETS, "head_mlp": HEAD_MLP, "unlabeled_path": UNLABELED_PATH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_target_layers": LORA_TARGET_LAYERS,
        "best_val_macro_f1": best_val_macro_f1, "test_macro_f1": test_f1, "test_toxic_f1": f1_tox,
        "bootstrap": {"n": N_BOOTSTRAP, "ci": CI_LEVEL, "results": ci_results,
                      "diff_vs_baseline": diff_results},
    }, os.path.join(run_dir, "config.pt"))
    print(f"\nSaved to: {run_dir}")


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if max(ALIGN_LAMBDA_START, ALIGN_LAMBDA_END) > 0:
        parts = [n for n, w in (("nce", ALIGN_W_NCE), ("rel", ALIGN_W_REL), ("cos", ALIGN_W_COS)) if w > 0]
        where = ALIGN_LAYERS if ALIGN_LAYERS is not None else ALIGN_LAYER_FRACS
        tag = f"align_{'+'.join(parts)}_lam{ALIGN_LAMBDA_START}-{ALIGN_LAMBDA_END}_L{'-'.join(map(str, where))}"
        if UNLABELED_PATH:
            tag += "_unl"
    else:
        tag = "baseline_lambda0"
    if ALIGN_TARGET_MODE == "shuffle":
        tag += "_SHUFFLED-TARGETS"
    if MAX_TRAIN:
        tag += f"_n{MAX_TRAIN}"
    tag = f"{DATASET}_s{RANDOM_SEED}_{tag}"
    run_dir = os.path.join(SAVE_ROOT, f"run_{ts}_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = open(os.path.join(run_dir, "train_log.txt"), "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    try:
        _main_inner(run_dir)
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()