"""
Zero-shot toxicity/hate-speech classification — Sinhala code-mixed.

Split out of the original notebook so this runs as its own process. That on
its own removes the main cause of the OOM (two ~multi-billion-parameter
models being loaded into the same session back-to-back, with the notebook's
accelerator set to "none" so both landed in CPU RAM). On top of that, this
script:
  - detects the device and warns if there's no GPU
  - loads with low_cpu_mem_usage=True to cut peak RAM during loading
  - wraps batched generation in a helper that auto-halves the batch and
    retries on a CUDA OOM instead of crashing the run
  - periodically clears the CUDA cache during the evaluation loop

pip install torch transformers scikit-learn pandas numpy tqdm
"""

import gc
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CSV_PATH = "sentence-level-annotation.csv"
# NOTE: the original notebook read 'sentence-level-annotation.csv' (no path,
# no " (1)") while the dataset listing showed the file above — adjust this
# to match whatever your dataset mount actually names the file.
# MODEL_NAME = "google/gemma-4-E2B-it" 
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen3.5-2B"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# NOTE: this name isn't one we recognize —
# if it 404s, it was likely meant to be "google/gemma-3n-E2B-it" (Gemma 3n).
BATCH_SIZE = 8
MAX_LENGTH = 512
MAX_NEW_TOKENS = 3
N_BOOTSTRAP = 10_000
USE_4BIT = False  # set True (needs `pip install bitsandbytes`) for extra headroom on small GPUs

TEST_SIZE = 0.20  # held out, then split 50/50 into val/test (10%/10%)
VAL_SPLIT = 0.50
RANDOM_STATE = 42

LABEL_MAP = {
    "Not offensive": 0,
    "Abusive": 1,
    "Hate-Inducing": 1,
}

hf_token = "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"
from huggingface_hub import login
login(token=hf_token)
# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)
    print("\nLabel distribution (original):")
    print(df["Hate_speech"].value_counts())

    df["label"] = df["Hate_speech"].map(LABEL_MAP)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    sentences = df["Sentence"].astype(str).tolist()
    labels = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        sentences, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\n--- Data Split Sizes ---")
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(sentences)*100:.1f}%)")
    print(f"Val set:   {len(X_val)} samples ({len(X_val)/len(sentences)*100:.1f}%)")
    print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(sentences)*100:.1f}%)")
    print(
        f"\nTest Set Label Distribution:\n0 (Non-toxic): {sum(y_test == 0)}\n"
        f"1 (Toxic):     {sum(y_test == 1)}"
    )

    return X_test, y_test


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no GPU detected — running a multi-billion-parameter "
            "model on CPU. This is slow and the most common cause of OOM; "
            "enable a GPU accelerator if you can."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(low_cpu_mem_usage=True)
    if device == "cuda":
        load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["dtype"] = torch.float32

    if USE_4BIT:
        from transformers import BitsAndBytesConfig

        load_kwargs.pop("dtype", None)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    print(f"Loading {model_name} on {device}...")
    # model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tokenizer, device


# ----------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------
def format_prompt(tokenizer, text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert NLP content moderator for Sinhala. "
                "Classify the following text as either 0 (non-offensive) or "
                "1 (abusive/hate-speech). Answer ONLY with a single digit: "
                "0 or 1."
            ),
        },
        {"role": "user", "content": f"Text: {text}\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ----------------------------------------------------------------------
# OOM-safe batched generation
# ----------------------------------------------------------------------
def generate_batch_safe(model, tokenizer, prompts):
    """Generate for one batch; on a CUDA OOM, halve the batch and retry
    instead of letting the whole run die."""
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        del inputs, outputs
        return decoded
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        print(f"  OOM at batch size {len(prompts)} — retrying as {mid} + {len(prompts) - mid}")
        return generate_batch_safe(model, tokenizer, prompts[:mid]) + generate_batch_safe(
            model, tokenizer, prompts[mid:]
        )


def evaluate(model, tokenizer, X_test, device):
    preds = []
    print("Evaluating on Sinhala Test Set...")
    for i in tqdm(range(0, len(X_test), BATCH_SIZE)):
        batch = X_test[i : i + BATCH_SIZE]
        prompts = [format_prompt(tokenizer, t) for t in batch]
        decoded_texts = generate_batch_safe(model, tokenizer, prompts)

        for text in decoded_texts:
            match = re.search(r"[01]", text)
            preds.append(int(match.group()) if match else 0)

        if device == "cuda" and (i // BATCH_SIZE) % 20 == 0:
            torch.cuda.empty_cache()

    return np.array(preds)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, ci=95, seed=42):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    accs, f1_macros, f1_toxics, precisions, recalls = [], [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress undefined-F1 warnings on degenerate resamples
        for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
            idx = rng.integers(0, n, n)
            yt, yp = y_true[idx], y_pred[idx]
            accs.append(accuracy_score(yt, yp))
            f1_macros.append(f1_score(yt, yp, average="macro"))
            f1_toxics.append(f1_score(yt, yp, pos_label=1))
            precisions.append(precision_score(yt, yp, pos_label=1))
            recalls.append(recall_score(yt, yp, pos_label=1))

    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2

    def summarize(vals):
        vals = np.array(vals)
        return vals.mean(), np.percentile(vals, lo), np.percentile(vals, hi)

    return {
        "accuracy": summarize(accs),
        "f1_macro": summarize(f1_macros),
        "f1_toxic": summarize(f1_toxics),
    }


def report(y_test, preds, boot):
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_toxic = f1_score(y_test, preds, pos_label=1)

    print("\n" + "=" * 45)
    print("   MINISTRAL 3B - SINHALA TEST EVALUATION   ")
    print("=" * 45)
    print(f"Accuracy:    {acc:.6f}")
    print(f"F1 (Macro):  {f1_macro:.6f}")
    print(f"F1 (Toxic):  {f1_toxic:.6f}\n")
    print(classification_report(y_test, preds, target_names=["Not offensive (0)", "Abusive/Hate (1)"], digits=6))

    print("\n" + "=" * 40)
    print(f"   BOOTSTRAP 95% CONFIDENCE INTERVALS (n={N_BOOTSTRAP:,})   ")
    print("=" * 40)
    for name, (mean, lo, hi) in boot.items():
        print(f"{name:10s}: {mean:.6f}  [{lo:.6f}, {hi:.6f}]")


# ----------------------------------------------------------------------
def main():
    X_test, y_test = load_data(CSV_PATH)
    model, tokenizer, device = load_model(MODEL_NAME)

    preds = evaluate(model, tokenizer, X_test, device)
    boot = bootstrap_ci(y_test, preds)
    report(y_test, preds, boot)

    # Free the model explicitly rather than waiting on process exit.
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
