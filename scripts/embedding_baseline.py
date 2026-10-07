#!/usr/bin/env python
# coding=utf-8

# ==============================================================================
# title           : embedding_baseline.py
# description     : Answer-level embeddings and baseline coherence metrics for comparison with Optuna search solutions (parameters for topic modelling)
#                   - Optuna search scores every model with "embedding coherence" (C_embed), i.e., for each topic, the mean cosine similarity between the 
#                   embeddings of every pair of answers in the topic, averaged over the topics - but all answers are about the same film, so any two answers 
#                   are already inherently similar
#                   - script computes an overall and group baselines to compare with the C_embed, assisting the decision of best parameters for topic modelling:
#                       1) the mean cosine similarity between ANY two answers of the dataset (random "null" model) - C_embeb must be clearly above it for 
#                          model to be informative
#                       2) the score a "model" would get if its topics were simply the participants or the three moments (computed like C_embed) - a topic model
#                          that does not score above, has most likely not found more structure than who is speaking, or which scene is discussed
#                          (to be checked with topic content in analyse_topics.py)
#
# output          : <dataset>_embeddings.npy          float32 array, one row per answer, in the order of the dataset's rows (as in MOSAIC)
#                   <dataset>_embeddings_index.csv    row number -> answer_id, participant_id, moment (no interview text)
#                   <dataset>_embeddings_meta.json    model, number of rows, dimensions, date, fingerprint of the texts
#                   <dataset>_similarity_baseline.csv the baselines, one row each
#                   <dataset>_similarity_baseline.txt the same, readable, with the interpretation
#
# usage           : cd $PROJECT/MOSAIC
#                   export CUDA_VISIBLE_DEVICES=0
#                   python $PROJECT/scripts/embedding_baseline.py --dataset gutslei_answers_full
#                   adding --device cpu - avoids GPU entirelys
#
# dependencies    : Python  : >= 3.10 ; 
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-10-07
# version         : 1.0
# ==============================================================================

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# MOSAIC runs offline and only reads the models stored in $HF_HOME; same here
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pandas as pd

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # MOSAIC's default
TEXT_COLUMNS = ["cleaned_reflection", "reflection_answer", "text", "cleaned_text"]  # MOSAIC's order
ID_COLUMNS = ["answer_id", "participant_id", "moment", "moment_source", "segment"]  # copied to the index (no text)

# Function to load the .csv dataset and returns a table with texts as list
def load_texts(csv_path: Path, text_col: str | None):
    """Read the dataset as MOSAIC's optuna_search.py does: rows without text are dropped, order kept."""
    df = pd.read_csv(csv_path)
    col = text_col or next((c for c in TEXT_COLUMNS if c in df.columns), None)
    if col is None or col not in df.columns:
        sys.exit(f"ERROR: no text column in {csv_path.name} (looked for: {text_col or ', '.join(TEXT_COLUMNS)})")
    df = df[df[col].notna()].reset_index(drop=True)
    return df, col, df[col].tolist()

# Function to set the embedding model according to MOSAIC config
def model_from_config(dataset: str) -> str:
    """Embedding model named in the dataset's MOSAIC config (as with --use-config), else MOSAIC's default."""
    try:
        config = importlib.import_module(f"mosaic.configs.{dataset}").config
        return getattr(config, "transformer_model", DEFAULT_MODEL)
    except (ImportError, AttributeError):
        print(f"  ! Config mosaic.configs.{dataset} not found: using the default model.")
        return DEFAULT_MODEL

#Function to check txts and their order, to save embeddings appropriately
def fingerprint(texts: list[str]) -> str:
    """Short identifier of the exact texts and their order, to know when saved embeddings are still valid."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]

# Function to turn the texts into vectors with the embedding model
def embed(texts: list[str], model_name: str, device: str | None) -> np.ndarray:
    """Same call as MOSAIC: SentenceTransformer(model).encode(texts), default settings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, show_progress_bar=True), dtype=np.float32) # one row per text (answer), one column per dimension (1024 for Qwen model)

# Function to compute how similar every pair of texts is
def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows. It measures the angle between two vectors: 
    1 means the same direction (very similar meaning in semantic space), 0 unrelated"""
    x = emb.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1.0, norms)
    return x @ x.T

# Function to compute the average similarity between any two answers in the dataset -
def overall_baseline(sim: np.ndarray) -> dict:
    """Similarity between any two different answers (upper triangle of the matrix). This gives a reference point: how similar answers are in general"""
    v = sim[np.triu_indices(len(sim), k=1)]
    return {"baseline": "any two answers", "groups": 1, "pairs": int(v.size),
            "mean_similarity": float(v.mean()), "sd": float(v.std(ddof=1)), "sd_describes": "pair_similarities",
            "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "between_groups_mean": np.nan}


def group_baseline(sim: np.ndarray, labels: pd.Series, name: str) -> dict | None:
    """MOSAIC's embedding coherence if the topics were the groups of `labels`:
    the mean pairwise similarity inside each group, averaged over the groups
    (groups with one answer are skipped, as in MOSAIC). Also the mean
    similarity between answers of different groups."""
    lab = labels.astype(str).to_numpy()
    means, n_pairs = [], 0
    for g in pd.unique(lab):
        idx = np.flatnonzero(lab == g)
        if len(idx) < 2:
            continue
        sub = sim[np.ix_(idx, idx)]
        v = sub[np.triu_indices(len(idx), k=1)]
        means.append(v.mean())
        n_pairs += v.size
    if not means:
        return None
    upper = np.triu(np.ones(sim.shape, dtype=bool), k=1)
    different = upper & (lab[:, None] != lab[None, :])
    between = float(sim[different].mean()) if different.any() else np.nan
    return {"baseline": f"same {name}", "groups": len(means), "pairs": int(n_pairs),
            "mean_similarity": float(np.mean(means)), "sd": float(np.std(means, ddof=1)) if len(means) > 1 else np.nan, "sd_describes": "per-group means",
            "p05": np.nan, "median": float(np.median(means)), "p95": np.nan,
            "between_groups_mean": between}


def report(rows: list[dict], dataset: str, model_name: str, n: int, dim: int) -> str:
    overall = rows[0]["mean_similarity"]
    lines = [f"Embedding similarity baseline: {dataset}",
             f"{n} answers, model {model_name} ({dim} dimensions), cosine similarity", "",
             f"Any two answers          mean {overall:.3f}   spread (sd) {rows[0]['sd']:.3f}   "
             f"5%-95% {rows[0]['p05']:.3f} to {rows[0]['p95']:.3f}   ({rows[0]['pairs']} pairs)"]
    for r in rows[1:]:
        lines.append(f"{r['baseline'].capitalize():<24} mean {r['mean_similarity']:.3f}   "
                     f"different: {r['between_groups_mean']:.3f}   ({r['groups']} groups, "
                     f"mean of the per-group means, as MOSAIC scores topics)")
    lines += ["",
              "How to read MOSAIC's embedding coherence (first value of each Optuna trial):",
              f"  - {overall:.3f} is what random groups of answers would score. Only the part above",
              "    it reflects structure found by the topic model.",
              f"  - example: a trial with embedding coherence 0.550 is {0.550 - overall:+.3f} from that baseline."]
    if len(rows) > 1:
        lines += ["  - the 'same ...' lines are what a model would score if its topics were simply",
                  "    those groups. A model near those values may be separating speakers or scenes",
                  "    rather than kinds of experience: check the topics' spread over participants",
                  "    and moments (analyse_gutslei.py topics)."]
    lines += ["  - the sd is the spread of the pairwise similarities, not the uncertainty of the mean:",
              "    pairs share answers and are not independent, so do not build a significance test",
              "    or a confidence interval on it."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Save answer embeddings and compute the baseline similarity "
                                            "for MOSAIC's embedding coherence.")
    p.add_argument("--dataset", default="gutslei_answers_full",
                   help="dataset name, as given to optuna_search.py (default: gutslei_answers_full)")
    p.add_argument("--data-dir", default="DATA/preprocessed",
                   help="folder with <dataset>_preprocessed.csv (default: DATA/preprocessed)")
    p.add_argument("--out", default="DATA/derivatives/embeddings",
                   help="output folder (default: DATA/derivatives/embeddings)")
    p.add_argument("--model", default=None,
                   help="embedding model (default: transformer_model of the dataset's MOSAIC config)")
    p.add_argument("--text-col", default=None, help="text column (default: as MOSAIC, cleaned_reflection first)")
    p.add_argument("--by", nargs="*", default=["participant_id", "moment"],
                   help="columns for the group baselines (default: participant_id moment; none: --by)")
    p.add_argument("--device", default=None, help="'cpu' to avoid the GPU (default: GPU if available)")
    p.add_argument("--recompute", action="store_true", help="compute the embeddings again even if saved ones match")
    args = p.parse_args()

    csv_path = Path(args.data_dir) / f"{args.dataset}_preprocessed.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Run this from the MOSAIC folder, or give --data-dir.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / f"{args.dataset}_embeddings.npy"
    meta_path = out / f"{args.dataset}_embeddings_meta.json"

    df, col, texts = load_texts(csv_path, args.text_col)
    if len(texts) < 2:
        sys.exit(f"ERROR: {csv_path.name} has fewer than 2 answers with text.")
    model_name = args.model or model_from_config(args.dataset)
    fp = fingerprint(texts)
    print(f"Dataset: {csv_path}  ({len(texts)} answers, column '{col}')")
    print(f"Model:   {model_name}")

    # --- embeddings: reuse if the same texts and model, else compute ---
    emb = None
    if emb_path.exists() and meta_path.exists() and not args.recompute:
        meta = json.loads(meta_path.read_text())
        if meta.get("texts_fingerprint") == fp and meta.get("model") == model_name:
            emb = np.load(emb_path)
            print(f"Reusing saved embeddings: {emb_path}")
        else:
            print("Saved embeddings belong to other texts or another model: computing again.")
    if emb is None:
        print("Generating embeddings...")
        emb = embed(texts, model_name, args.device)
        np.save(emb_path, emb)
        meta_path.write_text(json.dumps({
            "dataset": args.dataset, "data_file": str(csv_path), "text_column": col, "model": model_name,
            "n_rows": int(emb.shape[0]), "dimensions": int(emb.shape[1]), "dtype": str(emb.dtype),
            "texts_fingerprint": fp, "created": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
        print(f"Saved: {emb_path}  {emb.shape}")
    if emb.shape[0] != len(texts):
        sys.exit("ERROR: number of embeddings and answers differ. Run again with --recompute.")

    index = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    index.insert(0, "row", range(len(index)))
    index.to_csv(out / f"{args.dataset}_embeddings_index.csv", index=False)

    # --- baselines ---
    sim = cosine_matrix(emb)
    rows = [overall_baseline(sim)]
    for c in args.by:
        if c not in df.columns:
            print(f"  ! Column '{c}' not in the dataset: skipped.")
            continue
        r = group_baseline(sim, df[c], c)
        if r:
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / f"{args.dataset}_similarity_baseline.csv", index=False)
    text = report(rows, args.dataset, model_name, len(texts), emb.shape[1])
    (out / f"{args.dataset}_similarity_baseline.txt").write_text(text)
    print("\n" + text)
    print(f"Written to: {out}/")


if __name__ == "__main__":
    main()