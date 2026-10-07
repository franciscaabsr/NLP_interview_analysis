#!/usr/bin/env python
# coding=utf-8

# TODO CORRECT THIS SCRIPT 
# checks each solution's embedding coherence minues the baseline (any two answers)
# checks the topic count (without outlier group)
# marks the most balanced solution
# gives table per 50 trials to show whether the search had settled


"""
summarise_optuna.py -- Summary of a MOSAIC Optuna search: the Pareto front with
trial numbers and topic counts, compared with the similarity baseline (GUTSLEI study)
===========================================================================

Authors : [AUTHOR NAMES]
Paper   : [PAPER REFERENCE / DOI]
Version : 1.0
License : [LICENSE, e.g. MIT]
Python  : >= 3.10 ; dependencies: pandas

Overview
--------
MOSAIC's optuna_search.py fits one topic model per trial and scores it on two
objectives, both to be maximised:
  - embedding_coherence: mean cosine similarity between the answers of a
    topic, averaged over the topics;
  - objective_cv: the C_v coherence of the topics' keywords (MOSAIC's column
    name: "objective" = a score the search maximises, "cv" = C_v).
There is no single best trial when the two scores disagree. The search
therefore returns the Pareto front: every trial that no other trial beats on
both scores at once. Its size is not chosen; it is however many such
trade-offs the trials contain.

MOSAIC prints the front without trial numbers or topic counts, and saves only
the per-trial table. This script reads that table and writes:
  - the Pareto front, sorted from "best keyword coherence" to "best embedding
    coherence", with trial number, number of topics and settings;
  - each solution's embedding coherence relative to the baseline computed by
    embedding_baseline.py (if that has been run);
  - a "balanced" solution: the one closest to the best value of both scores
    after rescaling them to 0-1 over the front. This is a starting point for
    inspection, not a decision: the final choice is made by reading the topics;
  - whether the search had settled: the best scores and the number of front
    solutions per generation of trials (NSGA-II works in generations of 50).

Usage (from the MOSAIC folder, environment active)
--------------------------------------------------
    cd $PROJECT/MOSAIC
    python $PROJECT/scripts/summarise_optuna.py --dataset gutslei_answers_full

Outputs (in --out, default: the search's own results folder, so that the
summary is deleted together with the results it describes)
---------------------------------------------------------
<dataset>_optuna_summary.txt    the readable summary (also printed)
<dataset>_pareto_front.csv      the front, one row per solution

Note: n_topics is MOSAIC's count, which includes the outlier group (-1) when
there is one; "topics" in this summary is n_topics - 1.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PARAMS = ["n_components", "n_neighbors", "min_dist", "min_cluster_size", "min_samples"]
EC, CV = "embedding_coherence", "objective_cv"


def find_results(results_dir: Path, results: str | None) -> Path:
    if results:
        path = Path(results)
        if not path.exists():
            sys.exit(f"ERROR: {path} not found.")
        return path
    files = sorted(results_dir.glob("*_results.csv"))
    if not files:
        sys.exit(f"ERROR: no *_results.csv in {results_dir}. Run this from the MOSAIC folder, after the search.")
    if len(files) > 1:
        sys.exit("ERROR: several result files, choose one with --results:\n  " + "\n  ".join(map(str, files)))
    return files[0]


def pareto_mask(d: pd.DataFrame) -> list[bool]:
    """True for trials that no other trial beats: at least as good on both scores and better on one."""
    a, b = d[EC].to_numpy(), d[CV].to_numpy()
    return [not (((a >= a[i]) & (b >= b[i])) & ((a > a[i]) | (b > b[i]))).any() for i in range(len(d))]


def read_baseline(path: Path) -> dict:
    """Baselines written by embedding_baseline.py: {'any two answers': 0.41, 'same moment': 0.45, ...}."""
    if not path.exists():
        return {}
    b = pd.read_csv(path)
    return dict(zip(b["baseline"], b["mean_similarity"]))


def main():
    p = argparse.ArgumentParser(description="Summarise a MOSAIC Optuna search (Pareto front, baseline, convergence).")
    p.add_argument("--dataset", default="gutslei_answers_full",
                   help="dataset name, as given to optuna_search.py (default: gutslei_answers_full)")
    p.add_argument("--results-dir", default=None, help="folder of the search (default: results/optuna/<dataset>)")
    p.add_argument("--results", default=None, help="the *_results.csv file itself, if the folder holds several")
    p.add_argument("--baseline", default=None,
                   help="baseline CSV of embedding_baseline.py "
                        "(default: DATA/derivatives/embeddings/<dataset>_similarity_baseline.csv)")
    p.add_argument("--generation", type=int, default=50, help="trials per generation of the search (default: 50)")
    p.add_argument("--out", default=None, help="output folder (default: the results folder)")
    args = p.parse_args()

    results_dir = Path(args.results_dir or f"results/optuna/{args.dataset}")
    results_path = find_results(results_dir, args.results)
    out = Path(args.out) if args.out else results_path.parent
    out.mkdir(parents=True, exist_ok=True)

    d = (pd.read_csv(results_path).dropna(subset=[EC, CV])
         .drop_duplicates("trial_number").reset_index(drop=True))
    if d.empty:
        sys.exit(f"ERROR: no completed trials in {results_path}.")
    d["topics"] = d["n_topics"] - 1
    n_started = int(d["trial_number"].max()) + 1
    baselines = read_baseline(Path(args.baseline or
                                   f"DATA/derivatives/embeddings/{args.dataset}_similarity_baseline.csv"))
    overall = baselines.get("any two answers")

    # --- Pareto front ---
    d["on_front"] = pareto_mask(d)
    front = (d[d["on_front"]].drop_duplicates([EC, CV] + PARAMS)
             .sort_values(CV, ascending=False).reset_index(drop=True))
    front.insert(0, "solution", range(1, len(front) + 1))
    if overall is not None:
        front["above_baseline"] = front[EC] - overall
    # balanced solution: closest to the best of both scores, each rescaled to 0-1 over the front
    span = {c: (front[c].max() - front[c].min()) or 1.0 for c in (EC, CV)}
    dist = sum(((front[c].max() - front[c]) / span[c]) ** 2 for c in (EC, CV)) ** 0.5
    front["balanced"] = ["<--" if i == dist.idxmin() and len(front) > 2 else "" for i in front.index]
    cols = (["solution", "trial_number", CV, EC] + (["above_baseline"] if overall is not None else [])
            + ["topics"] + PARAMS + ["balanced"])
    front[cols].to_csv(out / f"{args.dataset}_pareto_front.csv", index=False)

    # --- per generation ---
    d["generation"] = d["trial_number"] // args.generation + 1
    gen = d.groupby("generation").agg(first_trial=("trial_number", "min"), last_trial=("trial_number", "max"),
                                      completed=("trial_number", "size"), best_cv=(CV, "max"), best_emb=(EC, "max"),
                                      median_topics=("topics", "median"),
                                      front_solutions=("on_front", "sum")).reset_index()

    # --- text ---
    L = [f"Optuna search summary: {args.dataset}", f"Results file: {results_path}", "",
         f"Trials: {len(d)} completed of {n_started} started"
         + ("" if len(d) == n_started else f"  ({n_started - len(d)} not completed: the model could not be fitted or scored)"),
         f"Scores over all trials:  C_v {d[CV].min():.3f} to {d[CV].max():.3f}   "
         f"embedding coherence {d[EC].min():.3f} to {d[EC].max():.3f}   "
         f"topics {int(d['topics'].min())} to {int(d['topics'].max())} (median {d['topics'].median():.0f})", ""]
    if baselines:
        L += ["Similarity baselines (embedding_baseline.py):"]
        L += [f"  {k:<22} {v:.3f}" for k, v in baselines.items()]
        L += ["  'above_baseline' below = embedding coherence minus 'any two answers'.", ""]
    else:
        L += ["No similarity baseline found: run embedding_baseline.py to compare the embedding coherence.", ""]
    n_front_trials = int(d["on_front"].sum())
    L += [f"Pareto front: {len(front)} solutions, from best keyword coherence (C_v) to best embedding coherence"
          + ("" if n_front_trials == len(front) else
             f"\n({n_front_trials} trials are on the front; trials with identical settings and scores are listed once)"),
          front[cols].round(3).to_string(index=False), "",
          "Reading the front:",
          "  - no solution beats another on both scores: moving down the list trades keyword",
          "    coherence for tighter groups of answers (usually more, smaller topics).",
          "  - 'topics' excludes the outlier group. The share of answers left as outliers is not",
          "    saved by the search: it is seen when a solution is fitted again.",
          "  - '<--' marks the most balanced solution (closest to the best of both scores). It is a",
          "    starting point: choose by fitting 2-3 solutions and reading their topics.", "",
          f"Did the search settle? (generations of {args.generation} trials)",
          gen.round(3).to_string(index=False), ""]
    last = gen.iloc[-1]
    if len(gen) > 1:
        share = last["front_solutions"] / max(d["on_front"].sum(), 1)
        L += [f"  {int(last['front_solutions'])} of {int(d['on_front'].sum())} front trials come from the last generation "
              f"({share:.0%}). " + ("The search was still improving: consider a fresh search with more trials."
                                    if share > 0.4 else "The front was mostly found earlier: more trials are "
                                                        "unlikely to change it much.")]
    text = "\n".join(L) + "\n"
    (out / f"{args.dataset}_optuna_summary.txt").write_text(text)
    print(text)
    print(f"Written to: {out}/")


if __name__ == "__main__":
    main()