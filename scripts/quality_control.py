#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : quality_control.py
# description     : Quality control of preprocessing interview transcripts before topic modelling
#                   - audits transcripts and preprocessing applied by preprocessing.py before texts are passed to MOSAIC ((https://github.com/romybeaute/MOSAIC), a 
#                   BERTopic-based pipeline for analysing first-person experiential reports.
#                   - imports parsing, segmentation, answer-selection and moment-filling functions directly from preprocessing.py
#                   - only reads transcripts, no modification
#                   - check 1 - Moments: parse timestamped speaker turns
#                   - step 2 - Removal of Introduction: removal of all turns up to the start of experiential description (most intense moment) - discarding 
#                   readiness replies (short turns) 
#                   - step 3 - Segmentation into moments: moment 1 (most intense moment, after the introduction, can be any scene), moment 2 (celebration) 
#                   and moment 3 (woman, final scene of the movie) - taking into account backchannel words + most intense moment correspoding to moment 2 or 3
#                   - step 4 - Answers selection: select only answers from the participant (audit file with corresponding questions is generated); merging 
#                   some turns into single answer (if separated by interviewer backchannel words like "um", "yeah" or if .../- snd signals for continuation, when
#                   short continuation interruption from the interviewer - max 3 words)
#                   - step 5 - Text cleaning (for participant's text only): remove non-verbal annotations, hesitations, keep answers with min. 3 words (default)
#                   - step 6 - Experience vs narrative tagging: every sentence of kept answer is tagged as experiential (feelings, bodily sensations, perception,
#                   attention, thoughts, evaluations, generic you/we), narrative (film content only like characters and events), mixed (containing both) or other 
#                   (neither) + narrative share calculation and answer type classification - used for interpreting topics and needs manual validation TODO
#                   - no language model is used; all steps are rule-based (regular expressions)
# output          : preprocessed data for analysis - .csv files (one dataset for full interview plus one per moment - extra .txt files for inspection), audit .csv file and segmentation report .csv file 
# dependencies    : Python  : >= 3.10 ; pandas and nltk installation
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-09-30
# version         : 1.0
# ==============================================================================



import argparse
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_gutslei as prep  # noqa: E402
 
# ----------------------------------------------------------------------------
# Optional dependencies (all have fallbacks)
# ----------------------------------------------------------------------------
try:
    from nltk.tokenize import PunktSentenceTokenizer  # same as MOSAIC
    _PUNKT = PunktSentenceTokenizer()
    SENT_SPLITTER = "nltk Punkt (as MOSAIC)"
 
    def split_sentences(text):
        return _PUNKT.tokenize(text)
except ImportError:
    SENT_SPLITTER = "regex fallback (install nltk to match MOSAIC)"
 
    def split_sentences(text):
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
 
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set("""a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves im ive dont didnt wasnt its thats
    theres isnt couldnt wouldnt""".split())
 
try:
    from wordfreq import zipf_frequency
 
    def is_rare(word):
        return zipf_frequency(word, "en") < 2.0
    RARE_METHOD = "wordfreq (Zipf < 2.0)"
except ImportError:
    is_rare = None
    RARE_METHOD = "skipped (pip install wordfreq to enable)"
 
# Rough English conversion for the answer-length check (1 token ~ 0.75 words)
TOKENS_PER_WORD = 1.4
 
# ----------------------------------------------------------------------------
# Word lists (edit freely)
# ----------------------------------------------------------------------------
FILLERS = {
    "hesitation": prep.HESITATIONS,  # same list the prepare script removes
    "discourse": [r"you know", r"i mean", r"kind of", r"sort of", r"i guess", r"okay", r"ok", r"yeah",
                  r"well", r"actually", r"basically", r"let'?s say"],
    "like (ambiguous)": [r"like"],  # filler OR comparison ('like a heaviness') -> inspect, don't remove blindly
}
FILLER_RES = {cat: re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.I) for cat, pats in FILLERS.items()}
REPETITION_RE = re.compile(r"\b(\w+)(?:[\s,.-]+\1\b)+", re.I)
WORD_RE = re.compile(r"[a-z]+(?:['’-][a-z]+)*")
 
GUIDE_PHRASES_RE = re.compile(
    r"how (?:were|are|was) you feeling|what were you paying attention|how was your body|"
    r"another moment|when(?:ever)? you feel ready|do you have any questions|"
    r"so if i understand|to recap|let me summari[sz]e",
    re.I,
)
 
 
# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_sec(ts):
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s
 
 
def words(text):
    return WORD_RE.findall(text.lower())
 
 
class Flags:
    def __init__(self):
        self.rows = []
 
    def add(self, file, pid, check, severity, detail):
        self.rows.append({"file": file, "participant_id": pid, "check": check,
                          "severity": severity, "detail": detail})
 
 
def raw_structure_scan(path, known_speakers):
    """Things parse_transcript() silently tolerates: text before the first
    timestamp, and speaker labels inside a turn (= missing timestamp)."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    before_first_ts, hidden_labels = [], []
    seen_ts, after_ts = False, False
    label_re = re.compile(r"^\s*(" + "|".join(re.escape(s) for s in known_speakers) + r")\s*:", re.I) \
        if known_speakers else None
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if prep.TIMESTAMP_RE.match(line):
            seen_ts, after_ts = True, True
            continue
        if not seen_ts:
            before_first_ts.append((n, line.strip()))
        elif after_ts:
            after_ts = False  # this is the speaker line
        elif label_re and label_re.match(line):
            hidden_labels.append((n, line.strip()))
    return before_first_ts, hidden_labels
 
 
# ----------------------------------------------------------------------------
# Per-file QC
# ----------------------------------------------------------------------------
def qc_file(path, is_participant, args, flags, store):
    fname = path.name
    turns = prep.parse_transcript(path)
    if not turns:
        flags.add(fname, None, "parsing", "ERROR", "no timestamped turns found")
        return
 
    speakers = Counter(t["speaker"] for t in turns)
    # one participant per transcript: the first speaker matching the pattern
    pid = next((t["speaker"] for t in turns if is_participant(t["speaker"])), None)
 
    # ---------------- 2. SPEAKERS ----------------
    spk_words = Counter()
    for t in turns:
        spk_words[t["speaker"]] += len(words(t["text"]))
    total_w = sum(spk_words.values()) or 1
    for s in speakers:
        store["speakers"].append({"file": fname, "speaker": s,
                                  "role": "participant" if is_participant(s) else "other",
                                  "turns": speakers[s], "words": spk_words[s],
                                  "word_share": round(spk_words[s] / total_w, 3)})
 
    if pid is None:
        flags.add(fname, None, "speakers", "ERROR",
                  f"no speaker matches participant pattern; speakers = {sorted(speakers)}")
        return
    if "UNKNOWN" in speakers:
        flags.add(fname, pid, "speakers", "WARN", f"{speakers['UNKNOWN']} turn(s) with timestamp but no 'Name:' label")
    if is_participant(turns[0]["speaker"]):
        flags.add(fname, pid, "speakers", "INFO", "the participant speaks first (the interviewer usually does)")
    id_in_name = re.search(r"gutslei\d+", fname, re.I)
    if id_in_name and id_in_name.group(0).lower() not in pid.lower():
        flags.add(fname, pid, "speakers", "ERROR", f"filename ID '{id_in_name.group(0)}' != speaker ID '{pid}'")
    p_share = spk_words[pid] / total_w
    if p_share < args.min_participant_share:
        flags.add(fname, pid, "speakers", "WARN",
                  f"participant speaks only {p_share:.0%} of words (labels swapped? very short answers?)")
 
    before, hidden = raw_structure_scan(path, list(speakers.keys() - {"UNKNOWN"}))
    if before:
        flags.add(fname, pid, "parsing", "WARN",
                  f"{len(before)} line(s) before first timestamp are ignored, e.g. line {before[0][0]}: '{before[0][1][:60]}'")
    for n, line in hidden:
        flags.add(fname, pid, "parsing", "ERROR",
                  f"speaker label inside a turn (missing timestamp? two speakers merged) at line {n}: '{line[:70]}'")
 
    for t in turns:
        if is_participant(t["speaker"]) and GUIDE_PHRASES_RE.search(t["text"]):
            flags.add(fname, pid, "speakers", "WARN",
                      f"interview-guide phrasing in participant turn at {t['start']}: '{t['text'][:70]}' (label swap?)")
 
    prev = -1
    for t in turns:
        s, e = to_sec(t["start"]), to_sec(t["end"])
        if e < s:
            flags.add(fname, pid, "timestamps", "WARN", f"end before start at {t['start']} - {t['end']}")
        if s < prev:
            flags.add(fname, pid, "timestamps", "WARN", f"timestamp goes backwards at {t['start']}")
        prev = s
 
    # ---------------- 1. MOMENTS ----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg_turns, info = prep.segment(turns, is_participant, fname)
    mrow = {"file": fname, "participant_id": pid,
            "interviewer": prep.main_interviewer(turns, is_participant), "intro_found": info["intro_found"],
            "moment2_start": info["moment2_start"], "moment3_start": info["moment3_start"]}
    if seg_turns is None:
        flags.add(fname, pid, "moments", "ERROR",
                  "'most intense' not found in an interviewer turn -> file excluded by prepare script")
        mrow["status"] = "EXCLUDED (no 'most intense')"
        store["moments"].append(mrow)
        return
 
    intro_end = max((i for i, t in enumerate(seg_turns) if t["moment"] == "intro"), default=-1)
    mrow["kept_from_time"] = info["intro_end_time"]
    mrow["intro_turns"] = intro_end + 1
    mrow["evocation_replies"] = info["n_evocation_replies"]
    first_kept = next((t for t in seg_turns if t["moment"] == 1 and is_participant(t["speaker"])
                       and not t["evocation"]), None)
    if first_kept is not None:
        mrow["first_answer_time"] = first_kept["start"]
        mrow["first_answer"] = first_kept["text"][:120]
    if intro_end + 1 > 0.5 * len(seg_turns):
        flags.add(fname, pid, "moments", "WARN",
                  f"intro covers {intro_end + 1}/{len(seg_turns)} turns - 'most intense' said late?")
 
    starts = {m: next((i for i, t in enumerate(seg_turns) if t["moment"] == m), None) for m in (1, 2, 3)}
    for idx in prep.unclassified_transitions(seg_turns, is_participant):
        flags.add(fname, pid, "moments", "WARN",
                  f"unclassified transition phrase at {seg_turns[idx]['start']}: "
                  f"'{seg_turns[idx]['text'][:80]}' (new wording? add it to MOMENT2_RE / MOMENT3_RE)")
 
    answers = prep.build_answers(seg_turns, is_participant, args.min_words, not args.keep_hesitations)
 
    # duration & words per moment
    last_end = to_sec(seg_turns[-1]["end"])
    present = []
    for m in (1, 2, 3):
        if starts[m] is None:
            mrow[f"m{m}_duration_s"] = 0
            mrow[f"m{m}_words"] = 0
            continue
        nxt = [starts[k] for k in (2, 3) if k > m and starts[k] is not None]
        end_s = to_sec(seg_turns[min(nxt)]["start"]) if nxt else last_end
        mrow[f"m{m}_duration_s"] = end_s - to_sec(seg_turns[starts[m]]["start"])
        mrow[f"m{m}_words"] = sum(a["n_words"] for a in answers if a["moment"] == m and a["kept"])
        if mrow[f"m{m}_words"] > 0:
            present.append(m)
    total_kept = sum(mrow[f"m{m}_words"] for m in (1, 2, 3)) or 1
    for m in present:
        share = mrow[f"m{m}_words"] / total_kept
        if mrow[f"m{m}_words"] < args.min_moment_words or share < 0.10:
            flags.add(fname, pid, "moments", "WARN",
                      f"moment {m} is very short ({mrow[f'm{m}_words']} words, {share:.0%}) - boundary matched too early/late?")
 
    dec = prep.decide_moment_sources(seg_turns, answers, is_participant, args.no_fill)
    mrow["moments_present"] = ",".join(map(str, present))
    mrow["n_moments"] = len(present)
    mrow["m1_scene"], mrow["m1_scene_confidence"] = dec["scene"], dec["scene_conf"]
    mrow["m1_scenes"] = dec["scenes"]
    mrow["m1_scene_evidence"] = dec["scene_evidence"]
    mrow["moment2_source"], mrow["moment3_source"] = dec["source"][2], dec["source"][3]
    copied = [m for m in (2, 3) if dec["source"][m] == "copied_from_moment1"]
    if len(present) == 3:
        mrow["status"] = "complete"
    elif copied:
        mrow["status"] = f"2 moments (m{copied[0]} = copy of m1)"
    else:
        odd = [f"m{m}: {dec['source'][m]}" for m in (2, 3) if dec["source"][m] != "own"]
        mrow["status"] = f"{len(present)} moment(s); " + "; ".join(odd)
    for sev, msg in dec["notes"]:
        flags.add(fname, pid, "moments", sev, msg)
    store["moments"].append(mrow)
 
    # ---------------- 3/4/5. TEXT: sentences, words, totals ----------------
    for m in (1, 2, 3):
        m_ans = [a for a in answers if a["moment"] == m]
        kept = [a for a in m_ans if a["kept"]]
        sents = []
        for a in kept:
            store["answers"].append({"file": fname, "participant_id": pid, "moment": m,
                                     "answer_id": a.get("answer_id"), "n_words": a["n_words"],
                                     "approx_tokens": round(a["n_words"] * TOKENS_PER_WORD),
                                     "answer": a["clean"]})
            for s in split_sentences(a["clean"]):
                nw = len(words(s))
                sents.append(nw)
                store["sentences"].append({"file": fname, "participant_id": pid, "moment": m,
                                           "sentence": s, "n_words": nw,
                                           "kept_by_mosaic": nw >= args.min_sentence_words})
        text = " ".join(a["clean"] for a in kept)
        store["totals"].append({
            "participant_id": pid, "moment": m,
            "answers_total": len(m_ans), "answers_kept": len(kept),
            "sentences_total": len(sents),
            "sentences_kept": sum(n >= args.min_sentence_words for n in sents),
            "words_all_answers": sum(a["n_words"] for a in m_ans),
            "words_kept": sum(a["n_words"] for a in kept),
            "annotations_removed": sum(a["n_annotations"] for a in m_ans),
            "evocation_replies_dropped": sum(a["evocation"] for a in m_ans),
            "hesitations_removed": 0 if args.keep_hesitations else sum(a["n_hesitations"] for a in m_ans),
        })
        # fillers & repetitions, measured on the ORIGINAL participant text (annotations
        # removed, hesitations kept), i.e. before hesitation removal and length filter
        full_text = " ".join(prep.ANNOTATION_RE.sub(" ", a["raw"]) for a in m_ans)
        frow = {"participant_id": pid, "moment": m, "words": len(words(full_text))}
        for cat, rx in FILLER_RES.items():
            frow[cat] = len(rx.findall(full_text))
        frow["repetitions"] = len(REPETITION_RE.findall(full_text))
        store["fillers"].append(frow)
        store["texts"][(pid, m)] = text
 
 
# ----------------------------------------------------------------------------
# Aggregation & report
# ----------------------------------------------------------------------------
def describe(series):
    s = pd.Series(series)
    if s.empty:
        return "n=0"
    g = lambda x: f"{x:.2f}" if s.max() <= 1 else f"{x:.0f}"  # shares vs counts
    return (f"n={len(s)}, mean={s.mean():.2f}, median={g(s.median())}, min={g(s.min())}, "
            f"p5={g(s.quantile(.05))}, p95={g(s.quantile(.95))}, max={g(s.max())}")
 
 
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 description="Quality checks of interview transcripts before topic modelling "
                                             "(see file header for details).")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="qc")
    ap.add_argument("--participant-regex", default=prep.DEFAULT_PARTICIPANT_RE,
                    help="same option as prepare script (must match the whole label)")
    ap.add_argument("--min-words", type=int, default=3, help="answer filter (same as prepare script)")
    ap.add_argument("--keep-hesitations", action="store_true", help="same flag as prepare script")
    ap.add_argument("--no-fill", "--no-fill-m3", dest="no_fill", action="store_true",
                    help="same flag as prepare script")
    ap.add_argument("--min-sentence-words", type=int, default=3, help="MOSAIC sentence filter (min_word_count)")
    ap.add_argument("--long-sentence", type=int, default=60, help="flag sentences longer than this")
    ap.add_argument("--min-moment-words", type=int, default=30)
    ap.add_argument("--min-participant-share", type=float, default=0.30)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--long-answer-tokens", type=int, default=512,
                    help="report answers longer than about this many tokens (default %(default)s)")
    ap.add_argument("--per-participant-dir", default=None,
                    help="Also write <dir>/<participant>/<participant>_qc_flags.csv "
                         "(e.g. the 'participants' folder of prepare_gutslei.py)")
    ap.add_argument("--check-options", action="store_true", help=argparse.SUPPRESS)
    args, unknown = ap.parse_known_args()
    # run_gutslei.sh passes the same options to both scripts: options of prepare_gutslei.py are
    # ignored here; anything else is a typo and stops the run (instead of silently using a default)
    PREPARE_ONLY = {"--mode", "--prefix", "--data-dir", "--input-file", "--participant",
                    "--participants-file", "--combine"}
    bad = [u for u in unknown if u.startswith("-") and u.split("=")[0] not in PREPARE_ONLY]
    if bad:
        sys.exit(f"Unknown option(s): {' '.join(bad)} (see --help)")
    if args.check_options:  # options are valid: stop here
        sys.exit(0)
 
    files = sorted(Path(args.input_dir).rglob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {args.input_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    part_re = re.compile(args.participant_regex, re.I)
    is_participant = lambda s: bool(part_re.fullmatch(s))  # the whole speaker label
    flags = Flags()
    store = defaultdict(list)
    store["texts"] = {}
 
    for f in files:
        qc_file(f, is_participant, args, flags, store)
 
    moments = pd.DataFrame(store["moments"])
    speakers = pd.DataFrame(store["speakers"])
    totals = pd.DataFrame(store["totals"])
    sents = pd.DataFrame(store["sentences"])
    answers_df = pd.DataFrame(store["answers"])
    fillers = pd.DataFrame(store["fillers"])
 
    # duplicate IDs across files
    if not moments.empty:
        for pid, grp in moments.groupby("participant_id"):
            if len(grp) > 1:
                flags.add(", ".join(grp["file"]), pid, "speakers", "ERROR", "same participant ID in several files")
 
    # ---- totals: add 'all' rows per participant and summary per moment
    if not totals.empty:
        all_rows = totals.groupby("participant_id", as_index=False).sum(numeric_only=True)
        all_rows["moment"] = "all"
        totals = pd.concat([totals, all_rows], ignore_index=True)
        totals["retention"] = (totals["words_kept"] / totals["words_all_answers"].replace(0, float("nan")).astype(float)).round(3)
        totals = totals.sort_values(["participant_id", "moment"], key=lambda c: c.astype(str))
        low = totals[(totals["moment"] == "all") & (totals["retention"] < 0.8)]
        for _, r in low.iterrows():
            flags.add(None, r["participant_id"], "totals", "INFO",
                      f"only {r['retention']:.0%} of participant words kept after min-words filter")
 
    # ---- sentence flags
    if not sents.empty:
        for _, r in sents[sents["n_words"] > args.long_sentence].iterrows():
            flags.add(r["file"], r["participant_id"], "sentences", "INFO",
                      f"long sentence ({r['n_words']} words, moment {r['moment']}): '{r['sentence'][:60]}...'")
 
    # ---- fillers: rates
    if not fillers.empty:
        fall = fillers.groupby("participant_id", as_index=False).sum(numeric_only=True)
        fall["moment"] = "all"
        fillers = pd.concat([fillers, fall], ignore_index=True)
        for cat in list(FILLERS) + ["repetitions"]:
            fillers[f"{cat}_per100w"] = (100 * fillers[cat] / fillers["words"].replace(0, float("nan")).astype(float)).round(2)
 
    # ---- word frequencies on kept text
    content_rows, rare_counter, rare_example = [], Counter(), {}
    per_moment = {m: Counter() for m in (1, 2, 3)}
    doc_freq = Counter()
    filler_words = {"um", "uh", "uhm", "erm", "er", "ah", "eh", "hmm", "mm", "mhm", "okay", "ok", "yeah",
                    "like", "well", "actually", "basically"}
    pids = {pid for pid, _ in store["texts"]}
    for pid in pids:
        seen = set()
        for m in (1, 2, 3):
            text = store["texts"].get((pid, m), "")
            for w in words(text):
                if (w in STOPWORDS or w in filler_words or len(w) < 3
                        or re.split(r"['’]", w)[0] in STOPWORDS):  # it's, don't, i'm ...
                    continue
                per_moment[m][w] += 1
                seen.add(w)
                if is_rare and is_rare(w):
                    rare_counter[w] += 1
                    rare_example.setdefault(w, (pid, m, re.search(rf".{{0,40}}\b{re.escape(w)}\b.{{0,40}}", text, re.I)))
        doc_freq.update(seen)
    overall = sum(per_moment.values(), Counter())
    n_p = len(pids) or 1
    for w, c in overall.most_common():
        content_rows.append({"word": w, "count_all": c, "count_m1": per_moment[1][w], "count_m2": per_moment[2][w],
                             "count_m3": per_moment[3][w], "n_participants": doc_freq[w],
                             "participant_share": round(doc_freq[w] / n_p, 2)})
    content = pd.DataFrame(content_rows)
    rare = pd.DataFrame([{"word": w, "count": c, "example_participant": rare_example[w][0],
                          "example_moment": rare_example[w][1],
                          "context": rare_example[w][2].group(0) if rare_example[w][2] else ""}
                         for w, c in rare_counter.most_common()])
 
    # ---- write CSVs
    flags_df = pd.DataFrame(flags.rows, columns=["file", "participant_id", "check", "severity", "detail"])
    sev_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    flags_df = flags_df.sort_values(["severity", "file"], key=lambda c: c.map(sev_order) if c.name == "severity" else c.astype(str))
    for name, df in [("flags", flags_df), ("moments", moments), ("speakers", speakers), ("totals", totals),
                     ("answers", answers_df), ("sentences", sents), ("fillers", fillers),
                     ("words_content", content), ("words_rare", rare)]:
        df.to_csv(out / f"qc_{name}.csv", index=False)
 
    # ---- per-participant flag files
    if args.per_participant_dir and not flags_df.empty:
        root = Path(args.per_participant_dir)
        # folder = existing participant folder named like the file (run_gutslei.sh names the
        # selected transcripts <ID>.txt), else the speaker ID found in the transcript
        def folder_for(row):
            stem = Path(str(row["file"])).stem if pd.notna(row["file"]) else None
            if stem and (root / stem).is_dir():
                return stem
            return row["participant_id"] if pd.notna(row["participant_id"]) else None
        keyed = flags_df.assign(_folder=flags_df.apply(folder_for, axis=1)).dropna(subset=["_folder"])
        for key, grp in keyed.groupby("_folder"):
            pdir = root / str(key)
            pdir.mkdir(parents=True, exist_ok=True)
            grp.drop(columns="_folder").to_csv(pdir / f"{key}_qc_flags.csv", index=False)
 
    # ---- summary
    L = []
    L.append("=" * 72)
    L.append(f"QC SUMMARY  -  {len(files)} file(s) in {args.input_dir}")
    L.append("=" * 72)
    L.append(f"Sentence splitter: {SENT_SPLITTER} | rare-word check: {RARE_METHOD}")
    L.append(f"Filters: answers >= {args.min_words} words, sentences >= {args.min_sentence_words} words")
 
    L.append("\n--- FLAGS ---")
    if flags_df.empty:
        L.append("No issues flagged.")
    else:
        L.append(flags_df.groupby(["severity", "check"]).size().rename("n").reset_index().to_string(index=False))
        L.append("(details in qc_flags.csv - fix ERRORs before modelling)")
 
    L.append("\n--- MOMENTS ---")
    if not moments.empty:
        L.append(moments["status"].value_counts().to_string())
        if "interviewer" in moments and moments["interviewer"].nunique() > 1:
            # markers were written from one interviewer's phrasing; others may word things differently
            L.append("\nstatus by interviewer (unexpected structures concentrated in one interviewer "
                     "suggest phrasing the moment rules do not cover):")
            L.append(pd.crosstab(moments["interviewer"].fillna("?"), moments["status"]).to_string())
        for m in (1, 2, 3):
            col = f"m{m}_duration_s"
            if col in moments:
                d = moments.loc[moments[col] > 0, col]
                if len(d):
                    L.append(f"moment {m} duration (s): median {d.median():.0f}, range {d.min()}-{d.max()}")
 
    L.append("\n--- SPEAKERS ---")
    if not speakers.empty:
        other = speakers[speakers["role"] == "other"]["speaker"].value_counts()
        L.append("interviewers: " + ", ".join(f"{k} ({v} files)" for k, v in other.items()))
        ps = speakers[speakers["role"] == "participant"]["word_share"]
        L.append(f"participant word share: {describe(ps)}")
 
    L.append("\n--- TOTALS (kept text) ---")
    if not totals.empty:
        agg = totals.groupby(totals["moment"].astype(str)).agg(
            participants=("participant_id", lambda s: (totals.loc[s.index, "words_kept"] > 0).sum()),
            answers_kept=("answers_kept", "sum"), sentences_total=("sentences_total", "sum"),
            sentences_kept=("sentences_kept", "sum"), words_kept=("words_kept", "sum"),
            words_per_participant_median=("words_kept", "median"))
        L.append(agg.to_string())
        L.append(f"overall retention of participant words: "
                 f"{totals.loc[totals['moment'] == 'all', 'words_kept'].sum() / max(totals.loc[totals['moment'] == 'all', 'words_all_answers'].sum(), 1):.0%}")
 
    L.append("\n--- ANSWERS (units of the answer-level datasets) ---")
    if not answers_df.empty:
        L.append(f"all:      {describe(answers_df['n_words'])}  (words)")
        for m in (1, 2, 3):
            L.append(f"moment {m}: {describe(answers_df.loc[answers_df['moment'] == m, 'n_words'])}")
        long_ = answers_df[answers_df["approx_tokens"] > args.long_answer_tokens]
        L.append(f"answers over ~{args.long_answer_tokens} tokens (check the embedding model's max_seq_length): "
                 f"{len(long_)}; longest ~{answers_df['approx_tokens'].max()} tokens")
        L.append(f"answers per participant: {describe(answers_df.groupby('participant_id').size())}")
 
    L.append("\n--- SENTENCES ---")
    if not sents.empty:
        L.append(f"all:      {describe(sents['n_words'])}")
        for m in (1, 2, 3):
            L.append(f"moment {m}: {describe(sents.loc[sents['moment'] == m, 'n_words'])}")
        L.append(f"< {args.min_sentence_words} words (dropped by MOSAIC): {(~sents['kept_by_mosaic']).sum()} | "
                 f"< 5 words: {(sents['n_words'] < 5).sum()} | > {args.long_sentence} words: "
                 f"{(sents['n_words'] > args.long_sentence).sum()}")
        L.append(f"sentences without final . ! ? : {(~sents['sentence'].str.strip().str[-1].isin(list('.!?'))).sum()}")
 
    L.append("\n--- FILLERS (per 100 words, all moments) ---")
    if not fillers.empty:
        fa = fillers[fillers["moment"] == "all"]
        tot_w = fa["words"].sum() or 1
        for cat in list(FILLERS) + ["repetitions"]:
            L.append(f"{cat:18s}: total {fa[cat].sum():5d} | {100 * fa[cat].sum() / tot_w:.2f} per 100 words | "
                     f"per participant max {fa[f'{cat}_per100w'].max()}")
 
    L.append(f"\n--- TOP {args.top_n} CONTENT WORDS (no stopwords/fillers) ---")
    if not content.empty:
        L.append(", ".join(f"{r.word} ({r.count_all})" for r in content.head(args.top_n).itertuples()))
        common = content[(content["participant_share"] >= 0.5) & (content.index < 200)]
        if len(common):
            L.append("\nUsed by >= 50% of participants (candidate domain stopwords - review, don't auto-remove):")
            L.append(", ".join(f"{r.word} ({r.participant_share:.0%})" for r in common.head(40).itertuples()))
 
    L.append("\n--- RARE / UNKNOWN WORDS (possible transcription errors, names, non-English) ---")
    if is_rare is None:
        L.append(RARE_METHOD)
    elif not rare.empty:
        L.append(", ".join(f"{r.word} ({r.count})" for r in rare.head(40).itertuples()))
    else:
        L.append("none")
 
    L.append(f"\nAll tables written to {out.resolve()}")
    summary = "\n".join(L)
    (out / "qc_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
 
 
if __name__ == "__main__":
    main()