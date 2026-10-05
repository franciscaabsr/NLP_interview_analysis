#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : quality_control.py
# description     : Quality control of preprocessing interview transcripts before topic modelling
#                   - audits transcripts and preprocessing applied by preprocessing.py before texts are passed to MOSAIC ((https://github.com/romybeaute/MOSAIC), a 
#                   BERTopic-based pipeline for analysing first-person experiential reports.
#                   - imports parsing, segmentation, answer-selection and moment-filling functions directly from preprocessing.py
#                   - only reads transcripts, no modification
#                   - check 1 - Moments: 
#                                   1) beginning of interview (start of kept part, share of turns removed before it, number of leading readiness replies dropped,
#                                first real answer and timestamp)
#                                      Two warnings: NO introduction removed ("most intense" in the very first turn: the introduction was not caught, or the transcript 
#                                      starts mid-interview), and MORE THAN HALF of the interview removed as introduction ("most intense" probably found too late, e.g. 
#                                      in a recap at the end rather than in the actual prompt);
#                                   2) start timestamps, duration and word count of each moment
#                                   3) interview structure (3 moments or 2 moments with skipped moment copied from moment 1 or unexpected - missing/ambiguous)
#                                   4) scene of moment 1 (most intense) with confidence, keyword evidence and consistency with copied moment
#                                   5) transitions phrases not matched by any moment rule
#                                   6) moments that are very short in words or relative to the interview: fewer than --min-moment-words kept words (default 30) or less 
#                                 than 10% of the interview's kept words
#                                   7) interview structure per interviewer (moment rules are based on interviewer phrasing thus may differ between interviewers)
#                   - check 2 - Speaker labels and file structure:
#                                   1) speakers per file with number of turns, words and share of words
#                                   2) 1 participant speaker in every file, matching ID in file name, no ID use in more than one file
#                                   3) interviews in which the participant speakes first (might not even happen, but just in case)
#                                   4) timestamps without speaker label, text before first timestamp, speaker labels inside turn (should not happen at all)
#                                   5) interview-guide phrasing inside participant turns (possible label swap) and low participant share of words (very unlikely)
#                                   6) timestamps that decrease or end before start (will not happen for sure) 
#                   - check 3 - Answers and sentence splitting:
#                                   1) answers (units) - length distribution per moment and overall, answers per participant, answers longer than --long-answer-tokens 
#                                (defaul 512), which embedding model could trunctate - MOSAIC default embedding model (Qwen3-Embedding-0.6B) takes until 32768 tokens,
#                                 still worth inspection (several experiences in one unit)
#                                   2) sentences (for participant-level datasets) - retained answer split into sentences as MOSAIC does (NLTK Punkt), giving sentence-
#                                 -length distribution per moment and overall, sentences below the MOSAIC minimum length (--min-sentence-word; MOSAIC drops sentences
#                                 under 3 words, counted by spaces, only when it splits into sentences, i.e. optuna_search.py --sentences), very long sentences
#                                 (--long-sentence; sign of missing punctuation) and sentences without final punctuation
#                   - check 4 - Word frequencies:
#                                   1) number of fillers per 100 words, counted in ALL participant answers as spoken (including dropped backchannels and readines; 
#                                 considering text before hesitation removal and never including interviewer turns) - hesitations, discourse markers, and "like" (can be 
#                                 filler or comparison)
#                                   2) immediate word repetitions (e.g. I I think)
#                                   3) content-word frequencies, counted in RETAINED answers only (dropped backchannels, readiness replies and participant questions are
#                                 not counted; stopwords, fillers and acknowledgement words excluded), overall and per moment, with the share of participant using each
#                                 word - words used by at least half of participants listed as candidate domain stopwords for manual review
#                                   4) rare or unknown english words with a context snippet, pointing to transcription errors, names or non-English words - IMPORTANT
#                   - check 5 - Totals (per participant and moment, and aggregated per moment and overall)
#                                   1) answers (all and retained), sentences (all and retained), words, removed annotations and hesitations, and retention (share of 
#                                 participant words kept after minimum-length filter)
#                                   - copied moments (2-moment interviews) are not included in the totalsper participant and moment, and aggregated per moment and worText cleaning 
#                                 (for participant's text only): remove non-verbal annotations, hesitations, keep answers with min. 3 words (default)
#                                   - All word counts (answers, sentences, totals, filler rates) use the preprocessing script's n_words, so they are on one scale; only the MOSAIC sentence 
#                                 filter uses MOSAIC's own count.
#                   - every issue is recorder in qc_flags.csv with severity level (ERROR to be fixed before modelling, WARN to be inspected, INFO for information)
#                   - no language model is used; all steps are rule-based (regular expressions)
# output          : qc_summary.txt        overview of all checks (also printed to the console)
#                   qc_flags.csv          every issue: file, participant, check, severity, detail
#                   qc_moments.csv        one row per file: markers, structure, scene, durations
#                   qc_speakers.csv       one row per file and speaker
#                   qc_totals.csv         one row per participant and moment (including "all")
#                   qc_answers.csv        every retained answer (the units of the answer-level datasets), with length in words and approximate tokens
#                   qc_sentences.csv      every sentence of the retained answers, with length
#                   qc_fillers.csv        filler and repetition counts and rates
#                   qc_words_content.csv  content-word frequencies, overall and per moment
#                   qc_words_rare.csv     rare / unknown words with an example context With --per-participant-dir, each participant's flags are also written to 
#                                         <dir>/<participant>/<participant>_qc_flags.csv.
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
import preprocessing as prep  # noqa: E402
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 1 - Tools used for quality control (optional dependencies, all have fallbacks) and word patterns
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Sentence splitting: the same function as preprocessing.py (NLTK Punkt as in MOSAIC, or its fallback)
split_sentences = prep.split_sentences
try:
    import nltk.tokenize  # noqa: F401  (only to report which splitter is used)
    SENT_SPLITTER = "nltk Punkt (as MOSAIC)"
except ImportError:
    SENT_SPLITTER = "regex fallback (install nltk to match MOSAIC)"

# Stopwords - to be excluded from the content-word frequencies
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
    theres isnt couldnt wouldnt 
    don didn doesn isn wasn weren aren couldn wouldn shouldn hasn haven hadn ain mustn needn mightn shan""".split()) # words are split as whole contractions and dropped if the part before is stopword

# Contractions written without an apostrophe in transcripts ("dont", "im"), for either list
STOPWORDS |= set("""im ive id youre youve theyre weve dont didnt doesnt isnt wasnt werent arent
couldnt wouldnt shouldnt hasnt havent hadnt wont cant aint its thats theres whats""".split())
# keep "won" (the victory in the celebration scene) as a content word, but still drop "won't"
STOPWORDS = (STOPWORDS - {"won"}) | {"won't"}

# Rare or unknown words in English
try:
    from wordfreq import zipf_frequency  # knows how common words are in English - base10 log
 
    def is_rare(word):
        return zipf_frequency(word, "en") < 2.0
    RARE_METHOD = "wordfreq (Zipf < 2.0)"
except ImportError: # no reasonable fallback
    is_rare = None
    RARE_METHOD = "skipped (pip install wordfreq to enable)"
 
# Token estimation - rough English conversion for the answer-length check (1 token ~ 0.75 words)
TOKENS_PER_WORD = 1.4

# Fillers including hesitations, discourse markers (little content) and like (ambiguous case) - counts are approximate as several markers are also ordinary words.
FILLERS = {
    "hesitation": prep.HESITATIONS,  # same list the preprocessing script removes
    "discourse": [r"you know", r"i mean", r"i guess", r"actually", r"basically", r"let['’]?s say",
                  # "kind of"/"sort of" as hedge, not after a/the/this/... ("a kind of warmth")
                  r"(?<!\ba )(?<!\bthe )(?<!\bthis )(?<!\bthat )(?<!\bsome )(?<!\bwhat )(?<!\bany )"
                  r"(?<!\bone )(?<!\bsame )(?:kind|sort) of",
                  # "well" as discourse marker, not "as well", "felt well", "very well", "well-being"
                  r"(?<!\bas )(?<!\bvery )(?<!\bquite )(?<!\bpretty )(?<!\bnot )(?<!\bfelt )(?<!\bfeel )"
                  r"(?<!\bfeeling )(?<!\bdoing )well(?!-)"],
    "acknowledgement (ambiguous)": [r"okay", r"ok", r"yeah"],  # filler OR a real answer ('yeah, it was scary')
    "like (ambiguous)": [r"like"],  # filler OR comparison ('like a heaviness') -> inspect, don't remove blindly
}
FILLER_RES = {cat: re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.I) for cat, pats in FILLERS.items()}

#Words left out of the content-word frequencies (besides stopwords). FILLERS above are PATTERNS used to count fillers, including phrases ("you know", "kind of"); content words are counted per
# single word, so this list holds the single words that are fillers or acknowledgements:
#   - every word matching a hesitation pattern (mhm, hmmm, ...): HESITATION_WORD_RE;
#   - the one-word discourse markers and the ambiguous words of FILLERS;
#   - the preprocessing script's acknowledgement words (BACKCHANNEL_WORDS), except words that can also describe experience ("see", "good"; "i" is a stopword anyway).
# The parts of phrases ("know" in "you know", "kind" in "kind of") are NOT excluded: as single words they are often content ("I didn't know", "he was kind").
HESITATION_WORD_RE = re.compile(r"(?:" + "|".join(prep.HESITATIONS) + r")", re.I)
NON_CONTENT_WORDS = ({"actually", "basically", "well", "okay", "ok", "yeah", "like"}
                     | (set(prep.BACKCHANNEL_WORDS) - {"i", "see", "good"}))
 
def is_content_word(w):
    """True if word w (lowercase, from words()) counts in the content-word frequencies."""
    return not (len(w) < 3 or w in STOPWORDS or w in NON_CONTENT_WORDS
                or HESITATION_WORD_RE.fullmatch(w)
                or w.split("'")[0] in STOPWORDS) 

# Immediate repetitions of a word
REPETITION_RE = re.compile(r"\b(\w+)(?:[\s,.-]+\1\b)+", re.I)
# Splitting text into words for frequency counts (including words with internal apostrophes or hyphens)
WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")

# Typical phrases only the interviewer says (not complete list) - flag potential label swap 
GUIDE_PHRASES_RE = re.compile(
    r"how (?:were|are|was) you feeling|what (?:were|are) you paying attention|how was your body|"
    r"what do you notice|(?:can|could) you describe|close your eyes|take your time|"
    r"do you have any questions|let['’]?s (?:go|move) (?:on )?to (?:another|the next|a different)|"    r"how was your experience|how did you experience|"
    r"if i (?:understood|understand) correctly|so if i understand|to recap|let me summari[sz]e",
    re.I,
)
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 2 - Helper functions for the quality control script
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to turn a timestamp into seconds
def to_sec(ts):
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s

# Function to do word count exactly as MOSAIC's optuna_search.py 
def mosaic_word_count(text):
    """Word count exactly as MOSAIC's sentence filter in optuna_search.py --sentences: len(sentence.split()) (split on whitespace; punctuation-only tokens count too)."""
    return len(text.split())

# Function to lowercase text and return its words as a list 
def words(text):
    return WORD_RE.findall(text.lower().replace("’", "'"))
 
# Class that collects QC issues, which become qc_flags.csv after 
class Flags:
    def __init__(self):
        self.rows = []
 
    def add(self, file, pid, check, severity, detail):
        self.rows.append({"file": file, "participant_id": pid, "check": check,
                          "severity": severity, "detail": detail})
 
# Function to check parsing and report if text before first timestamp or speaker label inside a turn
def raw_structure_scan(path, known_speakers):
    """Things parse_transcript() silently tolerates: text before the first timestamp, and speaker labels inside a turn (= missing timestamp)."""
    # read lines like parse_transcript
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    before_first_ts, hidden_labels = [], [] # two flags
    seen_ts, after_ts = False, False # timestamp that already appeared and previous not-empty line was a timestamp, respectively
    # builds pattern matching any known speaker label at the start of a line
    label_re = re.compile(r"^\s*(" + "|".join(re.escape(s) for s in known_speakers) + r")\s*:", re.I) \
        if known_speakers else None
    for n, line in enumerate(lines, 1):
        if not line.strip(): # blank lines are skipped
            continue
        if prep.TIMESTAMP_RE.match(line): # timestamp recognised
            seen_ts, after_ts = True, True
            continue
        if not seen_ts: # line is text before first timestamp
            before_first_ts.append((n, line.strip())) # saved line number
        elif after_ts: # first line after timestamp is the speaker line so following lines are continuation lines
            after_ts = False  # this is the speaker line
        elif label_re and label_re.match(line): # continuation line that starts with speaker label, missing timestamp
            hidden_labels.append((n, line.strip()))
    return before_first_ts, hidden_labels
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 3 - Quality control script for one transcript
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def qc_file(path, is_participant, args, flags, store):
    """Parses and segments the file with preprocessing.py functions, checking what it does and flagging along the way"""
    fname = path.name
    turns = prep.parse_transcript(path) # parses the file
    if not turns: # no recognisable timestamps - nothing can be checked
        flags.add(fname, None, "parsing", "ERROR", "no timestamped turns found")
        return
    
    speakers = Counter(t["speaker"] for t in turns) # counts turns per speaker
    # one participant per transcript: the first speaker matching the pattern
    pid = next((t["speaker"] for t in turns if is_participant(t["speaker"])), None) # participant ID
 
    # ---------------- 2. SPEAKERS ----------------
    spk_words = Counter() 
    for t in turns: # counts words per spEaker
        spk_words[t["speaker"]] += prep.n_words(t["text"])
    total_w = sum(spk_words.values()) or 1 # all words in the file; none is 1 
    for s in speakers: # one row per speaker with turns, words and share of all words
        store["speakers"].append({"file": fname, "speaker": s,
                                  "role": "participant" if is_participant(s) else "other",
                                  "turns": speakers[s], "words": spk_words[s],
                                  "word_share": round(spk_words[s] / total_w, 3)})
    
    if pid is None: # no participant - error
        flags.add(fname, None, "speakers", "ERROR",
                  f"no speaker matches participant pattern; speakers = {sorted(speakers)}")
        return
    if "UNKNOWN" in speakers: # turns without speaker label
        flags.add(fname, pid, "speakers", "WARN", f"{speakers['UNKNOWN']} turn(s) with timestamp but no 'Name:' label")
    if is_participant(turns[0]["speaker"]): # participant speaking first (unusual, but not necessarily wrong)
        flags.add(fname, pid, "speakers", "INFO", "the participant speaks first (the interviewer usually does)")
    # compare the extracted IDs exactly (to not accept gutslei001 for gutslei0010)
    id_in_name = re.search(r"gutslei\d+", fname, re.I)
    id_in_pid = re.search(r"gutslei\d+", pid, re.I)
    if id_in_name and id_in_pid and id_in_name.group(0).lower() != id_in_pid.group(0).lower(): # check if ID in filename matches speaker label
        flags.add(fname, pid, "speakers", "ERROR", f"filename ID '{id_in_name.group(0)}' != speaker ID '{pid}'")    
    p_share = spk_words[pid] / total_w # participant's share of all words
    if p_share < args.min_participant_share:
        flags.add(fname, pid, "speakers", "WARN",
                  f"participant speaks only {p_share:.0%} of words (labels swapped? very short answers?)")
    # raw line scan to check for errors (before any cleaning or removal)
    before, hidden = raw_structure_scan(path, list(speakers.keys() - {"UNKNOWN"}))
    if before: # flag if lines before first timestamp
        flags.add(fname, pid, "parsing", "WARN",
                  f"{len(before)} line(s) before first timestamp are ignored, e.g. line {before[0][0]}: '{before[0][1][:60]}'")
    for n, line in hidden: # error per hidden label - transcript needs fixing!
        flags.add(fname, pid, "parsing", "ERROR",
                  f"speaker label inside a turn (missing timestamp? two speakers merged) at line {n}: '{line[:70]}'")

    for t in turns: # if interview-guide phrasing is found in participant turn
        if is_participant(t["speaker"]) and GUIDE_PHRASES_RE.search(t["text"]):
            flags.add(fname, pid, "speakers", "WARN",
                      f"interview-guide phrasing in participant turn at {t['start']}: '{t['text'][:70]}' "
                      f"(possible label swap - or the participant quoting the interviewer; check the transcript)")
    
    prev = -1
    for t in turns: # time stamp checks - end must not be before start, and sequence between timestamps
        s, e = to_sec(t["start"]), to_sec(t["end"])
        if e < s:
            flags.add(fname, pid, "timestamps", "WARN", f"end before start at {t['start']} - {t['end']}")
        if s < prev:
            flags.add(fname, pid, "timestamps", "WARN", f"timestamp goes backwards at {t['start']}")
        prev = s
 
    # ---------------- 1. MOMENTS ----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg_turns, info = prep.segment(turns, is_participant, fname) # segments transcript labelling sentences with moment
    # add file's row in moments table
    mrow = {"file": fname, "participant_id": pid,
            "interviewer": prep.set_interviewer(turns, is_participant), "intro_found": info["intro_found"],
            "moment2_start": info["moment2_start"], "moment3_start": info["moment3_start"]}
    if seg_turns is None: # no moment 1 - file cannot be segmented
        flags.add(fname, pid, "moments", "ERROR",
                  "'most intense' not found in an interviewer turn -> file excluded by preprocessing script")
        mrow["status"] = "EXCLUDED (no 'most intense')"
        store["moments"].append(mrow)
        return
    # index of last introduction turn - -1 if no introduction turns
    intro_end = max((i for i, t in enumerate(seg_turns) if t["moment"] == "intro"), default=-1)
    mrow["kept_from_time"] = info["moment1_start"] # where kept part begins
    mrow["intro_turns"] = intro_end + 1 # how many turns removed as introduction
    mrow["evocation_replies"] = info["n_evocation_replies"] # how many readiness replies flagged
    # first real answer in moment 1 (most intense)
    first_kept = next((t for t in seg_turns if t["moment"] == 1 and is_participant(t["speaker"])
                       and not t["evocation"]), None)
    if first_kept is not None:
        mrow["first_answer_time"] = first_kept["start"]
        mrow["first_answer"] = first_kept["text"][:120]
    if intro_end < 0:  # if introductio  has no turns
        flags.add(fname, pid, "moments", "WARN",
                  "no introduction removed: 'most intense' is in the very first turn - introduction not caught, "
                  "or the transcript starts mid-interview?")
    if intro_end + 1 > 0.5 * len(seg_turns): # if more than half of interview removed as introduction
        flags.add(fname, pid, "moments", "WARN",
                  f"intro covers {intro_end + 1}/{len(seg_turns)} turns (more than half) - 'most intense' "
                  f"probably found too late, e.g. in a recap instead of the actual prompt")    
    # first turn index of each moment
    starts = {m: next((i for i, t in enumerate(seg_turns) if t["moment"] == m), None) for m in (1, 2, 3)}
    for idx in prep.unclassified_transitions(seg_turns, is_participant): # one flag per unrecognised transition
        flags.add(fname, pid, "moments", "WARN",
                  f"unclassified transition phrase at {seg_turns[idx]['start']}: "
                  f"'{seg_turns[idx]['text'][:80]}' (new wording? add it to MOMENT2_RE / MOMENT3_RE)")
    # builds answers like the preprocessing script
    answers = prep.build_answers(seg_turns, is_participant, args.min_words, not args.keep_hesitations)
    for k, a in enumerate(answers, 1):  # same answer IDs as the preprocessing script (process_file)
        a["answer_id"] = f"{pid}_m{a['moment']}_a{k:03d}"
 
    # duration & words per moment
    last_end = to_sec(seg_turns[-1]["end"])
    present = []
    for m in (1, 2, 3):
        if starts[m] is None:
            mrow[f"m{m}_duration_s"] = 0
            mrow[f"m{m}_words"] = 0
            continue
        # moment ends at the start of the next moment or at end of interview
        nxt = [starts[k] for k in (2, 3) if k > m and starts[k] is not None]
        end_s = to_sec(seg_turns[min(nxt)]["start"]) if nxt else last_end  # if moment 2 is missing, moment 1 ends where moment 3 starts
        # duration in seconds, including interviewer time
        mrow[f"m{m}_duration_s"] = end_s - to_sec(seg_turns[starts[m]]["start"])
        # words in kept answers
        mrow[f"m{m}_words"] = sum(a["n_words"] for a in answers if a["moment"] == m and a["kept"])  # moment present only if it has kept words
        if mrow[f"m{m}_words"] > 0:
            present.append(m)
    total_kept = sum(mrow[f"m{m}_words"] for m in (1, 2, 3)) or 1  # total kept words
    for m in present:
        share = mrow[f"m{m}_words"] / total_kept
        if mrow[f"m{m}_words"] < args.min_moment_words or share < 0.10: # moment is short is has less than --min-moment-words or less than 10% of kept words
            flags.add(fname, pid, "moments", "WARN",
                      f"moment {m} is very short ({mrow[f'm{m}_words']} words, {share:.0%}) - boundary matched too early/late?")
    # moment sources and status
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
        odd = [f"m{m}: {dec['source'][m]}" for m in (1, 2, 3) if dec["source"][m] != "own"]
        mrow["status"] = f"{len(present)} moment(s); " + "; ".join(odd)    
    for sev, msg in dec["notes"]:
        flags.add(fname, pid, "moments", sev, msg)
    store["moments"].append(mrow)
 
    # ---------------- 3/4/5. TEXT: sentences, words, totals ----------------
    for m in (1, 2, 3): # for each moment, all answers, kept ones, and list of sentence lengths
        m_ans = [a for a in answers if a["moment"] == m] # all answers
        kept = [a for a in m_ans if a["kept"]] # kept answers
        sents = [] # wether MOSAIC keeps each sentence
        for a in kept: # one row per kept answer with estimated token count
            store["answers"].append({"file": fname, "participant_id": pid, "moment": m,
                                     "answer_id": a["answer_id"], "n_words": a["n_words"],
                                     "approx_tokens": round(a["n_words"] * TOKENS_PER_WORD),
                                     "answer": a["clean"]})
            for s in split_sentences(a["clean"]): # splits answers into sentences as MOSAIC does
                nw = prep.n_words(s)                      # same count as answers and totals
                kept_by_mosaic = mosaic_word_count(s) >= args.min_sentence_words
                sents.append(kept_by_mosaic)
                # saves sentence length
                store["sentences"].append({"file": fname, "participant_id": pid, "moment": m,
                                           "answer_id": a["answer_id"], "sentence": s, "n_words": nw,
                                           "kept_by_mosaic": kept_by_mosaic})    
        text = " ".join(a["clean"] for a in kept)  # moment's kept text for word frequencies
        store["totals"].append({   # one row per participant and moment
            "participant_id": pid, "moment": m,
            "answers_total": len(m_ans), "answers_kept": len(kept),
            "sentences_total": len(sents),
            "sentences_kept": sum(sents),
            "words_all_answers": sum(a["n_words"] for a in m_ans),
            "words_kept": sum(a["n_words"] for a in kept),
            "annotations_removed": sum(a["n_annotations"] for a in m_ans),
            "evocation_replies_dropped": sum(a["evocation"] for a in m_ans),
            "hesitations_removed": 0 if args.keep_hesitations else sum(a["n_hesitations"] for a in m_ans),
        })
        # fillers & repetitions, measured on the ORIGINAL participant text (annotations removed, hesitations kept), i.e. before hesitation removal and length filter
        full_text = " ".join(prep.remove_annotations(a["raw"]) for a in m_ans)
        frow = {"participant_id": pid, "moment": m, "words": prep.n_words(full_text)}
        for cat, rx in FILLER_RES.items(): # counts each filler category and repetitions
            frow[cat] = len(rx.findall(full_text))
        frow["repetitions"] = len(REPETITION_RE.findall(full_text))
        store["fillers"].append(frow) # stored filler words to compute rates (per 100 words) later
        store["texts"][(pid, m)] = text # stored kept text for content-word frequencies across participants later
 
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 4 - Aggregation of quality checks for each transcript, checks for all files together (duplicate IDs and word frequencies) and summary report
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to decide which folder a flag belongs to 
def folder_for(row, root):
    """Match listed ID with an existing participant folder. If it doesn't work usese speaker ID and None if neither is known."""
    stem = Path(str(row["file"])).stem if pd.notna(row["file"]) else None
    if stem and (root / stem).is_dir():
        return stem
    return row["participant_id"] if pd.notna(row["participant_id"]) else None

# Function to summarize list of numbers in one line for the summary report
def describe(series):
    """Returns count, mean, median, minimum, 5th and 9th percentile, and maximum."""
    s = pd.Series(series) # accepts list or existing column
    if s.empty:
        return "n=0"
    g = lambda x: f"{x:.2f}" if s.max() <= 1 else f"{x:.0f}"  # shares (values at most 1) vs counts (no decimals)
    return (f"n={len(s)}, mean={s.mean():.2f}, median={g(s.median())}, min={g(s.min())}, "
            f"p5={g(s.quantile(.05))}, p95={g(s.quantile(.95))}, max={g(s.max())}")
  
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False,   # does not accept shortened options
                                 description="Quality checks of interview transcripts before topic modelling "
                                             "(see file header for details).")
    # options shared with preprocessing.py
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="qc")
    ap.add_argument("--participant-regex", default=prep.DEFAULT_PARTICIPANT_RE,
                    help="same option as preprocessing script (must match the whole label)")
    ap.add_argument("--min-words", type=int, default=3, help="answer filter (same as preprocessing script)")
    ap.add_argument("--keep-hesitations", action="store_true", help="same flag as preprocessing script")
    ap.add_argument("--no-fill", "--no-fill-m3", dest="no_fill", action="store_true",
                    help="same flag as preprocessing script")
    # quality control thresholds
    ap.add_argument("--min-sentence-words", type=int, default=3, help="MOSAIC sentence filter (min_word_count)")
    ap.add_argument("--long-sentence", type=int, default=60, help="flag sentences longer than this")
    ap.add_argument("--min-moment-words", type=int, default=30,
                    help="warn when a moment has fewer kept words than this (or <10%% of the interview): "
                         "its boundary was probably matched too early or too late (default %(default)s)")
    ap.add_argument("--min-participant-share", type=float, default=0.30)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--long-answer-tokens", type=int, default=512,
                    help="report answers longer than about this many tokens (default %(default)s)")
    ap.add_argument("--per-participant-dir", default=None,
                    help="Also write <dir>/<participant>/<participant>_qc_flags.csv "
                         "(e.g. the 'participants' folder of preprocessing.py)")
    # validation switch for run_preproc.sh
    ap.add_argument("--check-options", action="store_true", help=argparse.SUPPRESS)
    args, unknown = ap.parse_known_args()
    # run_preproc.sh passes the same options to both scripts: options of preprocessing.py are ignored here; anything else is a typo and stops the run (instead of silently using a default)
    PREPARE_ONLY = {"--mode", "--prefix", "--data-dir", "--input-file", "--participant",
                    "--participants-file", "--combine"}
    bad = [u for u in unknown if u.startswith("-") and u.split("=")[0] not in PREPARE_ONLY]
    if bad:
        sys.exit(f"Unknown option(s): {' '.join(bad)} (see --help)")
    if args.check_options:  # options are valid: stop here
        sys.exit(0)

    # 1. Running the checks 
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # files cannot be found in output folder (in case --out-dir lies inside --input-dir)
    files = sorted(f for f in Path(args.input_dir).rglob("*.txt")
                   if out.resolve() not in f.resolve().parents)
    if not files:
        sys.exit(f"No .txt files in {args.input_dir}")
   
    # participant check
    part_re = re.compile(args.participant_regex, re.I)
    is_participant = lambda s: bool(part_re.fullmatch(s))  # the whole speaker label
    flags = Flags()
    store = defaultdict(list)
    store["texts"] = {}
    # runs checks on each file
    for f in files:
        try:
            qc_file(f, is_participant, args, flags, store)
        except Exception as e:  # error becomes a flag and QC continues with next file
            flags.add(f.name, None, "parsing", "ERROR", f"QC failed for this file ({type(e).__name__}: {e})")
    # turns each list of rows into a table to be saved later
    moments = pd.DataFrame(store["moments"])
    speakers = pd.DataFrame(store["speakers"])
    totals = pd.DataFrame(store["totals"])
    sents = pd.DataFrame(store["sentences"])
    answers_df = pd.DataFrame(store["answers"])
    fillers = pd.DataFrame(store["fillers"])
 
    # 2. Checks across files 
    if not moments.empty:
        for pid, grp in moments.groupby("participant_id"):
            if len(grp) > 1: # no ID in more than one file
                flags.add(", ".join(grp["file"]), pid, "speakers", "ERROR", "same participant ID in several files")
 
    # ---- totals: add 'all' rows per participant and summary per moment
    if not totals.empty:
        all_rows = totals.groupby("participant_id", as_index=False).sum(numeric_only=True)
        all_rows["moment"] = "all" # sums over the three moments
        totals = pd.concat([totals, all_rows], ignore_index=True)
        # the share of words kept after the filters
        totals["retention"] = (totals["words_kept"] / totals["words_all_answers"].replace(0, float("nan")).astype(float)).round(3)
        totals = totals.sort_values(["participant_id", "moment"], key=lambda c: c.astype(str)) # sort by participant and then by moment
        low = totals[(totals["moment"] == "all") & (totals["retention"] < 0.7)] # flags participants who keep less than 70% of their words
        for _, r in low.iterrows():
            flags.add(None, r["participant_id"], "totals", "INFO",
                      f"only {r['retention']:.0%} of participant words kept after min-words filter")
 
    # ---- sentence flags
    if not sents.empty: 
        for _, r in sents[sents["n_words"] > args.long_sentence].iterrows():
            # sign of missing punctuation in transcript
            flags.add(r["file"], r["participant_id"], "sentences", "INFO",
                      f"long sentence ({r['n_words']} words, moment {r['moment']}): '{r['sentence'][:60]}...'")
 
    # ---- fillers: rates per 100 words for each category
    if not fillers.empty:
        fall = fillers.groupby("participant_id", as_index=False).sum(numeric_only=True)
        fall["moment"] = "all" # sums over the three moments
        fillers = pd.concat([fillers, fall], ignore_index=True)
        for cat in list(FILLERS) + ["repetitions"]:
            fillers[f"{cat}_per100w"] = (100 * fillers[cat] / fillers["words"].replace(0, float("nan")).astype(float)).round(2)
 
    # ---- word frequencies on kept text
    content_rows, rare_counter, rare_example = [], Counter(), {}
    per_moment = {m: Counter() for m in (1, 2, 3)}
    doc_freq = Counter()
    pids = {pid for pid, _ in store["texts"]} # participants from (pid, moment) tuples with stored text
    for pid in sorted(pids): # fixed order 
        seen = set()
        for m in (1, 2, 3): # for each participant and moment
            text = store["texts"].get((pid, m), "") # kept text
            for w in words(text):
                if not is_content_word(w):
                    continue
                per_moment[m][w] += 1 # counts word for this specific moment
                # collects the participant's distinct words across all moments (not stopwords or fillers, words under 3 letters and contractions part of stopword)
                seen.add(w) 
                if is_rare and is_rare(w): # counting rare words
                    rare_counter[w] += 1
                    if w not in rare_example:  # regex search runs only for first example of each rare word
                        rare_example[w] = (pid, m, re.search(rf".{{0,40}}\b{re.escape(w)}\b.{{0,40}}", text, re.I))        
        doc_freq.update(seen)  # adds 1 for each distinct word this participant used
    overall = sum(per_moment.values(), Counter())  # adds three moment counters into one
    n_p = len(pids) or 1 # number of participants whose text is included in word frequencies
    # one row per content word, most frequent first, counts per moment and share of participants using it
    for w, c in overall.most_common():   
        content_rows.append({"word": w, "count_all": c, "count_m1": per_moment[1][w], "count_m2": per_moment[2][w],
                             "count_m3": per_moment[3][w], "n_participants": doc_freq[w],
                             "participant_share": round(doc_freq[w] / n_p, 2)})
    # saving into table the word counts
    content = pd.DataFrame(content_rows)
    rare = pd.DataFrame([{"word": w, "count": c, "example_participant": rare_example[w][0],
                          "example_moment": rare_example[w][1],
                          "context": rare_example[w][2].group(0) if rare_example[w][2] else ""}
                         for w, c in rare_counter.most_common()])
 
    # ---- write CSVs 
    # flags table ordered according to error, warning and information
    flags_df = pd.DataFrame(flags.rows, columns=["file", "participant_id", "check", "severity", "detail"])
    sev_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    flags_df = flags_df.sort_values(["severity", "file"], key=lambda c: c.map(sev_order) if c.name == "severity" else c.astype(str))
    # writes every table created into a qc_<name>.csv file
    for name, df in [("flags", flags_df), ("moments", moments), ("speakers", speakers), ("totals", totals),
                     ("answers", answers_df), ("sentences", sents), ("fillers", fillers),
                     ("words_content", content), ("words_rare", rare)]:
        df.to_csv(out / f"qc_{name}.csv", index=False)
 
    # ---- per-participant flag files
    # optionally also writes each participant's flags into their own folder, next to preprocessing results
    if args.per_participant_dir: # if old files from part run
        root = Path(args.per_participant_dir)
        # first remove the old flag files of the participants checked in THIS run, so a participant without flags now doesn't keep the flags of an earlier run (others' files are left alone)
        checked = pd.concat([moments.reindex(columns=["file", "participant_id"]),
                             flags_df.reindex(columns=["file", "participant_id"])], ignore_index=True)
        for key in {k for k in checked.apply(folder_for, axis=1, root=root) if k}:
            (root / str(key) / f"{key}_qc_flags.csv").unlink(missing_ok=True)
    if args.per_participant_dir and not flags_df.empty:
        # folder = existing participant folder named like the file (run_preproc.sh names the selected transcripts <ID>.txt), else the speaker ID found in the transcript
        keyed = flags_df.assign(_folder=flags_df.apply(folder_for, axis=1, root=root)).dropna(subset=["_folder"])
        # writes each folder's flags
        for key, grp in keyed.groupby("_folder"):
            pdir = root / str(key)
            pdir.mkdir(parents=True, exist_ok=True)
            grp.drop(columns="_folder").to_csv(pdir / f"{key}_qc_flags.csv", index=False)
 
    # ---- summary (list of lines)
    L = []
    L.append("=" * 72)
    L.append(f"QC SUMMARY  -  {len(files)} file(s) in {args.input_dir}")
    L.append("=" * 72)
    L.append(f"Sentence splitter: {SENT_SPLITTER} | rare-word check: {RARE_METHOD}")
    L.append(f"Filters: answers >= {args.min_words} words, sentences >= {args.min_sentence_words} words")
 
    L.append("\n--- FLAGS ---") # a count per severity and check
    if flags_df.empty:
        L.append("No issues flagged.")
    else:
        L.append(flags_df.groupby(["severity", "check"]).size().rename("n").reset_index().to_string(index=False))
        L.append("(details in qc_flags.csv - fix ERRORs before modelling)")
 
    L.append("\n--- MOMENTS ---") # counts each status
    if not moments.empty:
        L.append(moments["status"].value_counts().to_string())
        if "interviewer" in moments and moments["interviewer"].nunique() > 1: # more than 1 interviewer
            # markers were written from one interviewer's phrasing; others may word things differently
            L.append("\nstatus by interviewer (unexpected structures concentrated in one interviewer "
                     "suggest phrasing the moment rules do not cover):")
            # makes table of interviewer by status (complete, 2 moments, 1 moment interviewer)
            L.append(pd.crosstab(moments["interviewer"].fillna("?"), moments["status"]).to_string())
        for m in (1, 2, 3):
            col = f"m{m}_duration_s"
            if col in moments:
                d = moments.loc[moments[col] > 0, col]
                if len(d):
                    L.append(f"moment {m} duration (s): median {d.median():.0f}, range {d.min()}-{d.max()}")
 
    L.append("\n--- SPEAKERS ---") # interviewers with the number of file each + participant word share
    if not speakers.empty:
        other = speakers[speakers["role"] == "other"]["speaker"].value_counts()
        L.append("interviewers: " + ", ".join(f"{k} ({v} files)" for k, v in other.items()))
        ps = speakers[speakers["role"] == "participant"]["word_share"]
        L.append(f"participant word share: {describe(ps)}")
 
    L.append("\n--- TOTALS (kept text) ---") # per moment, summed over particpants
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
    # answer-length overall and per moment
    if not answers_df.empty:
        L.append(f"all:      {describe(answers_df['n_words'])}  (words)")
        for m in (1, 2, 3):
            L.append(f"moment {m}: {describe(answers_df.loc[answers_df['moment'] == m, 'n_words'])}")
        # number of answers above token limit
        long_ = answers_df[answers_df["approx_tokens"] > args.long_answer_tokens]
        L.append(f"answers over ~{args.long_answer_tokens} tokens (check the embedding model's max_seq_length): "
                 f"{len(long_)}; longest ~{answers_df['approx_tokens'].max()} tokens")
        # answers per participant
        L.append(f"answers per participant: {describe(answers_df.groupby('participant_id').size())}")
 
    L.append("\n--- SENTENCES ---") # sentence-length distribution and three counts 
    if not sents.empty:
        L.append(f"all:      {describe(sents['n_words'])}")
        for m in (1, 2, 3):
            L.append(f"moment {m}: {describe(sents.loc[sents['moment'] == m, 'n_words'])}")
        L.append(f"< {args.min_sentence_words} words (dropped by MOSAIC): {(~sents['kept_by_mosaic']).sum()} | "
                 f"< 5 words: {(sents['n_words'] < 5).sum()} | > {args.long_sentence} words: "
                 f"{(sents['n_words'] > args.long_sentence).sum()}")
        ends = list(".!?…\"”’')")  # a sentence may end with an ellipsis or a closing quote/bracket
        L.append(f"sentences without final punctuation: {(~sents['sentence'].str.strip().str[-1].isin(ends)).sum()}")
 
    L.append("\n--- FILLERS (per 100 words, all moments) ---") # for each category
    if not fillers.empty:
        fa = fillers[fillers["moment"] == "all"]
        tot_w = fa["words"].sum() or 1 # total
        for cat in list(FILLERS) + ["repetitions"]: # rate over all words and highest rate of a single participant
            L.append(f"{cat:18s}: total {fa[cat].sum():5d} | {100 * fa[cat].sum() / tot_w:.2f} per 100 words | "
                     f"per participant max {fa[f'{cat}_per100w'].max()}")
 
    L.append(f"\n--- TOP {args.top_n} CONTENT WORDS (no stopwords/fillers) ---") # top words with counts
    if not content.empty:
        L.append(", ".join(f"{r.word} ({r.count_all})" for r in content.head(args.top_n).itertuples()))
        common = content[(content["participant_share"] >= 0.5) & (content.index < 200)]
        if len(common):
            L.append("\nUsed by >= 50% of participants (candidate domain stopwords - review, don't auto-remove):")
            L.append(", ".join(f"{r.word} ({r.participant_share:.0%})" for r in common.head(40).itertuples()))
 
    L.append("\n--- RARE / UNKNOWN WORDS (possible transcription errors, names, non-English) ---")
    if is_rare is None:
        L.append(RARE_METHOD)
    elif not rare.empty: # 40 most frequent rare words
        L.append(", ".join(f"{r.word} ({r.count})" for r in rare.head(40).itertuples()))
    else:
        L.append("none")
 
    L.append(f"\nAll tables written to {out.resolve()}")
    summary = "\n".join(L)
    (out / "qc_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
 
 
if __name__ == "__main__":
    main()