#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : inspect_moments.py TODO CORRECT AND CHECK EVERYTHING
# description     : Shows how preprocessing.py's moment rules read ONE transcript, to find out why a moment was not found
#                   - where each moment starts (as preprocessing.py decides)
#                   - every interviewer turn that could open a moment (mentions a scene/moment, a transition word or a scene keyword),
#                     with: its moment label, whether it counts as a transition, whether it is a deferral, and which scene keywords it contains
#                   - with --context, the turns around each candidate (both speakers)
#                   - with --scene, how moment 1 (most intense) is classified, the resulting decision for a skipped moment 2/3 (copy or not),
#                     and frequent moment-1 words of the participant that match NO scene keyword (candidates for missing keywords in SCENES)
#                   - only reads the transcript; uses the rules of preprocessing.py in the same folder
# usage           : python inspect_moments.py /path/to/sub-gutsleiXXXX_ses-01a_task-emtint.txt [--context 3] [--scene]
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-10-05
# version         : 1.0
# ==============================================================================

import argparse
import re
from collections import Counter
import sys
import warnings
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing as prep  # noqa: E402
 
# interviewer turns worth showing: anything that could be (part of) a moment prompt
CANDIDATE_RE = re.compile(r"\b(?:scene|scenes|moment|moments|part)\b", re.I)
# common words left out of the "unmatched words" list of --scene
try:
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except Exception:
    STOP = set("""the and that this with was were have had they them their there then than what when where which while into about
    from just like really very would could should been being also because some something thing things know think mean yeah okay""".split())
 
 
def scene_report(seg, is_participant):
    """How moment 1 is classified (as preprocessing.py does it), the decision for a skipped moment, and unmatched frequent words."""
    m1 = [t["text"] for t in seg if t["moment"] == 1 and not prep.is_procedural(t, is_participant)]
    print("\n  scene of moment 1 (strong keyword = 2 points, weak = 1):")
    for label, text in ((f"first {prep.SCENE_WINDOW_TURNS} content turns (used first)", " ".join(m1[:prep.SCENE_WINDOW_TURNS])),
                        ("all of moment 1 (used if the first turns name no scene)", " ".join(m1))):
        scene, conf, ev, present = prep.classify_scene(text)
        print(f"    {label}:\n      -> {scene} ({conf}); scenes present: {'+'.join(present) or 'none'}\n      evidence: {ev or '-'}")
    answers = prep.build_answers(seg, is_participant, 3)
    dec = prep.decide_moment_sources(seg, answers, is_participant)
    print(f"\n  decision (as preprocessing.py): scene {dec['scene']} ({dec['scene_conf']}) | "
          f"moment 2: {dec['source'][2]} | moment 3: {dec['source'][3]}")
    for sev, msg in dec["notes"]:
        print(f"    {sev}: {msg}")
    # the participant's own moment-1 words that no scene keyword matches
    text = " ".join(t["text"] for t in seg if t["moment"] == 1 and is_participant(t["speaker"]))
    for res in prep.SCENE_RES.values():
        for rx in res.values():
            text = rx.sub(" ", text)
    words = Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if len(w) > 3 and w not in STOP)
    print("\n  participant's moment-1 words matching NO scene keyword (most frequent first) - candidates for SCENES:")
    print("    " + ", ".join(f"{w} ({c})" for w, c in words.most_common(30)))
 
 
def main():
    ap = argparse.ArgumentParser(description="Show how the moment rules of preprocessing.py read one transcript.")
    ap.add_argument("transcript", help="path to one transcript (.txt)")
    ap.add_argument("--context", type=int, default=0, help="also show this many turns before and after each candidate")
    ap.add_argument("--scene", action="store_true", help="also show the scene classification of moment 1 and unmatched words")
    ap.add_argument("--participant-regex", default=prep.DEFAULT_PARTICIPANT_RE)
    args = ap.parse_args()
 
    f = Path(args.transcript)
    part_re = re.compile(args.participant_regex, re.I)
    is_participant = lambda s: bool(part_re.fullmatch(s))
    turns = prep.parse_transcript(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg, info = prep.segment(turns, is_participant, f.name)
    if seg is None:
        print(f"{f.name}: 'most intense' not found in any interviewer turn -> excluded")
        return
 
    print(f"{f.name}")
    print(f"  moment 1 starts {info['moment1_start']} | moment 2 starts {info['moment2_start'] or 'NOT FOUND'} | "
          f"moment 3 starts {info['moment3_start'] or 'NOT FOUND'}")
    unclassified = set(prep.unclassified_transitions(seg, is_participant))
    print(f"  unrecognised transitions: {[seg[i]['start'] for i in sorted(unclassified)] or 'none'}\n")
 
    print("  candidate interviewer turns (could open a moment):")
    print(f"  {'time':8} {'label':6} {'transition':10} {'deferral':8} {'m2 word':7} {'m3 word':7}  text")
    shown = set()
    for i, t in enumerate(seg):
        if is_participant(t["speaker"]):
            continue
        txt = t["text"]
        if not (CANDIDATE_RE.search(txt) or prep.MOMENT_TRANSITION_RE.search(txt)
                or prep.MOMENT2_RE.search(txt) or prep.MOMENT3_RE.search(txt)):
            continue
        if args.context:  # turns before the candidate (both speakers)
            for k in range(max(0, i - args.context), i):
                if k not in shown:
                    print(f"      {seg[k]['start']} {seg[k]['speaker'][:12]:12} {seg[k]['text'][:110]}")
                    shown.add(k)
        mark = "  <- UNRECOGNISED" if i in unclassified else ""
        print(f"  {t['start']:8} {str(t['moment']):6} {str(prep.is_transition(txt)):10} "
              f"{str(bool(prep.DEFERRAL_RE.search(txt))):8} {str(bool(prep.MOMENT2_RE.search(txt))):7} "
              f"{str(bool(prep.MOMENT3_RE.search(txt))):7}  {txt[:110]}{mark}")
        shown.add(i)
        if args.context:  # turns after the candidate
            for k in range(i + 1, min(len(seg), i + 1 + args.context)):
                if k not in shown:
                    print(f"      {seg[k]['start']} {seg[k]['speaker'][:12]:12} {seg[k]['text'][:110]}")
                    shown.add(k)
            print()
    if args.scene:
        scene_report(seg, is_participant)
 
 
if __name__ == "__main__":
    main()