#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module for phenomenological interview data (focusing on three film moments)
#                   - turns timestamped interview transcripts (.txt files) into preprocessed data for analysis (.csv files - for topic modelling with 
#                   MOSAIC ((https://github.com/romybeaute/MOSAIC) 
#                   - step 1 - Parsing: parse timestamped speaker turns
#                   - step 2 - Removal of Introduction: removal of all turns up to the start of experiential description (most intense moment) - discarding 
#                   readiness replies (short turns) 
#                   - step 3 - Segmentation into moments: moment 1 (most intense moment, after the introduction, can be any scene), moment 2 (celebration) 
#                   and moment 3 (woman, final scene of the movie) - taking into account backchannel words + most intense moment correspoding to moment 2 or 3
#                   - step 4 - Answers selection: select only answers from the participant (audit file with corresponding questions is generated); merging 
#                   some turns into single answer (if separated by interviewer backchannel words like "um", "yeah" or if .../- snd signals for continuation, when
#                   short continuation interruption from the interviewer - max 3 words)
#                   - step 5 - Text cleaning (for participant's text only): remove non-verbal annotations, hesitations, keep answers with min. 3 words (default)
#                   - no language model is used 
# output          : preprocessed data for analysis - .csv files (one dataset for full interview plus one per moment - extra .txt files for inspection), audit .csv file and segmentation report .csv file 
# dependencies    : Python  : >= 3.10 ; pandas installation
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-09-30
# version         : 1.0
# ==============================================================================

import argparse          
import re                
import sys               
import warnings    
from collections import Counter
from pathlib import Path          
import pandas as pd               

# ----------------------------------------------------------------------------
# SECTION 0: Configuration and patterns
# ----------------------------------------------------------------------------
# Basic features of the txt file
TIMESTAMP_RE = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\s*$") # timestamps in the format HH:MM:SS-HH:MM:SS
INTERVIEWER_RE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(.*)$") #any speaker who is not a participant is treated as an interviewer
DEFAULT_PARTICIPANT_RE = r"^sub-gutslei\d+$" 

# Set actual start of the interview: the interviewer's first mention of "most intense" moment. Everything before it is removed.
INTRO_START_RE = re.compile(r"\bmost\s+intense\b", re.I)

# Signals of readiness after evocation to remove - any combination of words (max. 8 words)
READY_WORDS = set("""yes yeah yep ok okay alright all right sure fine good perfect great ready i im m am me let lets s start started go we can now so and 
well here there it its is my eyes closed close see think set done too as no nope not yet clear question questions any understood understand""".split())

# Participant questions to the interviewer - not to be analyzed
PARTICIPANT_QUESTION_MAX_WORDS = 8

# Signal of moment focus change - they open a moment only together with moment's keyword
MOMENT_TRANSITION_RE = re.compile(r"\b(?:another|other|next|different|new|final|last|second|third)\s+(?:moment|scene)\b", re.I)
MOMENT2_RE = re.compile(r"celebrat", re.I)  # celebration / celebrating
MOMENT3_RE = re.compile(r"\b(?:wom[ae]n|mother|mom)\b", re.I)  # the woman (mother) comes in
# Deferrals: when interviewer postpones moment participant brings up, to still focus on the moment they are in
DEFERRAL_RE = re.compile( r"\b(?:that|this|it)(?:'s|\s+is|\s+was)\s+(?:\w+\s+){0,2}(?:moment|scene)\b"
    r"|\bwe(?:'ll|\s+will)\s+(?:\w+\s+){0,4}(?:after(?:wards)?|later)\b", re.I)

# Signals the presence of non-verbal annotations in the text. Removed (with an adjacent comma) anywhere in an answer unless --keep-annotations.
ANNOTATION_RE = re.compile(r"[()\[\]][^()\[\]]*[()\[\]]") 
 
# Pure hesitations: non-lexical sounds with no meaning of their own. Removed (with an adjacent comma) anywhere in an answer unless --keep-hesitations.
# Lexical discourse markers ('yeah', 'okay', 'like', 'you know') are NOT removed from the text: they can carry meaning 
HESITATIONS = [r"m+-?h+m+", r"u+h-?h+u+h+", r"u+h+m*", r"u+m+", r"erm+", r"h+m+", r"m+h*m+", r"a+h+", r"e+h+"]  # longest first
HESITATION_RE = re.compile(r"(?:,\s*)?\b(?:" + "|".join(HESITATIONS) + r")\b(?:\s*,)?", re.I)

# Interviewer backchannel words (max 3 words) that do not break a participant's answer. If the interviewer interrupts with more than 3 words, the answer is split.
BACKCHANNEL_WORDS = {"mhm", "okay", "ok", "yeah", "yes", "right", "sure", "alright", "ah", "oh", "i", "see", "uh", "huh", "mmm"}

# Continuation marks: '...' / '…' or a dash at the end of a turn and at the start of the participant's next turn. Connect the two turns into a single answer, if interviewer interruption is shorter than 4 words.
CONT_END_RE = re.compile(r"(?:\.\.\.|…|[-–—])\s*$")
CONT_START_RE = re.compile(r"^\s*(?:\.\.\.|…|[-–—])")
CONTINUATION_MAX_WORDS = 3  # longest interviewer turn bridged when both marks are present
 
# ----------------------------------------------------------------------------
# SECTION 1: Parsing the transcript
# ----------------------------------------------------------------------------

# Function to turn .txt file into a list of turns: turn t {start, end, speaker, text, turn_idx}
def parse_transcript(path: Path) -> list[dict]:
    """Return a list of turns: {start, end, speaker, text}."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    turns, current, pending_ts = [], None, None # turns = list of turns, current = current turn being built, pending_ts = timestamp waiting for speaker label (next line)

    # Parse the transcript line by line depending on whether it is a timestamp, speaker label, or continuation line. Blank lines are skipped.
    for line in lines:
        if not line.strip(): 
            continue
        ts = TIMESTAMP_RE.match(line)
        if ts: #new timestamp line
            if current: 
                turns.append(current) # save previous turn 
                current = None
            pending_ts = ts.groups() # save new timestamp
            continue
        if pending_ts is not None: # new turn with a timestamp, check if it has a speaker label
            m = SPEAKER_RE.match(line)
            if m:
                current = {"start": pending_ts[0], "end": pending_ts[1],
                           "speaker": m.group(1).strip(), "text": m.group(2).strip()} # to split speaker and text
            else:  # timestamp without speaker label
                current = {"start": pending_ts[0], "end": pending_ts[1],
                           "speaker": "UNKNOWN", "text": line.strip()}
            pending_ts = None
        elif current:  # continuation line - same turn, append to text
            current["text"] += " " + line.strip()

    if current: #close the last turn if any as there is no timestamp after it
        turns.append(current) 
    for i, t in enumerate(turns): #number each turn
        t["turn_idx"] = i
    return turns

# Function to identify the interviewer (first non-participant speaker label) in the transcript - no typos possible due to transcription procedure
def set_interviewer(turns, is_participant):
    """First non-participant speaker label (the interviewer)."""
    return next(
        (t["speaker"] for t in turns
         if not is_participant(t["speaker"]) and t["speaker"] != "UNKNOWN"),
        None,
    )

# Function to clean the text of each turn
def clean_text(text: str, remove_hesitations: bool = False) -> str:
    """Remove annotations (and optionally hesitations); tidy spaces, punctuation
    and dangling dashes. Words, numbers, case and punctuation are otherwise kept."""
    text = ANNOTATION_RE.sub(" ", text) # replace annotations with a space
    if remove_hesitations:
        text = HESITATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text) # collapse spaces
    text = re.sub(r"([.!?…])\s+[.!?…]", r"\1", text) # remove double punctuation
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text) # remove space before punctuation
    text = re.sub(r",\s*(?=[,.!?;:…])", "", text)  # remove leftover commas if followed by punctuation mark
    text = re.sub(r"([.!?…])\s*,", r"\1", text) # remove commas after punctuation marks
    text = re.sub(r"^[\s,.;:!?…\-–—]+", "", text) # clean beginning of string (including dashes)
    text = re.sub(r"(?:\s+[-–—]+|[–—]+)$", "", text)  # clean end of string (including dashes)
    return text.strip()

# Function to count the number of words in a text
def n_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

# Function to check if short turn (1-8 words) is signaling readiness
def is_ready_reply(text: str) -> bool:
    """Short participant turn that only signals readiness (e.g.'I am ready', 'I am there', "Let's start"), 'Not yet', 'No, it's clear'."""
    words = re.findall(r"[a-z]+", clean_text(text, True).lower())
    return 0 < len(words) <= 8 and all(w in READY_WORDS for w in words) # after removal of hesitations and annotations, still needs 1 word minimum

# Function to check if participantturn corresponds to a question
def is_participant_question(text: str) -> bool:
    """Short participant turn asking the interviewer something (e.g.'Can you repeat the question?', 'Which moment do you mean?')."""
    c = clean_text(text, True)
    return c.endswith("?") and 0 < n_words(c) <= PARTICIPANT_QUESTION_MAX_WORDS

# Function to check if the  turn is a backchannel (acknowledgment) turn
def is_backchannel(text: str) -> bool:
    """Turn, for either speaker, that only acknowledges (e.g. 'Mmm', 'Mhm', 'Okay', 'Yeah, right')."""
    words = re.findall(r"[a-z]+", clean_text(text, True).lower())
    return len(words) <= 3 and all(w in BACKCHANNEL_WORDS for w in words) # number of words can be zero after cleaning - still a bridge between parts of an answer

# Function to join two parts of one answer, removing continuation marks at the junction - done later if interruption from the interviewer is shorter than 4 words
def join_continuation(first: str, second: str) -> str:
    """Join two parts of one answer, removing continuation marks at the junction."""
    return CONT_END_RE.sub("", first).rstrip() + " " + CONT_START_RE.sub("", second).lstrip()

# --------------------------------------------------------------------------------------------
# SECTION 2: Segmentation - work on individual turns to form blocks of answers for each moment
# --------------------------------------------------------------------------------------------

# Function to check if the turn is a transition to another moment/scene (not a deferral)
def is_transition(text: str) -> bool:
    """Interviewer text that moves to another part of the film (and is not a deferral)."""
    return bool(MOMENT_TRANSITION_RE.search(text)) and not DEFERRAL_RE.search(text) 

# Function to find when moment/scene prompt ends (consecutive turns from the interviewer or participant backchannels)
def prompt_end(turns, i, is_participant):
    """Return the index of the last turn of the prompt starting at turn i. Interviewer turns and participant backchannels ('Yeah', 'Mhm') are
    included; stops before the participant's first real answer."""
    j = i
    while j + 1 < len(turns) and (not is_participant(turns[j + 1]["speaker"]) # looks at next turn to check
                                  or is_backchannel(turns[j + 1]["text"])):
        j += 1
    return j

# Function to find the first turn saying 'another moment' + scene keyword while making sure it is not a later reference to the scene (e.g. 'another moment' + 'celebration' but not 'mother' as a reference to the celebration scene)
def find_moment_start(turns, is_participant, topic_re, after_idx, exclude_re=None):
    """First interviewer turn (after 'after_idx') saying 'another/next moment/scene' + scene keyword.
    The keyword may also sit in the next interviewer turns, across participan backchannels.
    If `exclude_re` is given, the prompt must not match that pattern."""
    for i, t in enumerate(turns): # i position, t turn
        if i <= after_idx or is_participant(t["speaker"]) or not is_transition(t["text"]):
            continue
        prompt = " ".join(x["text"] for x in turns[i:prompt_end(turns, i, is_participant) + 1])
        if topic_re.search(prompt) and not (exclude_re and exclude_re.search(prompt)): # makes sure that turn is not mentioning the pattern outside the scene 
            return i
    return None

# Function to find interviewer turns (moment prompts) that were missed and are not deferrals - quality check
def unclassified_transitions(turns, is_participant):
    """Interviewer turns after the introduction saying 'another moment/scene' that are not part of a recognised moment-2 or moment-3 prompt (likely a
    moment introduced with wording the rules do not cover)."""
    covered = set() # set of turn indices that are part of a recognised moment-2 or moment-3 prompt
    for m in (2, 3):
        s = next((i for i, t in enumerate(turns) if t.get("moment") == m), None) # find the first turn of moment m checking all turns labelled with moment m
        if s is not None:
            covered.update(range(s, prompt_end(turns, s, is_participant) + 1)) # add indices of all turns in the prompt to the covered set
    return [i for i, t in enumerate(turns) # returns list of turn indices that are not covered by any recognised moment prompt (of interviewer), that is no a deferral
            if t.get("moment") in (1, 2, 3) and not is_participant(t["speaker"])
            and is_transition(t["text"]) and i not in covered]

# Function to segment/label transcript and assign each participant's turns in the transcript into moments that exist and collect summary for quality checks
def segment(turns, is_participant, fname):
    """Assign 'intro' / 1 / 2 / 3 to each participant turn (and flag leading readiness replies); return turns + marker info."""
    info = {"file": fname, "intro_found": False, "moment1_start": None, "n_evocation_replies": 0,
            "moment2_start": None, "moment3_start": None} # dictionary to store information about the segmentation process for quality checks

    # first interviewer turn (matching index) saying "most intense": moment 1 starts here
    m1 = next((i for i, t in enumerate(turns) if not is_participant(t["speaker"])
               and INTRO_START_RE.search(t["text"])), None)
    if m1 is None:
        warnings.warn(f"[{fname}] 'most intense' not found in an interviewer turn -> file skipped.")
        return None, info
    intro_end = m1 - 1  # last removed turn corresponding to the intro ([-1] if nothing precedes it)
    info["intro_found"] = True
    info["moment1_start"] = turns[m1]["start"]  # stores start timestamp
 
    # moment 2 ("celebration") and moment 3 ("woman/mother") - if there is any
    m2 = find_moment_start(turns, is_participant, MOMENT2_RE, intro_end, exclude_re=MOMENT3_RE) # a prompt naming the woman/mother is moment 3, even if it mentions the celebration
    m3 = find_moment_start(turns, is_participant, MOMENT3_RE, m2 if m2 is not None else intro_end)
 
    if m2 is None:
        warnings.warn(f"[{fname}] moment 2 marker not found.")
    else:
        info["moment2_start"] = turns[m2]["start"]
    if m3 is None:
        warnings.warn(f"[{fname}] moment 3 marker not found.")
    else:
        info["moment3_start"] = turns[m3]["start"]

    # label each turn with its moment and flagging evocation readiness replies (short turns) - only the first real answer counts as the start of moment 1
    current, leading = None, False  # still before the participant's first real answer
    for i, t in enumerate(turns):
        t["evocation"] = False # start with no evocation readiness replies
        if i <= intro_end: 
            t["moment"] = "intro"
        elif m3 is not None and i >= m3: # check from last moment backwards
            t["moment"] = 3
        elif m2 is not None and i >= m2:
            t["moment"] = 2
        else:
            t["moment"] = 1
        if t["moment"] != current: # if it has been labeled, i.e., a new moment starts here
            current, leading = t["moment"], t["moment"] != "intro" # leading = True 
        if leading and is_participant(t["speaker"]):
            if is_ready_reply(t["text"]):
                t["evocation"] = True # flag evocation readiness reply
            elif not (is_backchannel(t["text"]) or is_participant_question(t["text"])): # if not readiness, nor backchannel, nor question
                leading = False  # first real answer - beginning of moment is identified
    info["n_evocation_replies"] = sum(t["evocation"] for t in turns) # counts flagged turns for quality check
    return turns, info


# ----------------------------------------------------------------------------
# SECTION 3 - Scene classification and moment source decision - TO REVIEW
# ----------------------------------------------------------------------------

## Moment 1 = the participant's MOST INTENSE scene (any scene of the film). If it was the celebration, the interview skips 
# moment 2 -> moment 2 = moment 1. If it was the mother scene, the interview skips moment 3 -> moment 3 = moment 1.
# Keywords identify which scene moment 1 is about. (strong = 2 points, weak = 1). Keywords shared by several scenes (the boy, 
# crying, the table) are weak; scene-specific events are strong. Patterns use only non-capturing groups (?:...).
SCENES = {
    # the boxer's victory and the crowd celebrating
    "celebration":     {"strong": [r"celebrat\w*", r"won", r"win", r"wins", r"winning", r"cheer\w*"],
                        "weak":   [r"crowd"]},
    # the woman (the boy's mother) comes in
    "mother":          {"strong": [r"wom[ae]n", r"mother", r"mom", r"mum", r"hug\w*"],
                        "weak":   [r"comes? in", r"came in", r"coming in", r"together"]},
    # father and son at the beach, having fun
    "beach":           {"strong": [r"beach", r"sea", r"sand", r"waves?", r"swim\w*", r"play\w*"],
                        "weak":   [r"fun", r"laugh\w*"]},
    # the boxer sitting in the ring or lying on his side, before the fight
    "sitting_in_ring": {"strong": [r"(?:sitting|sat|sits?) (?:alone )?(?:in|on) the ring",
                                   r"(?:lying|laying|lay|lies) on (?:the|his) side",
                                   r"before (?:the |he (?:was |is |starts? |started |began ))?"
                                   r"(?:fight\w*|boxing(?: match)?|box(?: fight\w*)?|match|bout)"],
                        "weak":   [r"sitting", r"corner"]},
    # the boxing match ('before the fight' belongs to sitting_in_ring)
    "fighting":        {"strong": [r"fight\w*", r"boxing", r"box", r"punch\w*",
                                   r"opponent", r"knock\w*"],
                        "weak":   [r"hit\w*", r"round", r"ring", r"losing", r"loses"]},
    # before he dies: the father in bad shape on the table, talking with the boy
    "death":           {"strong": [r"death", r"dying", r"(?:about|going) to die", r"bad shape", r"last words",
                                   r"talk\w* (?:to|with) (?:his |the )?(?:son|boy|kid|father|dad)"],
                        "weak":   [r"(?:lying |laying |lies )?on the table"]},
    "doctor":          {"strong": [r"doctor"], "weak": []},
    # after the death: the father lies dead and the boy cries hard, asks to wake him up, goes around the room to the adults, is told his father is dead
    "boy":             {"strong": [r"(?:wake|wakes|waking)[\s-]+(?:\w+\s+){0,3}up", r"(?:won'?t|will not|doesn'?t|does not|didn'?t|not) wake",
                                   r"(?:is|was) dead", r"died",
                                   r"realiz\w*(?: \w+){0,5} (?:dead|died)",
                                   r"(?:told|tells?|got word|finds? out|found out|news)(?: \w+){0,5} (?:dead|died)",
                                   r"around the room", r"cr(?:y|ies|ying|ied) (?:so |really )?hard"],
                        "weak":   [r"boy", r"kid", r"child", r"son", r"cry\w*", r"cried", r"tears", r"dead",
                                   r"lost (?:his |the )?(?:father|dad)"]},
}
# Builds all the patterns for scene detection 
SCENE_RES = {sc: {w: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for w, p in d.items() if p}
             for sc, d in SCENES.items()} # take SCENES and turns every list of keywords into a compiled pattern
# Extra rules
SKIPPED_SCENE = {2: "celebration", 3: "mother"}  # to save the scene that is redundant, as it was most intense moment
SCENE_WINDOW_TURNS = 6  # first N content (not readiness, nor backchannels) turns of moment 1 (both speakers) name the scene as it is identified early on by the participant (ideally)
# A most intense moment can span several scenes: every scene scoring at least 2 points and at least this share of the top score is listed in m1_scenes.
SCENE_SHARE_FOR_MULTI = 0.5

# Function to classify the scene of moment 1 based on keyword scores
def classify_scene(text):
    """Score every scene by its keywords. Returns (best scene, confidence, evidence, scenes present). Each stretch of text counts once, for the longest 
    keyword that matches it (strong 2 points, weak 1 point). A scene needs at least one strong (scene-specific) keyword to be 'clear' or present; weak 
    (shared) words only add support. 'clear' = top scene has a strong keyword and scores at least twice the runner-up; scenes present = scenes with a 
    strong keyword and at least SCENE_SHARE_FOR_MULTI of the top score."""
    # dealing with overlapping matches - all keyword matches of all scenes saving (start, end, scene, is_strong)
    matches = [(m.start(), m.end(), sc, w == "strong")
               for sc, res in SCENE_RES.items() for w, rx in res.items() for m in rx.finditer(text)]
    matches.sort(key=lambda x: (-(x[1] - x[0]), not x[3], x[0])) # keep the longest match wherever matches overlap (strong first on equal length)
    kept = []
    for st, en, sc, is_strong in matches: 
        if all(en <= k[0] or st >= k[1] for k in kept): # checks if match doesn't overlap with match saved
            kept.append((st, en, sc, is_strong))
    scores, hits, strong = {}, {}, {} # hits - dictionary mapping list of words found to each scene
    for st, en, sc, is_strong in kept:
        scores[sc] = scores.get(sc, 0) + (2 if is_strong else 1) # score starts at 0 the first time scene appears
        strong[sc] = strong.get(sc, 0) + is_strong # count strong hits
        hits.setdefault(sc, []).append(text[st:en].lower()) # stores list of keywords for each scene
    if not scores:
        return "unknown", "none", "", []
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[0] # scene with the highest score - main scene of moment 1
    second = scores[ranked[1]] if len(ranked) > 1 else 0
    conf = "clear" if strong.get(top, 0) and scores[top] >= 2 * second else "ambiguous" # enough confidence if at least one strong keyword and score is at least double the second best score
    present = [sc for sc in ranked if strong.get(sc, 0) and scores[sc] >= SCENE_SHARE_FOR_MULTI * scores[top]] # keeps every scene with strong hit and score at least 0.5 the top score
    evidence = "; ".join(f"{sc}={scores[sc]} ({', '.join(sorted(set(hits[sc])))})" for sc in ranked[:3]) # summary of top 3 scenes and their keywords
    return top, conf, evidence, present

# Function to mark as turn as having no content about the film (readiness, backchannel, participant question)
def is_procedural(t, is_participant):
    """Turn without content about the film: a readiness reply, a backchannel (either speaker) or a participant's question to the interviewer."""
    return bool(t.get("evocation") or is_backchannel(t["text"])
                or (is_participant(t["speaker"]) and is_participant_question(t["text"])))

# Function to determine scene of moment 1 (most intense scene) based on first turns (or all moment's turns if needed)
def moment1_scene(turn, is_participant):
    """Classify the first SCENE_WINDOW_TURNS turns of moment 1, or all of moment 1 if those contain no scene keyword (confidence 'weak').
    Procedural turns (no content) are skipped"""
    m1 = [t["text"] for t in turns if t.get("moment") == 1 and not is_procedural(t, is_participant)]  # collects texts of all content turns related to moment 1
    scene, conf, ev, present = classify_scene(" ".join(m1[:SCENE_WINDOW_TURNS])) # join first 6 turns in 1 string for classification
    if scene == "unknown":
        scene, conf, ev, present = classify_scene(" ".join(m1)) # uses all turns related to moment 1 for classifcation
        if scene != "unknown":
            conf = "weak"
    return scene, conf, ev, present

# Function to organize moments, specially if moment 2 and/or 3 are missing, being copied from moment 1 then, plus check if moment 1 is ambiguous - for quality check
def decide_moment_sources(turns, answers, is_participant, no_fill=False):
    """Decide, per moment, whether text is the participant's own, copied from moment 1 (2-moment interviews), missing or ambiguous. 
    Used by both prepare_gutslei.py and qc_gutslei.py so they always agree.
    Returns dict(has, scene, scene_conf, scene_evidence, source{1,2,3}, notes[(sev, msg)])."""
    has = {m: any(a["moment"] == m and a["kept"] for a in answers) for m in (1, 2, 3)} # dictionary that says if a moment has usable answers (answer a)
    scene, conf, ev, present = moment1_scene(turns, is_participant) # classify most intense moment according to scene
    source = {m: ("own" if has[m] else "missing") for m in (1, 2, 3)} # moments with text are "own" and the rest is "missing"
    notes = []
    multi_noted = False # True when a note has said moment 1 covers multiple scene

    # list of transition turns that were not recognised as moment 2 or 3
    unclassified = unclassified_transitions(turns, is_participant)

    if not has[1]: # moment 1 missing - always error!
        notes.append(("ERROR", "no moment-1 text"))
    missing = [m for m in (2, 3) if not has[m]]
    if has[1] and len(missing) == 1: # either moment 2 or 3 missing - 2 moment interview
        m = missing[0]
        expected = SKIPPED_SCENE[m]
        if unclassified: # most likely there is another moment but not recognised
            source[m] = "ambiguous (unrecognised transition)"
            notes.append(("ERROR", f"moment {m} missing but an unrecognised transition phrase appears at "
                                   f"{turns[unclassified[0]]['start']} - probably moment {m} with new wording; "
                                   f"add it to MOMENT{m}_RE. Not copied."))
        elif conf == "clear" and expected not in present: # if moment 1 is different scene from the 2 selected scenes - skipping moment not justified
            source[m] = f"missing (moment 1 is '{scene}', not '{expected}')"
            notes.append(("WARN", f"moment {m} missing, but moment 1 looks like the '{scene}' scene, not "
                                  f"'{expected}' - interview cut short? Not copied. Evidence: {ev}"))
        elif no_fill: # to not copy most intense moment to moment 2 or 3 (copy done for later across participants comparison)
            source[m] = "missing"
        else: # copy most intense moment (1) text to moment 2 or 3 depending which is corresponds to
            source[m] = "copied_from_moment1"
            if scene == expected and conf == "clear": # moment 1 is clearly the expect scene - celebration or woman
                notes.append(("INFO", f"2-moment interview: moment {m} copied from moment 1 "
                                      f"('{expected}' scene confirmed)"))
            elif expected in present and len(present) > 1: # moment 2 or 3 is present but not top score scene - moment 1 might cover several scenes - need later segmentation for analysis TODO
                notes.append(("INFO", f"2-moment interview: moment {m} copied from moment 1, which covers "
                                      f"several scenes including '{expected}' ({'+'.join(present)})"))
                multi_noted = True
            elif expected in present:
                notes.append(("WARN", f"2-moment interview: moment {m} copied from moment 1; the '{expected}' "
                                      f"scene is detected but not confirmed (confidence: {conf}). Please verify. "
                                      f"Evidence: {ev}"))
            else: # scene of moment 1 is unknown or with only weak keywords - copy made but flagged
                notes.append(("WARN", f"2-moment interview: moment {m} copied from moment 1, but the scene is not "
                                      f"confirmed as '{expected}' (detected: {scene}, {conf}). Please verify. "
                                      f"Evidence: {ev or 'no scene keywords'}"))
    elif len(missing) == 2: # only 1 moment not possible - always error!
        notes.append(("ERROR", "moments 2 and 3 both missing"))
    elif unclassified: # all moments found but still unclassified transition - warning to check
        notes.append(("WARN", f"unrecognised 'another moment' at {turns[unclassified[0]]['start']} "
                              f"in a 3-moment interview - check boundaries"))

    twice = [sc for sc in present if sc in SKIPPED_SCENE.values()] if all(has.values()) else [] # if all three moments, lists scenes in moment 1 that are also part of moment 2 or 3 scene
    if twice: # if case it identifies moment 2 or 3 even if covered in (most intense) moment 1 - covered two times
        notes.append(("INFO", f"moment 1 covers the '{twice[0]}' scene AND the interview also has that moment "
                              f"separately - scene discussed twice?"))
    
    if len(present) > 1 and not multi_noted: # to record when moment 1 spans more than 1 scene - very important!
        notes.append(("INFO", f"moment 1 covers several scenes: {'+'.join(present)}"))
    return {"has": has, "scene": scene, "scene_conf": conf, "scene_evidence": ev,
            "scenes": "+".join(present) or "none",  # only scenes backed by a strong keyword
            "source": source, "notes": notes}


# ----------------------------------------------------------------------------
# SECTION 4 - Q/A pairing
# ----------------------------------------------------------------------------

# sets the reason for dropping an answer (if any) and whether it is kept for analysis. Returns a list of answers with metadata.
def build_answers(turns, is_participant, min_words, remove_hesitations=True):
    """Merge consecutive participant turns within a moment and attach the
    preceding interviewer question(s) as context."""
    answers, question_buf = [], []
    for t in turns:
        if t["moment"] == "intro":
            continue
        if not is_participant(t["speaker"]):
            question_buf.append(t["text"])
            if answers:
                answers[-1]["_open"] = False  # interviewer spoke: next answer is new
            continue
        prev = answers[-1] if answers else None
        if (prev and prev["_open"] and prev["moment"] == t["moment"]
                and prev["evocation"] == t.get("evocation", False)):
            prev["raw"] += " " + t["text"]
            prev["end"] = t["end"]
        else:
            if prev:
                prev["_open"] = False
            answers.append({"moment": t["moment"], "start": t["start"], "end": t["end"],
                            "question": clean_text(" ".join(question_buf)),
                            "raw": t["text"], "evocation": t.get("evocation", False), "_open": True})
            question_buf = []
    # close & clean
    for a in answers:
        a.pop("_open", None)
        a["clean"] = clean_text(a["raw"], remove_hesitations)
        a["n_annotations"] = len(ANNOTATION_RE.findall(a["raw"]))
        a["n_hesitations"] = len(HESITATION_RE.findall(ANNOTATION_RE.sub(" ", a["raw"])))
        a["n_words"] = n_words(a["clean"])
        a["drop_reason"] = ("evocation" if a["evocation"]
                            else "too short" if a["n_words"] < min_words else "")
        a["kept"] = a["drop_reason"] == ""
    return answers

# removes the answers that are not kept and formats the remaining ones into a single document, either in Q/A format or as a plain text. Returns the formatted document as a string.
def to_document(answers, mode):
    kept = [a for a in answers if a["kept"]]
    if mode == "qa":
        parts = [(f"Q: {a['question']}\nA: {a['clean']}" if a["question"] else f"A: {a['clean']}")
                 for a in kept]
        return "\n\n".join(parts)
    return " ".join(a["clean"] for a in kept)


# ----------------------------------------------------------------------------
# Participant list
# ----------------------------------------------------------------------------
SEGMENTS = ["full", "moment1", "moment2", "moment3"]  # datasets written for MOSAIC
 
 
def read_participant_list(path, part_re):
    """One participant ID per line (first column if CSV/TSV). Blank lines,
    '#' comments and a header line are ignored. IDs must match the speaker
    labels in the transcripts (e.g. sub-gutslei0007)."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pid = re.split(r"[,\t;\s]+", line)[0].strip().strip('"')
        if part_re.match(pid):
            if pid not in ids:
                ids.append(pid)
        else:
            print(f"[participants file] ignored line (not a participant ID): '{line[:60]}'")
    return ids
 
 
# ----------------------------------------------------------------------------
# One transcript -> one participant folder
# ----------------------------------------------------------------------------
def process_file(f, args, is_participant, expected_pid=None):
    """Process one transcript. Returns a dict with status ('ok', 'excluded',
    'id_mismatch'), participant id, segmentation info, dataset rows,
    readable documents and audit rows."""
    res = {"file": str(f), "status": "ok", "rows": [], "answer_rows": [], "docs": {}, "audit": []}
    turns = parse_transcript(f)
    subs = sorted({t["speaker"] for t in turns if is_participant(t["speaker"])})
    if not subs:
        warnings.warn(f"[{f.name}] no speaker matches the participant pattern -> excluded.")
        res.update(status="excluded", pid=expected_pid or f.stem,
                   info={"file": f.name, "participant_id": expected_pid or f.stem, "intro_found": None,
                         "status": "excluded (no participant speaker)"})
        return res
    if len(subs) > 1:
        warnings.warn(f"[{f.name}] several participant IDs {subs}; all treated as participant.")
    pid = subs[0]
    res["pid"] = pid
    interviewer = main_interviewer(turns, is_participant)
    if expected_pid and expected_pid.lower() != pid.lower():
        warnings.warn(f"[{f.name}] speaker ID '{pid}' does not match expected participant '{expected_pid}'.")
        res.update(status="id_mismatch", pid=expected_pid,
                   info={"file": f.name, "participant_id": expected_pid, "intro_found": None,
                         "status": f"id mismatch (speaker label '{pid}')"})
        return res
 
    turns, info = segment(turns, is_participant, f.name)
    info["participant_id"] = pid
    info["interviewer"] = interviewer
    if turns is None:
        info["status"] = "excluded (no 'most intense')"
        res.update(status="excluded", info=info)
        return res
 
    answers = build_answers(turns, is_participant, args.min_words, not args.keep_hesitations)
    # stable answer ids, e.g. sub-gutslei0024_m1_a003 (moment, running number)
    for k, a in enumerate(answers, 1):
        a["answer_id"] = f"{pid}_m{a['moment']}_a{k:03d}"
    res["audit"] = [{"participant_id": pid, "file": f.name, **a} for a in answers]
 
    # Which moments exist, which scene is moment 1, and what fills the gaps?
    dec = decide_moment_sources(turns, answers, is_participant, args.no_fill)
    has = dec["has"]
    info["moments_present"] = ",".join(str(m) for m in (1, 2, 3) if has[m])
    info["m1_scene"], info["m1_scene_confidence"] = dec["scene"], dec["scene_conf"]
    info["m1_scenes"] = dec["scenes"]
    info["m1_scene_evidence"] = dec["scene_evidence"]
    info["moment2_source"], info["moment3_source"] = dec["source"][2], dec["source"][3]
    for sev, msg in dec["notes"]:
        if sev != "INFO":
            warnings.warn(f"[{f.name}] {msg}")
 
    for seg in SEGMENTS:
        if seg == "full":
            subset, source = answers, "own"
        else:
            m = int(seg[-1])
            src_m = 1 if dec["source"][m] == "copied_from_moment1" else m
            subset = [a for a in answers if a["moment"] == src_m]
            source = "copied_from_moment1" if src_m != m else "own"
        doc = to_document(subset, args.mode)
        n_kept = sum(a["kept"] for a in subset)
        info[f"{seg}_answers"] = n_kept
        info[f"{seg}_words"] = n_words(doc)
        if doc.strip():  # missing moments get no file and no row
            res["docs"][seg] = doc
            res["rows"].append({"segment": seg, "participant_id": pid, "reflection_answer": doc,
                                "n_answers": n_kept, "n_words": n_words(doc),
                                "n_moments": sum(has.values()), "moment_source": source,
                                "m1_scene": dec["scene"], "m1_scene_confidence": dec["scene_conf"],
                                "m1_scenes": dec["scenes"],
                                "interviewer": interviewer})
            # answer-level units: one row per kept answer (the recommended input for MOSAIC)
            for a in subset:
                if a["kept"]:
                    res["answer_rows"].append({
                        "segment": seg, "answer_id": a["answer_id"], "participant_id": pid,
                        "reflection_answer": a["clean"], "n_words": a["n_words"],
                        "moment": a["moment"], "moment_source": source, "question": a["question"],
                        "start": a["start"], "end": a["end"],
                        "m1_scene": dec["scene"], "m1_scene_confidence": dec["scene_conf"],
                        "m1_scenes": dec["scenes"], "interviewer": interviewer})
    info["n_annotations_removed"] = sum(a["n_annotations"] for a in answers)
    info["n_hesitations_removed"] = 0 if args.keep_hesitations else sum(a["n_hesitations"] for a in answers)
    ok_sources = {"own", "copied_from_moment1"}
    info["status"] = "ok" if {dec["source"][2], dec["source"][3]} <= ok_sources else "check moments"
    res["info"] = info
    return res
 
 
def write_participant(res, out_dir):
    """Write <out-dir>/participants/<id>/ (previous files of this participant are replaced)."""
    pdir = out_dir / "participants" / res["pid"]
    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.glob("*"):
        if old.is_file():
            old.unlink()
    for seg, doc in res["docs"].items():
        (pdir / f"{res['pid']}_{seg}.txt").write_text(doc, encoding="utf-8")
    pd.DataFrame(res["rows"]).to_csv(pdir / f"{res['pid']}_datasets.csv", index=False)
    pd.DataFrame(res["answer_rows"]).to_csv(pdir / f"{res['pid']}_answers.csv", index=False)
    pd.DataFrame(res["audit"]).to_csv(pdir / f"{res['pid']}_audit.csv", index=False)
    pd.DataFrame([res["info"]]).to_csv(pdir / f"{res['pid']}_segmentation.csv", index=False)
    return pdir
 
 
# ----------------------------------------------------------------------------
# All participant folders -> MOSAIC datasets + combined reports
# ----------------------------------------------------------------------------
def _read_all(pdirs, pattern):
    frames = []
    for d in pdirs:
        for f in d.glob(pattern):
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def combine(args, out_dir, raw_dir, listed=None):
    part_root = out_dir / "participants"
    pdirs = sorted(d for d in part_root.glob("*") if d.is_dir()) if part_root.exists() else []
    if listed is not None:
        present = {d.name for d in pdirs}
        missing = [p for p in listed if p not in present]
        pdirs = [d for d in pdirs if d.name in set(listed)]
        if missing:
            print(f"\n{len(missing)} listed participant(s) without processed data: {', '.join(missing)}")
    if not pdirs:
        sys.exit(f"No participant folders found in {part_root}")
 
    rows = _read_all(pdirs, "*_datasets.csv")
    arows = _read_all(pdirs, "*_answers.csv")
    audit = _read_all(pdirs, "*_audit.csv")
    rep = _read_all(pdirs, "*_segmentation.csv")
 
    for seg in SEGMENTS:
        out = raw_dir / f"{args.prefix}_{seg}_raw.csv"
        sub = rows[rows["segment"] == seg].drop(columns="segment") if not rows.empty else pd.DataFrame()
        sub = sub.sort_values("participant_id") if not sub.empty else sub
        sub.to_csv(out, index=False)
        print(f"{out}  ({len(sub)} participants)")
    for seg in SEGMENTS:  # answer-level datasets: one row per answer
        out = raw_dir / f"{args.prefix}_answers_{seg}_raw.csv"
        sub = arows[arows["segment"] == seg].drop(columns="segment") if not arows.empty else pd.DataFrame()
        sub = sub.sort_values(["participant_id", "answer_id"]) if not sub.empty else sub
        sub.to_csv(out, index=False)
        n_p = sub["participant_id"].nunique() if not sub.empty else 0
        print(f"{out}  ({len(sub)} answers, {n_p} participants)")
    audit.to_csv(out_dir / f"{args.prefix}_turns_audit.csv", index=False)
    rep.to_csv(out_dir / f"{args.prefix}_segmentation_report.csv", index=False)
    print(f"Combined audit + segmentation report in {out_dir}")
    print_summary(rep)
 
 
def print_summary(rep):
    if rep.empty:
        return
    for m in (2, 3):
        col = f"moment{m}_source"
        if col in rep:
            copied = rep[rep[col] == "copied_from_moment1"]
            if len(copied):
                print(f"\n{len(copied)} two-moment interview(s): moment {m} copied from moment 1: "
                      f"{', '.join(copied['participant_id'].astype(str))}")
    if "m1_scene" in rep:
        print("\nMoment-1 (most intense) scene:\n" +
              rep.groupby(["m1_scene", "m1_scene_confidence"]).size().rename("n").reset_index().to_string(index=False))
    if "status" in rep:
        bad = rep["status"] != "ok"
        if bad.any():
            print(f"\n{int(bad.sum())} participant(s) need attention - see the segmentation report:")
            cols = [c for c in ["participant_id", "file", "status", "moments_present", "m1_scenes",
                                "moment2_source", "moment3_source"] if c in rep]
            print(rep.loc[bad, cols].to_string(index=False))
 
 
# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare interview transcripts for topic modelling with MOSAIC "
                                             "(see file header for details).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-dir", help="Folder with transcripts (searched recursively): process all "
                                          "(or those in --participants-file) and combine")
    mode.add_argument("--input-file", help="Process ONE transcript into its participant folder (no combining)")
    mode.add_argument("--combine", action="store_true",
                      help="Only combine existing participant folders into the MOSAIC datasets")
    ap.add_argument("--participant", help="With --input-file: expected participant ID; mismatch -> exit code 2")
    ap.add_argument("--participants-file", help="List of participant IDs to include (one per line)")
    ap.add_argument("--data-dir", default="DATA", help="MOSAIC DATA folder (datasets go to <data-dir>/raw)")
    ap.add_argument("--out-dir", default=None, help="Folder for participant folders + reports "
                                                    "(default: <data-dir>/<prefix>_prep)")
    ap.add_argument("--prefix", default="gutslei", help="Dataset name prefix for MOSAIC")
    ap.add_argument("--participant-regex", default=DEFAULT_PARTICIPANT_RE)
    ap.add_argument("--mode", choices=["answers", "qa"], default="answers",
                    help="answers = participant text only (recommended for topic modelling)")
    ap.add_argument("--no-fill", "--no-fill-m3", dest="no_fill", action="store_true",
                    help="Do NOT copy moment 1 into a skipped moment 2/3 for 2-moment interviews "
                         "(default: copy, because there moment 1 = the celebration or mother scene)")
    ap.add_argument("--keep-hesitations", action="store_true",
                    help="Keep hesitations (um, uh, hmm, mhm ...) in the text (default: remove)")
    ap.add_argument("--min-words", type=int, default=3,
                    help="Drop answers shorter than this (e.g. 'Okay.', 'Yeah.')")
    args, unknown = ap.parse_known_args()
    if unknown:  # options meant for the other script (run_gutslei.sh passes the same options to both)
        print(f"Note: ignoring options not used by this script: {' '.join(unknown)}")
 
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / f"{args.prefix}_prep"
    raw_dir = data_dir / "raw"
    for d in (raw_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    part_re = re.compile(args.participant_regex, re.I)
    is_participant = lambda spk: bool(part_re.match(spk))
    listed = read_participant_list(args.participants_file, part_re) if args.participants_file else None
 
    # --- single transcript (used by run_gutslei.sh, one call per participant)
    if args.input_file:
        f = Path(args.input_file)
        if not f.is_file():
            print(f"ERROR: file not found: {f}")
            sys.exit(3)
        res = process_file(f, args, is_participant, expected_pid=args.participant)
        pdir = write_participant(res, out_dir)
        info = res["info"]
        print(f"{res['pid']}: {info.get('status')} | moments {info.get('moments_present', '-')} | "
              f"m1 scene {info.get('m1_scene', '-')} ({info.get('m1_scene_confidence', '-')}) | "
              f"{info.get('full_words', 0)} words -> {pdir}")
        sys.exit({"ok": 0, "excluded": 1, "id_mismatch": 2}[res["status"]])
 
    # --- combine only
    if args.combine:
        combine(args, out_dir, raw_dir, listed)
        return
 
    # --- whole folder
    in_dir = Path(args.input_dir)
    files = sorted(f for f in in_dir.rglob("*.txt") if out_dir.resolve() not in f.resolve().parents)
    if not files:
        sys.exit(f"No .txt files found in {in_dir}")
    done = []
    for f in files:
        res = process_file(f, args, is_participant)
        if listed is not None and res["pid"] not in listed:
            print(f"[{f.name}] participant {res['pid']} not in participants file -> skipped")
            continue
        write_participant(res, out_dir)
        done.append(res["pid"])
    dup = sorted({p for p in done if done.count(p) > 1})
    if dup:
        print(f"WARNING: several transcripts for {dup}; only the last one processed is kept.")
    combine(args, out_dir, raw_dir, listed)
 
 
if __name__ == "__main__":
    main()
 