#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module for phenomenological interview data (focusing on three film moments) for topic modelling
#                   - turns timestamped interview transcripts (.txt files) into preprocessed data for analysis (.csv files - for topic modelling with 
#                   MOSAIC ((https://github.com/romybeaute/MOSAIC), a BERTopic-based pipeline for analysing first-person experiential reports.
#                   - interview design - covering participants' most intense scene and two pre-selected scenes (celebration and mother scene)
#                   - step 1 - Parsing: parse timestamped speaker turns
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

from pathlib import Path          
import pandas as pd               

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 0: Configuration and patterns
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Basic features of the txt file
TIMESTAMP_RE = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\s*$") # timestamps in the format HH:MM:SS-HH:MM:SS
INTERVIEWER_RE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(.*)$") #any speaker who is not a participant is treated as an interviewer
DEFAULT_PARTICIPANT_RE = r"^sub-gutslei\d+$" 

# Set actual start of the interview: the interviewer's first mention of "most intense" moment. Everything before it is removed.
INTRO_START_RE = re.compile(r"\bmost\s+intense\b", re.I)

# Signals of readiness after evocation to remove - any combination of words (max. 8 words)
READY_WORDS = set("""yes yeah yep ok okay alright all right sure fine good perfect great ready i im m am me let lets s start started go we can now so and 
well here there it its is my eyes closed close see think set done too as no nope not yet clear question questions any understood understand of course that""".split())

# Participant questions to the interviewer - not to be analyzed
PARTICIPANT_QUESTION_MAX_WORDS = 8

# Signal of moment focus change - they open a moment only together with moment's keyword
MOMENT_TRANSITION_RE = re.compile(r"\b(?:another|other|next|different|new|final|last|second|third)\s+(?:moment|scene)\b", re.I)
MOMENT2_RE = re.compile(r"celebrat", re.I)  # celebration / celebrating
MOMENT3_RE = re.compile(r"\b(?:wom[ae]n|mother|mom)\b", re.I)  # the woman (mother) comes in
# Deferrals: when interviewer postpones moment participant brings up, to still focus on the moment they are in
DEFERRAL_RE = re.compile(r"\b(?:that|this|it)(?:'s|’s|\s+is|\s+was|\s+will\s+be)\s+(?:(?:on|in|for|part\s+of)\s+)?(?:a\s+|an\s+)?"
                         r"(?:another|other|different|separate|later)\s+(?:moment|scene|part)\b"
                         r"|\bwe(?:'ll|’ll|\s+will)\s+(?:\w+\s+){0,4}(?:after(?:wards)?|later)\b", re.I)

# Transcriber annotations ("(laughs)", "[inaudible]"). Transcripts sometimes have unmatched brackets, so annotations are removed in three steps (remove_annotations):
#   1. ANNOTATION_RE: a matched pair of any length - (laughs), [inaudible], (long pause while looking down) - or a short mismatched pair - (laughs];
#   2. STRAY_ANNOTATION_RE: an unclosed or unopened bracket next to a typical annotation word - "(laughs and then", "pause) I felt";
#   3. STRAY_BRACKET_RE: any bracket left over is removed on its own, keeping the words around it ("tense) and then (relieved" -> "tense and then relieved").
ANNOTATION_RE = re.compile(r"\([^()\[\]]*\)|\[[^()\[\]]*\]"            # matched pair, any length
                           r"|\([^()\[\]]{0,20}\]|\[[^()\[\]]{0,20}\)")  # mismatched pair, short only
ANNOTATION_WORDS = (r"laugh\w*|pause[sd]?|inaudible|unintelligible|unclear|sigh\w*|cough\w*|crosstalk|silence|chuckl\w*|"
                    r"overlapping|interrupt\w*|sniff\w*|exhal\w*|inhal\w*|clears throat|smil\w*")

STRAY_ANNOTATION_RE = re.compile(r"[(\[]\s*(?:" + ANNOTATION_WORDS + r")\b"
                                 r"|\b(?:" + ANNOTATION_WORDS + r")\s*[)\]]", re.I)
STRAY_BRACKET_RE = re.compile(r"[()\[\]]")
 
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
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 1: Parsing the transcript
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
            m = INTERVIEWER_RE.match(line)
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

# Function to remove transcriber annotations, also with unmatched brackets
def remove_annotations(text: str) -> str:
    """Remove transcriber annotations, also with unmatched brackets (see ANNOTATION_RE)."""
    text = ANNOTATION_RE.sub(" ", text)
    text = STRAY_ANNOTATION_RE.sub(" ", text)
    return STRAY_BRACKET_RE.sub(" ", text)

# Function to clean the text of each turn
def clean_text(text: str, remove_hesitations: bool = False) -> str:
    """Remove annotations (and optionally hesitations); tidy spaces, punctuation
    and dangling dashes. Words, numbers, case and punctuation are otherwise kept."""
    text = remove_annotations(text) # remove annotations (also with unmatched brackets)
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

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 2: Segmentation - work on individual turns to form blocks of answers for each moment
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to check if the turn is a transition to another moment/scene (not a deferral)
def is_transition(text: str) -> bool:
    """Interviewer text that moves to another part of the film (and is not a deferral). Checked on the text without hesitations and annotations,
    so "the last, uh, scene" or "the final (pause) scene" count too."""
    c = clean_text(text, True)
    return bool(MOMENT_TRANSITION_RE.search(text)) and not DEFERRAL_RE.search(text) 

# A moment prompt can take several turns, with short participant replies in between
PROMPT_REPLY_MAX_WORDS = 6 # participant replies
PROMPT_MAX_TURNS = 15 # prompt is searched for the scene keyword over at most this many turns

# Function to check if a participant turn is only a short reply during the interviewer's prompt
def is_prompt_reply(text: str) -> bool:
    """Backchannel, readiness reply, participant question or short reply (<= PROMPT_REPLY_MAX_WORDS words) - does not end a moment prompt."""
    return (is_backchannel(text) or is_ready_reply(text) or is_participant_question(text)
            or n_words(clean_text(text, True)) <= PROMPT_REPLY_MAX_WORDS)

# Function to find when moment/scene prompt ends (consecutive turns from the interviewer or participant backchannels)
def prompt_end(turns, i, is_participant):
    """Return the index of the last turn of the prompt starting at turn i. Interviewer turns and participant backchannels ('Yeah', 'Mhm', 'Short replies') are
    included; stops before the participant's first real answer or after PROMPT_MAX_TURNS turns."""
    j = i
    while (j + 1 < len(turns) and j + 1 - i < PROMPT_MAX_TURNS
           and (not is_participant(turns[j + 1]["speaker"]) # looks at next turn to check
                                  or is_prompt_reply(turns[j + 1]["text"]))):
        j += 1
    return j

# Function to find the first turn saying 'another moment' + scene keyword while making sure it is not a later reference to the scene (e.g. 'another moment' + 'celebration' but not 'mother' as a reference to the celebration scene)
def find_moment_start(turns, is_participant, topic_re, after_idx, exclude_re=None):
    """First interviewer turn (after 'after_idx') saying 'another/next moment/scene' + scene keyword.
        The keyword may also sit in the next interviewer turns, across short participant replies; only the INTERVIEWER's words are searched,
    so a participant mentioning a scene in passing cannot start a moment. If `exclude_re` is given, the prompt must not match that pattern."""
    for i, t in enumerate(turns): # i position, t turn
        if i <= after_idx or is_participant(t["speaker"]) or not is_transition(t["text"]):
            continue
        prompt = " ".join(x["text"] for x in turns[i:prompt_end(turns, i, is_participant) + 1]
                          if not is_participant(x["speaker"])) # interviewer's words only
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


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 3 - Scene classification and moment source decision - TO REVIEW
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Moment 1 = the participant's MOST INTENSE scene (any scene of the film). If it was the celebration, the interview skips 
# moment 2 -> moment 2 = moment 1. If it was the mother scene, the interview skips moment 3 -> moment 3 = moment 1.
# Keywords identify which scene moment 1 is about. (strong = 2 points, weak = 1). Keywords shared by several scenes (the boy, 
# crying, the table) are weak; scene-specific events are strong. Patterns use only non-capturing groups (?:...).
SCENES = {
    # the boxer's victory and the crowd celebrating
        "celebration": {"strong": [r"celebrat\w*", r"won", r"win", r"wins", r"winning", r"cheer\w*", r"congratulat\w*",
                                   # after the match: leaving the ring, embracing the opponent (phrases win over 'ring'/'opponent'/'hug' alone)
                                   r"(?:stepp?\w*|got|gets?|came|comes?|coming|climb\w*|walk\w*)\s+out\s+of\s+the\s+ring",
                                   r"hug\w*\s+(?:\w+\s+){0,2}opponent", r"opponent\W+(?:\w+\W+){0,6}?hug\w*"],
                        "weak":   [r"crowd"]},
    # the woman (the boy's mother) comes in
    "mother":          {"strong": [r"wom[ae]n", r"mother", r"mom", r"hug\w*"],
                        "weak":   [r"comes? in", r"came in", r"coming in", r"together"]},
    # father and son at the beach, having fun
    "beach":           {"strong": [r"beach", r"sea", r"sand", r"waves?", r"swim\w*"],
                        "weak":   [r"fun", r"laugh\w*", r"play\w*"]},
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
def moment1_scene(turns, is_participant):
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
    Used by both preprocessing.py and quality_control.py so they always agree.
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


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 4 and 5 - Q/A pairing and cleaning
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Helper function that looks ahead to find next participant's turn - to be used to check if interviewer interrupts an answer or starts a new question
def next_participant(turns, i, is_participant):
    """Next participant turn after i, skipping interviewer backchannels; None if a real interviewer turn comes first."""
    j = i + 1
    while j < len(turns) and not is_participant(turns[j]["speaker"]) and is_backchannel(turns[j]["text"]):
        j += 1
    return turns[j] if j < len(turns) and is_participant(turns[j]["speaker"]) else None

# Function to merge consecutive participant labeled turns within a moment (add corresponding question in audit file) and to clean and filter each answer (readiness, backchannel, participant question or too short); 
# To work at answer-level (units) instead of sentence-level for context
def build_answers(turns, is_participant, min_words, remove_hesitations=True):
    """Group participant turns into answers and attach the preceding interviewer question(s) as context.
    Participant turns of one answer are merged when they are consecutive, separated only by interviewer backchannels, or separated by a short interviewer interruption with continuation marks 
    on both sides. For a participant backchannel or readiness reply without any content answer, no answer is recorded but dropped, and leaves the question for the participant's next answer.
    Question context is reset at each new moment"""
    answers, question_buf = [], [] # output list and collected interviewer turns until next real answer
    bridged = 0  # interviewer turns bridged (to skip) since the last participant turn - stored for quality check
    current_moment = None

    # looping over all transcript turns
    for i, t in enumerate(turns):
        if t["moment"] == "intro": # skipping into
            continue
        if t["moment"] != current_moment: # new moment
            current_moment, question_buf, bridged = t["moment"], [], 0
        prev = answers[-1] if answers else None # last answer created so far (None if first)
        # interviwer turns
        if not is_participant(t["speaker"]): 
            nxt = next_participant(turns, i, is_participant) # participant turn that follows it
            same_answer = (prev is not None and prev["_open"] and nxt is not None # still open previous answer - to add more turns to it
                           and nxt["moment"] == prev["moment"] == t["moment"] # make sure it is all part of the same moment
                           and nxt.get("evocation", False) == prev["evocation"]) # make sure next participant turn is the same kind as previous answer - either both readiness or real answers
            if same_answer and (is_backchannel(t["text"]) or ( # interviewer turn is backchannel or short interruption (with continuation marks and max 3 words)
                    CONT_END_RE.search(prev["raw"]) and CONT_START_RE.search(nxt["text"]) and n_words(clean_text(t["text"])) <= CONTINUATION_MAX_WORDS)):
                bridged += 1          # interruption: keep the answer open, interviewer turn is not a question
                continue              # move to next turn t
            # interviewer turn is real question so next turn starts new answer
            question_buf.append(t["text"]) 
            if prev:
                prev["_open"] = False  # close previous answer
            bridged = 0
            continue
        # participant turns
        if (prev and prev["_open"] and prev["moment"] == t["moment"]
                and prev["evocation"] == t.get("evocation", False)): # previous answer still open, same moment and same kind - keep adding continuation
            prev["raw"] = join_continuation(prev["raw"], t["text"])
            prev["end"] = t["end"] # update end timestamp
            prev["n_turns_merged"] += 1
            prev["n_interviewer_turns_bridged"] += bridged
        else: # starting new answer
            if prev:
                prev["_open"] = False # close previous answer
            # label turn 
            evoc = t.get("evocation", False) # readiness flag for the loop
            ack = not evoc and is_backchannel(t["text"])  # participant backchannel on its own
            pq = not (evoc or ack) and is_participant_question(t["text"])  # participant question (e.g 'Can you repeat?')
            # creation of answer itself
            answers.append({"moment": t["moment"], "start": t["start"], "end": t["end"],
                            "question": clean_text(" ".join(question_buf)),
                            "raw": t["text"], "evocation": evoc, "backchannel": ack,
                            "participant_question": pq,
                            "n_turns_merged": 1, "n_interviewer_turns_bridged": 0,
                            "_open": not (ack or pq)}) # readiness replies stay open so several in a row merge into one
            if not (evoc or ack or pq): # no label means real answer which uses the question that has been accumulated
                question_buf = []  # 
        bridged = 0
    # close & clean
    for a in answers:
        a.pop("_open", None)
        a["clean"] = clean_text(a["raw"], remove_hesitations) # cleaned text - annotations removed, hesitations removed if requested, punctuation tidied
        a["n_annotations"] = (len(ANNOTATION_RE.findall(a["raw"]))
                              + len(STRAY_ANNOTATION_RE.findall(ANNOTATION_RE.sub(" ", a["raw"]))))
        a["n_stray_brackets"] = len(STRAY_BRACKET_RE.findall(STRAY_ANNOTATION_RE.sub(
            " ", ANNOTATION_RE.sub(" ", a["raw"]))))  # unmatched brackets around speech: check the transcript
        a["n_hesitations"] = len(HESITATION_RE.findall(remove_annotations(a["raw"])))
        a["n_words"] = n_words(a["clean"])
        a["drop_reason"] = ("evocation" if a["evocation"]
                            else "backchannel" if a["backchannel"]
                            else "participant question" if a["participant_question"]
                            else "too short" if a["n_words"] < min_words else "")
        a["kept"] = a["drop_reason"] == ""
    return answers

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 6 - Experience vs film description (sentence tags)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Each sentence of a kept answer is tagged as:
#   experiential - the participant's own experience: an experience word (feeling, bodily sensation, perception, attention, thought)  with the experiencer (I / my, or the generic "you" / "we", 
#                  e.g. "you feel sad watching this"), or without any character or film event (e.g. "there was this heaviness", "it was a very sad scene"); or an evaluation, which always expresses 
#                  the participant's appraisal ("that was not good"); or an autobiographical reference to the participant's own life;
#   narrative    - film content only: characters or film events, with no experience of the participant ("The woman comes in and hugs him.");
#   mixed        - both in one sentence (e.g. "When the boy cried I felt a lump.", "I felt sad for the boy.", "The father was cruel to the boy.");
#   other        - neither (e.g. "Yes, exactly.", "It was the end.").
# First-person perception and attention ("I saw him fall", "I was focused on his face") count as experience, and so do thoughts ("I wondered why...", "it reminded me of..."). Words that only 
# name the film (scene, film, movie, screen) are not film content, nor is the participant's own family ("my father"). 
# These are rule-based word lists: validate the tags on a hand-coded sample (analyse_gutslei.py sample / agreement). Only words on list count for tagging and classification

# The experiencer: first person, and the generic "you" / "we" that participants use for their own experience ("you feel sad watching this"), except in the fillers "you know" and "you mean".
FIRST_PERSON_RE = re.compile( r"\b(?:i|i'm|i've|i'd|i'll|me|my|myself|mine|we|we're|us|our|ourselves" 
                             r"|you(?!\s+(?:know|mean)\b)|you're|your|yourself)\b", re.I)
EXPERIENCE_RE = re.compile(r"\b(?:" + "|".join([
    # feeling, sensing, attending
    r"feel\w*", r"felt", r"sens\w*", r"notic\w*", r"aware\w*", r"attention", r"attentive", r"focus\w*",
    r"see", r"saw", r"seeing", r"watch\w*", r"look(?:ed|ing)?", r"listen\w*", r"hear\w*", r"heard",
    r"experienc\w*", r"emotion\w*", r"moved", r"moving", r"touch(?:ed|ing)", r"intens\w*", r"overwhelm\w*",
    # thoughts ('I think' alone is a hedge: only 'think about/of' counts)
    r"thought\w*", r"thinking", r"think(?:s)? (?:about|of)", r"wonder\w*", r"remember\w*", r"remind\w*",
    r"memor(?:y|ies)", r"imagin\w*", r"realis\w*", r"realiz\w*", r"mind", r"wish\w*", r"worr\w*",
    r"expect\w*", r"believ\w*", r"asked myself", r"understood", r"understand\w*",
    # body
    r"body", r"chest", r"throat", r"stomach", r"belly", r"gut", r"heart\w*", r"breath\w*", r"shoulders?",
    r"neck", r"jaw", r"hands?", r"arms?", r"legs?", r"skin", r"muscles?", r"head", r"eyes", r"tear(?:s|y|ful|ed)?",
    r"goosebumps?", r"shiver\w*", r"chills?", r"lump", r"knot", r"pressure", r"tight\w*", r"tense",
    r"tension", r"heav(?:y|iness)", r"warm\w*", r"cold", r"pain\w*", r"ache\w*", r"hurt\w*", r"cry\w*", r"cries", r"cried",
    # emotions
    r"sad\w*", r"happ(?:y|ier|iest|ily|iness)", r"unhapp\w*", r"joy\w*", r"fear\w*", r"afraid", r"scared", r"anxi\w*", r"nervous",
    r"relie[fv]\w*", r"ang(?:er|ry)", r"calm\w*", r"excit\w*", r"empath\w*", r"compassion", r"grief",
    r"griev\w*", r"lonel\w*", r"helpless\w*", r"uncomfortable", r"discomfort", r"disgust\w*", r"shock\w*",
    r"upset", r"sorrow", r"despair", r"hope\w*", r"love", r"weird", r"strange", r"emptiness", r"empty",
]) + r")\b", re.I)
# Evaluations always express the SPEAKER's appraisal, even about a character
# ("the father was cruel", "it was a good fight"), unlike emotion words, which
# can describe a character ("the boy was sad").
EVALUATION_RE = re.compile(r"\b(?:" + "|".join([
    r"good", r"bad", r"nice", r"beautiful", r"horrible", r"terrible", r"awful", r"unfair", r"cruel",
    r"wrong", r"sweet", r"cute", r"funny", r"boring", r"annoy\w*", r"disturb\w*", r"powerful", r"impressive",
]) + r")\b", re.I)
# Autobiographical references: the participant's own life, which the film may
# evoke ("my own father", "when I was a kid", "in my life"). They are part of
# the viewing experience, and are removed before looking for film content, so
# that "kid" in "when I was a kid" is not taken for a film character.
AUTOBIO_RE = re.compile(
    r"\b(?:my|our)\s+(?:own\s+)?(?:father|dad|daddy|mother|mom|parents?|family|son|daughter|kids?|"
    r"children|child|brother|sister|grandfather|grandmother|grandpa|grandma|grandparents?|husband|wife|"
    r"partner|childhood|life|past|home)\b"
    r"|\bwhen\s+(?:i|we)\s+(?:was|were)\s+(?:young|younger|little|small|a\s+(?:kid|child|boy|girl|teenager))\b"
    r"|\bin\s+my\s+(?:own\s+)?(?:life|childhood|family)\b", re.I)
# Characters and film events. A family word after "my/our/your (own)" is the
# participant's own family ("my own father"), not a film character.
NARRATIVE_RE = re.compile(r"(?<!\bmy )(?<!\bour )(?<!\byour )(?<!\bown )\b(?:" + "|".join([
    # characters (third person)
    r"he", r"him", r"his", r"she", r"her", r"they", r"them", r"their", r"boy", r"kid", r"child", r"son",
    r"father", r"dad", r"daddy", r"mother", r"mom", r"woman", r"women", r"boxer", r"champ", r"man",
    r"men", r"crowd", r"people", r"opponent", r"doctor", r"nurse", r"character\w*",
    # film events (words that only NAME the film - scene, film, movie, screen - are not
    # narrative: "it was a very sad scene" is about the participant's sadness)
    r"ring", r"fight\w*", r"box\w*",
    r"punch\w*", r"hits?", r"knock\w*", r"falls?", r"fell", r"wins?", r"won", r"winning", r"die[sd]?",
    r"dying", r"dead", r"comes? in", r"came in", r"hugs?", r"hugged", r"lifts?", r"lifted",
    r"celebrat\w*", r"beach", r"table", r"says", r"said", r"talks?", r"talked", r"walks?", r"walked",
]) + r")\b", re.I)

# Function to split sentences for tagging 
try:
    from nltk.tokenize import PunktSentenceTokenizer  # same splitter as MOSAIC
    _PUNKT = PunktSentenceTokenizer()
    
    # Splits text into list of sentences leaving out any that are empty or only whitespace - matching what MOSAIC will produce
    def split_sentences(text):
        return [x for x in _PUNKT.tokenize(text) if x.strip()]
except ImportError:
    def split_sentences(text):
        return [x for x in re.split(r"(?<=[.!?…])\s+", text) if x.strip()]
 
# Function to tag answer's sentences according to rule
def tag_sentence(sentence):
    """Return (tag, experience markers, narrative markers, first person, autobiographical markers) for one sentence.
    Rule:
        autobiographical references are removed before film content is looked for;
      experience signal = an evaluation word, OR an experience word with either the experiencer (I, my, generic you/we) or no character/event in the sentence;
      narrative signal  = any character or film-event word;
      both -> mixed, experience only -> experiential, narrative only -> narrative, neither -> other."""
    auto = [m.lower() for m in AUTOBIO_RE.findall(sentence)] # list of autiobiographical markers
    exp = [m.lower() for m in EXPERIENCE_RE.findall(sentence)] # list of experience markers
    evl = [m.lower() for m in EVALUATION_RE.findall(sentence)] # list of evaluation markers
    film_text = AUTOBIO_RE.sub(" ", sentence)   # copy of sentence without autobiographical markers - own life is not film content
    narr = [m.lower() for m in NARRATIVE_RE.findall(film_text)] # list of narrative markers
    fp = bool(FIRST_PERSON_RE.search(sentence)) # check if personal/experiencer sentence
    
    # core rule - evaluation OR experience with personal or no narrative in it
    experiential = bool(evl) or bool(auto) or (bool(exp) and (fp or not narr))
    if experiential and narr:
        tag = "mixed"
    elif experiential:
        tag = "experiential"
    elif narr:
        tag = "narrative"
    else:
        tag = "other"
    return tag, exp + evl, narr, fp, auto
 
 # Function to go over kept answers prepared before and tag sentences
def tag_answers(answers, pid):
    """Tag the sentences of every kept answer. Adds tags counts, the narrative share and an answer type to each answer; 
    Returns one row per sentence (sentence-level output, update answer-level)."""
    rows = []
    for a in answers:
        if not a["kept"]:
            continue
        counts = {"experiential": 0, "mixed": 0, "narrative": 0, "other": 0}
        n_auto = 0 # count for autobiographical - important to know but not impacting result
        for k, sent in enumerate(split_sentences(a["clean"]), 1): # numbering of sentences start in 1 - splitting according to MOSAIC
            tag, exp, narr, fp, auto = tag_sentence(sent)
            counts[tag] += 1
            n_auto += bool(auto)
            rows.append({"answer_id": a["answer_id"], "participant_id": pid, "moment": a["moment"],
                         "sentence_no": k, "sentence": sent, "tag": tag, "first_person": fp,
                         "autobiographical": bool(auto),
                         "experience_markers": ", ".join(exp), "narrative_markers": ", ".join(narr),
                         "autobiographical_markers": ", ".join(auto)})
        # summaries for the answer
        n = sum(counts.values()) # total number of sentences
        a.update({f"n_{t}": c for t, c in counts.items()}) # count of each tag added
        a["n_sentences"] = n
        a["n_autobiographical"] = n_auto  # sentences referring to the participant's own life
        a["narrative_share"] = round(counts["narrative"] / n, 3) if n else 0.0 # avoids dividing by zero
        # answer type: purely narrative, purely experiential, mixed (both), or other (neither)
        has_exp = counts["experiential"] + counts["mixed"] > 0
        has_narr = counts["narrative"] + counts["mixed"] > 0
        # classification of answer type rule
        a["answer_type"] = ("mixed" if has_exp and has_narr else "experiential" if has_exp
                            else "narrative" if has_narr else "other")
    return rows
 
# Function to turn list of answers into one text document - called once per moment and participant - participant level dataset (not for MOSAIC)
def to_document(answers, mode):
    """mode input decides how participant's kept answers are joined into document per participant - either just answers or with question-answer pair"""
    kept = [a for a in answers if a["kept"]]
    if mode == "qa":
        parts = [(f"Q: {a['question']}\nA: {a['clean']}" if a["question"] else f"A: {a['clean']}")
                 for a in kept]
        return "\n\n".join(parts)
    return " ".join(a["clean"] for a in kept)
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 7 - Preprocessing
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SEGMENTS = ["full", "moment1", "moment2", "moment3"]  # datasets written for MOSAIC
 
# Function to prepare list with participant IDs 
def read_participant_list(path, part_re):
    """Needs one participant ID per line in txt file (first column if CSV/TSV). 
    Blank lines, '#' comments and a header line are ignored. IDs must match the speaker labels in the transcripts (e.g. sub-gutslei0007)."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines(): # reads file line by line
        line = line.strip() # removes spaces, tabs
        if not line or line.startswith("#"): # skip blank lines and comment lines
            continue
        pid = re.split(r"[,\t;\s]+", line)[0].strip().strip('"') # taking first column and not including anything that is not sub-gutsleiXXXX
        if part_re.fullmatch(pid): # if pattern matches to ID
            if pid not in ids:
                ids.append(pid)
        else: # report typos
            print(f"[participants file] ignored line (not a participant ID): '{line[:60]}'")
    return ids
 

# Function to run the whole preprocessing pipeline for one transcript (inside corresponding participant folder)
def process_file(f, args, is_participant, expected_pid=None):
    """Process one transcript. Returns a dict with status ('ok', 'excluded','id_mismatch'), participant id, segmentation info, dataset rows, readable documents and audit rows.
    row - one row per segment/moment with whole document txt; answer_rows - one row per kept answer (MOSAIC input); sentence_rows - one row per sentence, with tags"""
    # prepare readable file, which is not the dataset
    res = {"file": str(f), "status": "ok", "rows": [], "answer_rows": [], "sentence_rows": [],
           "docs": {}, "audit": []} 
    # Parsing the transcript txt file
    turns = parse_transcript(f)
    # Setting the interviewer and participant
    pid = next((t["speaker"] for t in turns if is_participant(t["speaker"])), None)
    if pid is None: # wrong file
        warnings.warn(f"[{f.name}] no speaker matches the participant pattern -> excluded.")
        res.update(status="excluded", pid=expected_pid or f.stem,
                   info={"file": f.name, "participant_id": expected_pid or f.stem, "intro_found": None,
                         "status": "excluded (no participant speaker)"})
        return res
    if expected_pid and expected_pid.lower() != pid.lower(): # transcript id not match participant id
            warnings.warn(f"[{f.name}] speaker ID '{pid}' does not match expected participant '{expected_pid}'.")
            res.update(status="id_mismatch", pid=expected_pid,
                       info={"file": f.name, "participant_id": expected_pid, "intro_found": None,
                             "status": f"id mismatch (speaker label '{pid}')"})
            return res
    res["pid"] = pid
    interviewer = set_interviewer(turns, is_participant)
    # Segmenting the turns into moments and outputing summary dictionary (info)
    turns, info = segment(turns, is_participant, f.name)
    info["participant_id"] = pid
    info["interviewer"] = interviewer
    if turns is None: # if most intense moment was not found thus transcript is broken
        info["status"] = "excluded (no 'most intense')"
        res.update(status="excluded", info=info)
        return res
    # Building answer-level units for MOSAIC
    answers = build_answers(turns, is_participant, args.min_words, not args.keep_hesitations)
    # Gives ids to answers, e.g. sub-gutslei0024_m1_a003 (moment, running number)
    for k, a in enumerate(answers, 1):
        a["answer_id"] = f"{pid}_m{a['moment']}_a{k:03d}"
    # Tagging of sentences and classification of answers    
    res["sentence_rows"] = tag_answers(answers, pid)  # experience vs film description
    res["audit"] = [{"participant_id": pid, "file": f.name, **a} for a in answers]
    # Setting which scenes are covered in moment 1 (most intense), adding to summary dictionary (info)
    dec = decide_moment_sources(turns, answers, is_participant, args.no_fill) # decides for each moment whether text is own, copied (from most intense) or missing
    has = dec["has"]
    info["moments_present"] = ",".join(str(m) for m in (1, 2, 3) if has[m]) # which moments are present
    info["m1_scene"], info["m1_scene_confidence"] = dec["scene"], dec["scene_conf"]
    info["m1_scenes"] = dec["scenes"]
    info["m1_scene_evidence"] = dec["scene_evidence"]
    info["moment2_source"], info["moment3_source"] = dec["source"][2], dec["source"][3]
    for sev, msg in dec["notes"]: # show warning and error notes
        if sev != "INFO":
            warnings.warn(f"[{f.name}] {msg}")
    # Building the datasets 
    for seg in SEGMENTS:
        if seg == "full": # full interview dataset
            subset, source = answers, "own" # no copies - can have 2 or 3 moments
        else: # each moment dataset
            m = int(seg[-1]) # moment number
            src_m = 1 if dec["source"][m] == "copied_from_moment1" else m # which moment text comes from - moment 1 if moment is copied from most intense
            subset = [a for a in answers if a["moment"] == src_m] # copying answers if moment was covered in most intense (moment 1)
            source = "copied_from_moment1" if src_m != m else "own" 
        doc = to_document(subset, "answers") # 1 document per transcript for later - participant-level analysis (input for MOSAIC is always only answers)
        n_kept = sum(a["kept"] for a in subset) # number of kept answers
        info[f"{seg}_answers"] = n_kept
        info[f"{seg}_words"] = n_words(doc) # number of kept words
        if doc.strip():  # missing moments get no file and no row
            res["docs"][seg] = to_document(subset, args.mode) # readable, dependent on mode
            res["rows"].append({"segment": seg, "participant_id": pid, "cleaned_reflection": doc, # cleaned_reflection is the column MOSAIC reads
                                "n_answers": n_kept, "n_words": n_words(doc),
                                "n_moments": sum(has.values()), "moment_source": source, # shows n_moments = 2 if only two moments, even if copied
                                "m1_scene": dec["scene"], "m1_scene_confidence": dec["scene_conf"],
                                "m1_scenes": dec["scenes"],
                                "interviewer": interviewer})
            # answer-level units: one row per kept answer (the recommended input for MOSAIC)
            for a in subset:
                if a["kept"]:
                    res["answer_rows"].append({ # kept answer in this segment 
                        "segment": seg, "answer_id": a["answer_id"], "participant_id": pid,
                        "cleaned_reflection": a["clean"], "n_words": a["n_words"],
                        "moment": a["moment"], "moment_source": source, "question": a["question"],
                        "start": a["start"], "end": a["end"],
                        "m1_scene": dec["scene"], "m1_scene_confidence": dec["scene_conf"],
                        "m1_scenes": dec["scenes"], "interviewer": interviewer,
                        "n_sentences": a["n_sentences"], "n_experiential": a["n_experiential"],
                        "n_mixed": a["n_mixed"], "n_narrative": a["n_narrative"], "n_other": a["n_other"],
                        "n_autobiographical": a["n_autobiographical"],
                        "narrative_share": a["narrative_share"], "answer_type": a["answer_type"]})
    # Final status
    info["n_annotations_removed"] = sum(a["n_annotations"] for a in answers)
    info["n_hesitations_removed"] = 0 if args.keep_hesitations else sum(a["n_hesitations"] for a in answers)
    ok_sources = {"own", "copied_from_moment1"}
    sources = {dec["source"][m] for m in (1, 2, 3)} # moment 1 must have text too
    info["status"] = "ok" if sources <= ok_sources else "check moments" # ok if all moments are own/copied, otherwise missing/ambiguous gives check warning
    res["info"] = info
    return res

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Section 8 - Writing up datasets and documents for each participant or combined across participants - MOSAIC dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to save all results in participant folder 
def write_participant(res, out_dir):
    """Write <out-dir>/participants/<id>/."""
    pdir = out_dir / "participants" / res["pid"]
    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.glob("*"): # clean up in case still have files from past test runs
        if old.is_file():
            old.unlink()
    for seg, doc in res["docs"].items(): # writes txt file per segment that has text
        (pdir / f"{res['pid']}_{seg}.txt").write_text(doc, encoding="utf-8")
    pd.DataFrame(res["rows"]).to_csv(pdir / f"{res['pid']}_datasets.csv", index=False)
    pd.DataFrame(res["answer_rows"]).to_csv(pdir / f"{res['pid']}_answers.csv", index=False)
    pd.DataFrame(res["sentence_rows"]).to_csv(pdir / f"{res['pid']}_sentences.csv", index=False)
    pd.DataFrame(res["audit"]).to_csv(pdir / f"{res['pid']}_audit.csv", index=False)
    pd.DataFrame([res["info"]]).to_csv(pdir / f"{res['pid']}_segmentation.csv", index=False)
    return pdir
 
# Helper function to combine csv files across participants stacking tables
def _read_all(pdirs, pattern):
    """Collects every answers csv into one table with all participants' answers, which is what is given to MOSAIC or for overall totals"""
    frames = []
    for d in pdirs: # go through each participant folder
        for f in d.glob(pattern): # find file that mataches for example "_answers.csv"
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError: # if empty folder
                pass
    frames = [x for x in frames if not x.empty] # removes tables that have no row
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() # stacks tables, matching columns by name
 
# Function to combine answers for each segment (full, 1, 2 or 3) from all participants/transcripts - decides what to combine and where
def combine(args, out_dir, mosaic_dir, listed=None):
    """Receives participant id list and writes csv files combining all participants"""
    part_root = out_dir / "participants"
    pdirs = sorted(d for d in part_root.glob("*") if d.is_dir()) if part_root.exists() else [] # all participants folder sorted
    if listed is not None:
        present = {d.name.lower() for d in pdirs} # participant ids we have folder
        missing = [p for p in listed if p.lower() not in present] # participant ids that don't have folder
        pdirs = [d for d in pdirs if d.name.lower() in {p.lower() for p in listed}] # keep only folders of listed participants
        if missing:
            print(f"\n{len(missing)} listed participant(s) without processed data: {', '.join(missing)}")
    if not pdirs:
        sys.exit(f"No participant folders found in {part_root}")
    
    # stack tables across all participants
    rows = _read_all(pdirs, "*_datasets.csv") 
    arows = _read_all(pdirs, "*_answers.csv")
    srows = _read_all(pdirs, "*_sentences.csv")
    audit = _read_all(pdirs, "*_audit.csv")
    rep = _read_all(pdirs, "*_segmentation.csv")
    # split the dataset tables per moment/segment and write the 8 MOSAIC files (4 answer level and 4 participant level)
    for seg in SEGMENTS: # document dataset: one row per participant
        out = mosaic_dir / f"{args.prefix}_{seg}_preprocessed.csv" # builds output filename
        sub = rows[rows["segment"] == seg].drop(columns="segment") if not rows.empty else pd.DataFrame()
        sub = sub.sort_values("participant_id") if not sub.empty else sub
        sub.to_csv(out, index=False) 
        print(f"{out}  ({len(sub)} participants)")
    for seg in SEGMENTS:  # answer-level datasets: one row per answer
        out = mosaic_dir / f"{args.prefix}_answers_{seg}_preprocessed.csv"
        sub = arows[arows["segment"] == seg].drop(columns="segment") if not arows.empty else pd.DataFrame()
        sub = sub.sort_values(["participant_id", "answer_id"]) if not sub.empty else sub
        sub.to_csv(out, index=False)
        n_p = sub["participant_id"].nunique() if not sub.empty else 0
        print(f"{out}  ({len(sub)} answers, {n_p} participants)")
    # review files without splitting by segment
    audit.to_csv(out_dir / f"{args.prefix}_turns_audit.csv", index=False)
    srows.to_csv(out_dir / f"{args.prefix}_sentence_tags.csv", index=False)
    rep.to_csv(out_dir / f"{args.prefix}_segmentation_report.csv", index=False)
    print(f"Combined audit + segmentation report in {out_dir}")
    print_summary(rep)
 
# Function to print overview of all the datasets - which interviews need attention/have segmentation problems
def print_summary(rep):
    if rep.empty:
        return
    # check which participants have 2-moment interview
    for m in (2, 3):
        col = f"moment{m}_source"
        if col in rep:
            copied = rep[rep[col] == "copied_from_moment1"]
            if len(copied):
                print(f"\n{len(copied)} two-moment interview(s): moment {m} copied from moment 1: "
                      f"{', '.join(copied['participant_id'].astype(str))}")
    # print the scenes present in most intense moment
    if "m1_scene" in rep:
        print("\nMoment-1 (most intense) scene:\n" +
              rep.groupby(["m1_scene", "m1_scene_confidence"]).size().rename("n").reset_index().to_string(index=False))
    # check interviews that might have segmentation problems
    if "status" in rep:
        bad = rep["status"] != "ok"
        if bad.any():
            print(f"\n{int(bad.sum())} participant(s) need attention - see the segmentation report:")
            cols = [c for c in ["participant_id", "file", "status", "moments_present", "m1_scenes",
                                "moment2_source", "moment3_source"] if c in rep]
            print(rep.loc[bad, cols].to_string(index=False))
 
 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # exit code: 0 processed, 10 excluded (no most intense or no participant speaker), 11 speaker ID differs from --participant, 12 file not found
    ap = argparse.ArgumentParser(allow_abbrev=False, 
                                    description="Prepare interview transcripts for topic modelling with MOSAIC "
                                                "(see file header for details).")
    # only one added argument allowed
    mode = ap.add_mutually_exclusive_group(required=True)
    # process every transcript in the folder, then combine
    mode.add_argument("--input-dir", help="Folder with transcripts (searched recursively): process all "
                                          "(or those in --participants-file) and combine")
    # process one transcript into its participant folder, no combining
    mode.add_argument("--input-file", help="Process ONE transcript into its participant folder (no combining)")
    # only combine the existing participant folders
    mode.add_argument("--combine", action="store_true",
                      help="Only combine existing participant folders into the MOSAIC datasets")
    # participant ids
    ap.add_argument("--participant", help="With --input-file: expected participant ID; mismatch -> exit code 11") 
    ap.add_argument("--participants-file", help="List of participant IDs to include (one per line)")
    ap.add_argument("--data-dir", default="DATA", help="MOSAIC DATA folder (datasets go to <data-dir>/preprocessed)")
    ap.add_argument("--out-dir", default=None, help="Folder for participant folders + reports "
                                                    "(default: <data-dir>/derivatives)")
    ap.add_argument("--prefix", default="gutslei", help="Dataset name prefix for MOSAIC")
    ap.add_argument("--participant-regex", default=DEFAULT_PARTICIPANT_RE) # pattern for participant ID
    # set format of readable documents - just answers or Q&A format
    ap.add_argument("--mode", choices=["answers", "qa"], default="answers",
                    help="format of the readable .txt documents: answers only (default) or "
                         "question-answer pairs; the MOSAIC datasets always contain answers only")
    ap.add_argument("--no-fill", "--no-fill-m3", dest="no_fill", action="store_true",
                    help="Do NOT copy moment 1 into a skipped moment 2/3 for 2-moment interviews "
                         "(default: copy, because there moment 1 = the celebration or mother scene)")
    # clearning options
    ap.add_argument("--keep-hesitations", action="store_true",
                    help="Keep hesitations (um, uh, hmm, mhm ...) in the text (default: remove)")
    ap.add_argument("--min-words", type=int, default=3,
                    help="Drop answers shorter than this (e.g. 'Okay.', 'Yeah.')")
    # check proper options, otherwise doesn't run
    ap.add_argument("--check-options", action="store_true", help=argparse.SUPPRESS)
    args, unknown = ap.parse_known_args()
    # run_preproc.sh passes the same options to both scripts: options of qc_gutslei.py are ignored here; anything else is a typo and stops the run (instead of silently using a default)
    QC_ONLY = {"--min-sentence-words", "--long-sentence", "--min-moment-words", "--min-participant-share",
               "--top-n", "--long-answer-tokens", "--per-participant-dir"}
    bad = [u for u in unknown if u.startswith("-") and u.split("=")[0] not in QC_ONLY]
    if bad:
        sys.exit(f"Unknown option(s): {' '.join(bad)} (see --help)")
    if args.check_options:  # options are valid: stop here
        sys.exit(0)
    
    # Setting up and folder creation
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "derivatives"
    mosaic_dir = data_dir / "preprocessed" # where MOSAIC's optuna_search.py looks for <dataset>_preprocessed.csv
    for d in (mosaic_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    part_re = re.compile(args.participant_regex, re.I)
    # participant check variable set
    is_participant = lambda spk: bool(part_re.fullmatch(spk))
    # reads participant list if given
    listed = read_participant_list(args.participants_file, part_re) if args.participants_file else None 
 
    # --- Mode 2 - single transcript (used by run_preproc.sh, one call per participant)
    if args.input_file:
        f = Path(args.input_file)
        if not f.is_file(): # checks if file exists
            print(f"ERROR: file not found: {f}")
            sys.exit(12)
        # processing transcript, checking id and writing into folder
        res = process_file(f, args, is_participant, expected_pid=args.participant)
        pdir = write_participant(res, out_dir)
        info = res["info"]
        # print the main information
        print(f"{res['pid']}: {info.get('status')} | moments {info.get('moments_present', '-')} | "
              f"m1 scene {info.get('m1_scene', '-')} ({info.get('m1_scene_confidence', '-')}) | "
              f"{info.get('full_words', 0)} words -> {pdir}")
        sys.exit({"ok": 0, "excluded": 10, "id_mismatch": 11}[res["status"]])
 
    # --- Mode 3 - combine only
    if args.combine:
        combine(args, out_dir, mosaic_dir, listed)
        return
 
    # --- Mode 1 - whole folder
    in_dir = Path(args.input_dir)
    # find all txt file in folder and subfolders
    files = sorted(f for f in in_dir.rglob("*.txt") if out_dir.resolve() not in f.resolve().parents)
    if not files:
        sys.exit(f"No .txt files found in {in_dir}")
    # Process each file - one per participant
    for f in files:
        res = process_file(f, args, is_participant)
        if listed is not None and res["pid"].lower() not in {p.lower() for p in listed}: 
            print(f"[{f.name}] participant {res['pid']} not in participants file -> skipped")
            continue
        # saves participant data 
        write_participant(res, out_dir)
    # combines answers into dataset ready for MOSAIC
    combine(args, out_dir, mosaic_dir, listed)
 
 
if __name__ == "__main__":
    main()
