#!/usr/bin/env bash
# =============================================================================
# run_preproc.sh -- Preprocess all interview transcripts listed in a
#participants file, combine them into MOSAIC datasets and run quality checks

# title           : run_preproc.sh
# description     : Preprocess all interview transcripts listed in a participant file, combine them into MOSAIC datasets and run quality checks
#                   - step 1 - read participant list
#                   - step 2 - find transcript for each participant; participants without transcript or with multiple are reported and skipped 
#                   - step 3 - run preprocessing.py on each transcript separately writing <out>/participants/<ID>/ and a log in <out>/logs/<ID>.log 
#                   (fail transcript doesn't stop others)
#                   - step 4 - combine all participant folders into MOSAIC datasets: <mosaic-data>/preprocessed/gutslei_answers_{full,moment1,moment2,moment3}_preprocessed.csv
#                   (one row per answer, recommended) and <mosaic-data>/preprocessed/gutslei_{full,moment1,moment2,moment3}_preprocessed.csv (one row per participant).
#                   - step 5 - run quality_control.py on selected participants (report in <out>/qc, plus <ID>_qc_flags.csv in each participant folder).
#                   - step 6 - write summary: <out>/run_summary.txt and <out>/selected_files.tsv(participant -> transcript path)
# usage           : ./run_preproc.sh                           (uses the default paths below)
#                   ./run_preproc.sh -i /path/to/transcripts -1 participants.txt -m /path/to/MOSAIC/DATA -- --min-word 33
#
#                   -i DIR   transcripts folder                (default: /TRANSCRIPTS_DIR)
#                   -l FILE  participant list                  (default: PARTICIPANTS_FILE)
#                   -m DIR   MOSAIC DATA folder (MOSAIC/DATA)  (default: MOSAIC_DATA_DIR)
#                   -o DIR   output folder                     (default: <MOSAIC DATA>/derivatives)
#                   -p EXE   python executable                 (default: python3)
#                   -g GLOB  transcript file patterns          (default: *_task-emtint.txt)
#                   -h       help
#                   Arguments after "--" are passed to BOTH Python scripts, so the two use
#                   identical settings (e.g. --min-words 3, --no-fill, --keep-hesitations);
#                   each script ignores (and reports) options that only the other one uses.
# output          : exit code 0 if every listed participant was processed without error, 1 otherwise (see run_summary.txt)
# dependencies    : Works with bash >= 3.2 (macOS default) and Linux.
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-10-04
# version         : 1.0
# =============================================================================

# Shell safety settings
set -uo pipefail
 
# Set default values and sets paths
TRANSCRIPTS_DIR="/data00/GUTS/francisca/interview_preliminary_analysis/data"
PARTICIPANTS_FILE="/data00/GUTS/francisca/interview_preliminary_analysis/participants.txt"
MOSAIC_DATA_DIR="/data00/GUTS/francisca/interview_preliminary_analysis/NLP_interview_analysis/MOSAIC/DATA"
OUT_DIR=""
PYTHON="python3"
TRANSCRIPT_GLOB="*_task-emtint.txt"    # only these files are transcripts (other .txt files are ignored)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
# Function to print help
usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; }

# Reading the options for the script 
while getopts "i:l:m:o:p:g:h" opt; do
    case "$opt" in
        i) TRANSCRIPTS_DIR="$OPTARG" ;;
        g) TRANSCRIPT_GLOB="$OPTARG" ;;
        l) PARTICIPANTS_FILE="$OPTARG" ;;
        m) MOSAIC_DATA_DIR="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        p) PYTHON="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS=("$@")                      # passed to both Python scripts
EXTRA_STR="$*"
[ -z "$OUT_DIR" ] && OUT_DIR="$MOSAIC_DATA_DIR/derivatives"
 
PREPARE="$SCRIPT_DIR/preprocessing.py"
QC="$SCRIPT_DIR/quality_control.py"
 
# ---------------------------------------------------------------- checks (to run successfully)
# Error function to stop the script
fail() { echo "ERROR: $*" >&2; exit 1; }
# checks whether program exists, if file exists, if folder exists
command -v "$PYTHON" >/dev/null 2>&1 || fail "python not found: $PYTHON (use -p)" 
[ -f "$PREPARE" ] || fail "preprocessing.py not found next to this script ($SCRIPT_DIR)" 
[ -f "$QC" ]      || fail "quality_control.py not found next to this script ($SCRIPT_DIR)"
[ -d "$TRANSCRIPTS_DIR" ]   || fail "transcripts folder not found: $TRANSCRIPTS_DIR (use -i)" 
[ -f "$PARTICIPANTS_FILE" ] || fail "participant list not found: $PARTICIPANTS_FILE (use -l)"
 
TRANSCRIPTS_DIR="$(cd "$TRANSCRIPTS_DIR" && pwd)"      # absolute paths
mkdir -p "$OUT_DIR" "$MOSAIC_DATA_DIR" || fail "cannot create output folders"        # creates output folders
OUT_DIR="$(cd "$OUT_DIR" && pwd)" || fail "cannot open output folder: $OUT_DIR"
MOSAIC_DATA_DIR="$(cd "$MOSAIC_DATA_DIR" && pwd)" || fail "cannot open MOSAIC data folder: $MOSAIC_DATA_DIR"
SELECTED="$OUT_DIR/_selected"          # links to the transcripts used (input for QC)
LOGS="$OUT_DIR/logs"               # logs outputting each python call
rm -rf "$SELECTED"
mkdir -p "$SELECTED" "$LOGS"
 
SUMMARY="$OUT_DIR/run_summary.txt"  
MAPPING="$OUT_DIR/selected_files.tsv"
printf "participant_id\ttranscript\n" > "$MAPPING"
 
# ---------------------------------------------------------------- options
# check the options passed after "--" once, before processing anyone: a typo stops the run here instead of failing every participant
"$PYTHON" "$PREPARE" --combine --check-options ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    || fail "invalid option(s) for preprocessing.py (see message above)"
"$PYTHON" "$QC" --input-dir . --check-options ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    || fail "invalid option(s) for quality_control.py (see message above)"
 
# ---------------------------------------------------------------- participant list
# strip Windows line endings, comments, blanks; keep the first column; skip header 
# reading participant list, creating one ID per line
PIDS="$(tr -d '\r' < "$PARTICIPANTS_FILE" | sed 's/#.*//' \
        | awk -F'[,;\t ]+' 'NF && $1 != "" {print $1}' | tr -d '"' \
        | grep -Ei '^sub-' | awk '!seen[$0]++')"
N_LISTED=$(printf "%s\n" "$PIDS" | grep -c . || true)    # counts number of IDs
[ "$N_LISTED" -gt 0 ] || fail "no participant IDs (starting with 'sub-') in $PARTICIPANTS_FILE"

# Finding the transcripts 
ALL_TXT="$(find "$TRANSCRIPTS_DIR" -type f -iname "$TRANSCRIPT_GLOB" ! -path "$OUT_DIR/*" | sort)" # list every .txt file and matches pattern

# Output current information - where everything comes from and goes to 
echo "Transcripts : $TRANSCRIPTS_DIR ($(printf "%s\n" "$ALL_TXT" | grep -c . || true) files matching $TRANSCRIPT_GLOB)"
echo "Participants: $PARTICIPANTS_FILE ($N_LISTED listed)"
echo "Output      : $OUT_DIR"
echo "MOSAIC data : $MOSAIC_DATA_DIR/preprocessed"
[ -n "$EXTRA_STR" ] && echo "Options     : $EXTRA_STR"
echo

# Main loop 
OK=""; CHECK=""; EXCLUDED=""; MISMATCH=""; MISSING=""; MULTIPLE=""; FAILED=""
i=0
for pid in $PIDS; do 
    i=$((i + 1))
    rm -rf "$OUT_DIR/participants/$pid"   # no results from earlier runs - clean slate
    # file name must contain the ID, not followed by another digit (so sub-gutslei001 does not match sub-gutslei0010)
    pid_re="$(printf "%s" "$pid" | sed 's/[.[\*^$]/\\&/g')" # regex safe version of the iD
    # finds transcripts whose filename contains the ID
    matches="$(printf "%s\n" "$ALL_TXT" | while IFS= read -r f; do   
                   [ -n "$f" ] && basename "$f" | grep -Eiq "${pid_re}([^0-9]|$)" && echo "$f" 
               done)"
    n=$(printf "%s\n" "$matches" | grep -c . || true) 
    # no transcript found for pid
    if [ "$n" -eq 0 ]; then
        printf "[%3d/%d] %-18s MISSING (no transcript found)\n" "$i" "$N_LISTED" "$pid"
        MISSING="$MISSING $pid"; continue
    # more than 1 transcript
    elif [ "$n" -gt 1 ]; then
        printf "[%3d/%d] %-18s SKIPPED (%d transcripts match):\n" "$i" "$N_LISTED" "$pid" "$n"
        printf "%s\n" "$matches" | sed 's/^/            /'
        MULTIPLE="$MULTIPLE $pid"; continue
    fi
    
    # Processing the transcript
    file="$matches"
    printf "%s\t%s\n" "$pid" "$file" >> "$MAPPING"
    ln -s "$file" "$SELECTED/$pid.txt" 2>/dev/null || cp "$file" "$SELECTED/$pid.txt"
    # runs preprocessing transcript in single-file mode with expected ID
    "$PYTHON" "$PREPARE" --input-file "$file" --participant "$pid" \
        --data-dir "$MOSAIC_DATA_DIR" --out-dir "$OUT_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$LOGS/$pid.log" 2>&1
    code=$?   # exit code of last command
    last="$(grep -E "^$pid_re: " "$LOGS/$pid.log" | tail -1 | cut -d'|' -f1-3)"  # takes result line from preprocessing.py main()
    # sorts result by exit code
    case $code in
        0) if printf "%s" "$last" | grep -q ": ok |"; then status="OK"; OK="$OK $pid"
           else status="CHECK"; CHECK="$CHECK $pid"; fi ;;
        10) status="EXCLUDED";  EXCLUDED="$EXCLUDED $pid" ;;
        11) status="ID MISMATCH"; MISMATCH="$MISMATCH $pid" ;;
        *) status="FAILED (exit $code)"; FAILED="$FAILED $pid" ;;
    esac
    # prints results
    printf "[%3d/%d] %-18s %-12s %s\n" "$i" "$N_LISTED" "$pid" "$status" "${last#*: }"
    [ "$status" != "OK" ] && echo "            see $LOGS/$pid.log and participants/$pid/${pid}_segmentation.csv"
done
 
# transcripts not used (not listed, or skipped because several matched)
UNUSED=""
while IFS= read -r f; do
    [ -z "$f" ] && continue
    grep -Fqx "$f" <(cut -f2 "$MAPPING") || UNUSED="$UNUSED
    $f"
done <<< "$ALL_TXT"
 
# ---------------------------------------------------------------- combine + QC
echo
echo "== Combining participant folders into MOSAIC datasets"
# running preprocessing.py just to combine the participant files
"$PYTHON" "$PREPARE" --combine --participants-file "$PARTICIPANTS_FILE" \
    --data-dir "$MOSAIC_DATA_DIR" --out-dir "$OUT_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$LOGS/_combine.log"
COMBINE_CODE=${PIPESTATUS[0]}
 
echo
echo "== Quality checks"
if ls "$SELECTED"/*.txt >/dev/null 2>&1; then
    # running quality_control.py
    "$PYTHON" "$QC" --input-dir "$SELECTED" --out-dir "$OUT_DIR/qc" \
        --per-participant-dir "$OUT_DIR/participants" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$LOGS/_qc.log" 2>&1
    QC_CODE=$?
    [ $QC_CODE -eq 0 ] && echo "QC report: $OUT_DIR/qc/qc_summary.txt" \
                       || echo "QC FAILED (exit $QC_CODE) - see $LOGS/_qc.log"
else
    QC_CODE=1; echo "No transcripts selected - QC skipped"
fi
 
# ---------------------------------------------------------------- summary
count() { printf "%s" "$1" | wc -w | tr -d ' '; }
{
    echo "GUTSLEI preprocessing run - $(date '+%Y-%m-%d %H:%M')"
    echo "transcripts: $TRANSCRIPTS_DIR"
    echo "participants file: $PARTICIPANTS_FILE ($N_LISTED listed)"
    echo "options: ${EXTRA_STR:-(defaults)}"
    echo
    echo "processed OK        : $(count "$OK")"
    echo "processed, CHECK    : $(count "$CHECK") $CHECK   (moment structure; see segmentation report)"
    echo "excluded            : $(count "$EXCLUDED") $EXCLUDED"
    echo "speaker ID mismatch : $(count "$MISMATCH") $MISMATCH"
    echo "failed (error)      : $(count "$FAILED") $FAILED"
    echo "no transcript       : $(count "$MISSING") $MISSING"
    echo "several transcripts : $(count "$MULTIPLE") $MULTIPLE"
    [ -n "$UNUSED" ] && echo "transcripts not used (not listed, or duplicates):$UNUSED"
    echo
    echo "combine: $([ "$COMBINE_CODE" -eq 0 ] && echo ok || echo "FAILED (see logs/_combine.log)")"
    echo "QC     : $([ "$QC_CODE" -eq 0 ] && echo "ok -> qc/qc_summary.txt, qc/qc_flags.csv" || echo FAILED)"
    echo
    echo "Participant folders: $OUT_DIR/participants/<ID>/"
    echo "MOSAIC datasets    : $MOSAIC_DATA_DIR/preprocessed/gutslei_answers_{full,moment1,moment2,moment3}_preprocessed.csv (answer units, recommended)"
    echo "                     $MOSAIC_DATA_DIR/preprocessed/gutslei_{full,moment1,moment2,moment3}_preprocessed.csv (one document per participant)"
} > "$SUMMARY"
echo
cat "$SUMMARY"
 
if [ -n "$CHECK$EXCLUDED$MISMATCH$FAILED$MISSING$MULTIPLE" ] || [ "$COMBINE_CODE" -ne 0 ] || [ "$QC_CODE" -ne 0 ]; then
    exit 1
fi
exit 0