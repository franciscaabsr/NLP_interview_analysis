# Setup guide: interview preprocessing + MOSAIC topic modelling

How to set up (or rebuild) the environment on the lab server, run the preprocessing
pipeline on the interview transcripts, and run MOSAIC with a local Llama model.
Follow the steps in order the first time; afterwards, see
[Every time you work on this](#every-time-you-work-on-this).

> **Keep this file up to date.** When the pipeline or the setup changes, update this file
> together with the scripts and commit them together.

**Contents**

- [Overview](#overview)
- [Step 0. Start a session and check the server](#step-0-start-a-session-and-check-the-server)
- [Step 1. MOSAIC in the right place](#step-1-mosaic-in-the-right-place)
- [Step 2. Create the Python environment](#step-2-create-the-python-environment)
- [Step 3. PyTorch for the GPU](#step-3-pytorch-for-the-gpu)
- [Step 4. llama-cpp-python (Llama on the GPU)](#step-4-llama-cpp-python-llama-on-the-gpu)
- [Step 5. MOSAIC's requirements](#step-5-mosaics-requirements)
- [Step 6. Language resources](#step-6-language-resources)
- [Step 7. Download the models](#step-7-download-the-models)
- [Step 8. Test everything](#step-8-test-everything)
- [Step 9. Put the pipeline in place](#step-9-put-the-pipeline-in-place)
- [Step 10. Preprocess and check](#step-10-preprocess-and-check)
- [Step 11. MOSAIC smoke test](#step-11-mosaic-smoke-test)
- [Step 12. Jupyter for MOSAIC's notebook](#step-12-jupyter-for-mosaics-notebook)
- [Step 13. Analysis commands](#step-13-analysis-commands)
- [Step 14. GitHub](#step-14-github)
- [Every time you work on this](#every-time-you-work-on-this)
- [Working on the shared server](#working-on-the-shared-server)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Project folder

On the lab server: `PROJECT=/data00/GUTS/francisca/interview_preliminary_analysis/NLP_interview_analysis`
(adapt the path if you set this up elsewhere). The transcripts stay **outside** this folder.

```
NLP_interview_analysis/                 ← PROJECT
├── .gitignore, README.md, SETUP.md, requirements-lock.txt   (GitHub)
├── scripts/            ← preprocessing.py, run_preproc.sh, quality_control.py, analyse_gutslei.py   (GitHub)
├── mosaic_configs/     ← the 10 gutslei config files, master copy                               (GitHub)
├── participants.txt    ← one participant ID per line                                           (GitHub if allowed)
├── caches/             ← models, pip cache, temporary files                                    (never on GitHub)
└── MOSAIC/             ← MOSAIC's own git repository                                           (never on GitHub)
    ├── .mosaicvenv/              ← the Python environment
    ├── src/mosaic/configs/       ← MOSAIC's configs + links to ../../mosaic_configs/
    ├── DATA/preprocessed/        ← MOSAIC-ready datasets      (interview text!)
    ├── DATA/derivatives/         ← QC, audit, reports, logs   (interview text!)
    └── results/optuna/           ← MOSAIC's search results
```

Why this layout:

- **`DATA/` must be inside `MOSAIC/`.** MOSAIC's search script only looks there.
- **The four scripts stay together in `scripts/`.** The QC script imports `preprocessing.py` from its own folder.
- **The configs are linked into `MOSAIC/src/mosaic/configs/`.** MOSAIC loads them by dataset name from there.

### Three rules that avoid most problems

1. **Always activate the environment** before working (see [Every time you work on this](#every-time-you-work-on-this)).
2. **Never move the project or MOSAIC folder.** A Python environment cannot be moved. If it happens, recreate it (step 2) and reinstall (steps 3–5).
3. **Install with `python -m pip …`.** Never use plain `pip`, and never `sudo pip`.

### What MOSAIC uses, and what we use

- **MOSAIC's preprocessing (`preprocessing/preprocessing.py`) is not used.** `scripts/preprocessing.py` does all cleaning and builds answer-level units.
- **MOSAIC's Llama is local:** `llama-cpp-python` with a GGUF file, used for topic labels. The only non-local model in MOSAIC is Google Gemini (optional, in MOSAIC's own preprocessing), which we don't use.
- **Python version.** MOSAIC's `pyproject.toml` asks for Python ≥ 3.12, but the code also parses on 3.10/3.11. We install from `requirements.txt` and make `mosaic` importable with `PYTHONPATH` instead of `pip install -e .`.

---

## Step 0. Start a session and check the server

**0a. Use `tmux`**, so long steps keep running if the connection drops:

```bash
tmux ls                       # existing sessions?
tmux attach -t mosaic         # continue in an existing "mosaic" session
tmux new -s mosaic            # or create one
```

| Action | Keys / command |
|---|---|
| leave the session running | `Ctrl-b`, then `d` |
| new window inside the session | `Ctrl-b`, then `c` |
| switch windows | `Ctrl-b`, then the window number |
| delete a session (stops everything in it) | `tmux kill-session -t mosaic` |

**0b. Set the project path for this session:**

```bash
PROJECT=/data00/GUTS/francisca/interview_preliminary_analysis/NLP_interview_analysis
cd $PROJECT && ls
```

**0c. Check the server:**

```bash
python3 --version                  # 3.10 or newer
nvidia-smi                         # GPUs, free memory, "CUDA Version" (top right) = highest the DRIVER supports
nvcc --version                     # CUDA toolkit used for compiling (step 4)
nproc                              # number of CPU cores (for the thread limits in step 2)
df -h $PROJECT                     # "Avail": at least ~20 GB
gcc --version; g++ --version; make --version; cmake --version
python3 -c "import sysconfig, os; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"
```

On our server, all build tools are present and the last line prints `True`, so **no `sudo` is needed**.
If something is missing:

- **No compilers:** `sudo apt install build-essential`.
- **`False` on the last line:** `sudo apt install python3-dev`, or the version-specific package such as `python3.10-dev`.
- **Red Hat-type servers:** `sudo dnf groupinstall "Development Tools"` and `sudo dnf install python3-devel cmake`.

**Our server (storm):**

| Item | Value | Consequence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GB, compute capability 7.5 | Llama 8B (~6 GB) + embedding model (~1.5 GB) fit |
| driver | supports CUDA **12.5** (`nvidia-smi`) | PyTorch **cu126** works (step 3); prebuilt llama-cpp-python **cu125** works (step 4) |
| CUDA toolkit | `nvcc` **11.5**, Ubuntu package (`/usr/bin/nvcc`) | **too old to compile** llama-cpp-python for the GPU (step 4) |
| compilers | gcc 11.4, cmake 3.22, Python 3.10 | fine for everything else |

Check your GPU's compute capability with `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.

If downloads hang in later steps, ask IT for the proxy setting (`export https_proxy=…`).

---

## Step 1. MOSAIC in the right place

MOSAIC must be at `$PROJECT/MOSAIC`:

```bash
cd $PROJECT/MOSAIC 2>/dev/null && git log -1 --format='%h %cd' || echo "MOSAIC not here yet"
# if not there:
cd $PROJECT && git clone https://github.com/romybeaute/MOSAIC.git
```

To use exactly the MOSAIC version recorded in `README.md`:

```bash
git -C $PROJECT/MOSAIC checkout <commit from README.md>
```

**Where to run commands.**

- **Installing packages** (`python -m pip install <package>`, steps 3, 4, 6, 7): the folder doesn't matter, only that the environment is **active**. pip installs into the active environment.
- **Commands that use MOSAIC's files by a short path:** run them from `$PROJECT/MOSAIC`. These are `requirements.txt` (step 5), `src/mosaic/optuna_search.py` (step 11), Jupyter (step 12) and `DATA/…` (step 13). From another folder they fail with "No such file or directory".

---

## Step 2. Create the Python environment

**2a. Remove any moved or broken environment and create a new one in place:**

```bash
deactivate 2>/dev/null
cd $PROJECT/MOSAIC
pwd                                    # must be …/NLP_interview_analysis/MOSAIC
rm -rf .mosaicvenv
python3 -m venv .mosaicvenv
```

**2b. Store the settings in the environment.** They're applied at every activation.
Replace `8` with about half of `nproc`. **Run these lines only once.** Repeating them adds
duplicate lines; if that happens, delete the extra lines at the end of `.mosaicvenv/bin/activate`.

```bash
echo "export PROJECT=$PROJECT"                         >> .mosaicvenv/bin/activate
echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH"         >> .mosaicvenv/bin/activate
echo "export HF_HOME=$PROJECT/caches/hf_cache"         >> .mosaicvenv/bin/activate
echo "export PIP_CACHE_DIR=$PROJECT/caches/pip_cache"  >> .mosaicvenv/bin/activate
echo "export TMPDIR=$PROJECT/caches/tmp"               >> .mosaicvenv/bin/activate
echo "export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8" >> .mosaicvenv/bin/activate
mkdir -p $PROJECT/caches/{hf_cache,pip_cache,tmp}
```

| Setting | Purpose |
|---|---|
| `PROJECT` | the project path, available as `$PROJECT` in every session |
| `PYTHONPATH` | makes `mosaic` importable from `MOSAIC/src` (replaces `pip install -e .`) |
| `HF_HOME` | models are stored in and read from `caches/hf_cache`; MOSAIC's search runs offline and only looks here |
| `PIP_CACHE_DIR` | pip's download copies go to `caches/`, not the home folder |
| `TMPDIR` | compiling space for step 4 |
| `…_NUM_THREADS` | stops UMAP, HDBSCAN and numerical libraries from taking all CPU cores on the shared server |

**2c. Activate and update pip's tools:**

```bash
source .mosaicvenv/bin/activate
python -m pip install -U pip wheel setuptools
```

**Check.** All lines must show `…/NLP_interview_analysis/…`:

```bash
which python                       # …/MOSAIC/.mosaicvenv/bin/python
head -1 .mosaicvenv/bin/pip        # #!…/MOSAIC/.mosaicvenv/bin/python3
tail -6 .mosaicvenv/bin/activate   # the six settings, with full paths
echo $HF_HOME $OMP_NUM_THREADS
```

A leftover `src/mosaic.egg-info/` from an earlier `pip install -e .` is harmless, but it makes pip
print dependency warnings. Remove it with `rm -rf src/mosaic.egg-info`.

---

## Step 3. PyTorch for the GPU

*(Any folder; the environment must be active.)*

On pytorch.org → "Get Started", choose **Linux, Pip, Python**, and the CUDA version offered that is
closest to the driver's version. On our server (driver: CUDA 12.5) that is **12.6**. It works through
CUDA's minor-version compatibility within 12.x.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "
import torch
print('torch', torch.__version__, '| built for CUDA', torch.version.cuda, '| GPU available:', torch.cuda.is_available())
x = torch.randn(2000, 2000, device='cuda'); print('GPU computation ok:', round((x @ x).sum().item(), 2))"
```

**Expected:** `GPU available: True` and `GPU computation ok: …`.

**Messages you can ignore:**

- A warning *"Failed to initialize NumPy"*: NumPy comes in step 5.
- *"mosaic 0.1.0 requires …, which is not installed"*: those packages come in step 5, or remove the leftover `src/mosaic.egg-info`.

**Fallback if the test fails:**

```bash
python -m pip uninstall -y torch
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
```

**Effect on colleagues: none.** PyTorch (2–5 GB) goes into `.mosaicvenv` only. It brings its own
CUDA libraries, and doesn't touch the server's CUDA, the driver, or anyone else's Python.

---

## Step 4. llama-cpp-python (Llama on the GPU)

*(Any folder; the environment must be active.)*

MOSAIC's topic labeller (`PhenoLabeler`) runs Llama through this library. Install it **before**
MOSAIC's requirements, which would otherwise install a CPU-only version.

### 4a. Prebuilt GPU version (our server: use this)

On our server, compiling for the GPU is not possible: `nvcc` 11.5 fails with gcc 11 (see 4c). Use a
version already built for CUDA 12.5. It needs only the driver, not the compiler:

```bash
python -m pip uninstall -y llama-cpp-python           # in case a partial/CPU version is there
python -m pip install llama-cpp-python --only-binary=llama-cpp-python --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
python -c "import llama_cpp; print(llama_cpp.__version__, '| GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

- **`--only-binary=llama-cpp-python`:** pip takes only a ready-made version and never tries to compile.
- **Expected:** a version number and `GPU offload: True`.
- **The address:** these versions can lag behind the newest release. If the address changes, check the llama-cpp-python README.

**If the check fails with `libcudart.so.12` / `libcublas.so.12: cannot open shared object file`:** the
server only has CUDA 11 libraries. The CUDA 12 libraries PyTorch installed in the environment (step 3)
can be used instead. Store their location in the environment (once):

```bash
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cublas/lib:\$LD_LIBRARY_PATH" >> $PROJECT/MOSAIC/.mosaicvenv/bin/activate
source $PROJECT/MOSAIC/.mosaicvenv/bin/activate
python -c "import llama_cpp; print(llama_cpp.__version__, '| GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

**If no prebuilt version installs:** take the CPU version for now, which is enough for labelling
10–20 topics: `python -m pip install llama-cpp-python`.

### 4b. Compiling for the GPU (only on servers with a CUDA 12 toolkit)

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" \
    nice -n 10 python -m pip install llama-cpp-python --no-cache-dir
```

| Part | Purpose |
|---|---|
| `-DGGML_CUDA=on` | build with GPU support |
| `-DCMAKE_CUDA_ARCHITECTURES=75` | build only for your GPU (compute capability without the dot: 7.5 → `75`, 8.6 → `86`) |
| `CMAKE_BUILD_PARALLEL_LEVEL=8` | compile with at most 8 cores (courtesy on the shared server) |
| `nice -n 10` | lower priority, so colleagues' work goes first |

Installing a CUDA 12 toolkit on our server would need `sudo` (it goes into `/usr/local/cuda-12.x`,
next to the old one, without touching the driver). It changes the shared server, so agree on it with
whoever manages it first. It isn't needed for the preliminary analysis.

### 4c. If a build fails

The last lines (`ERROR: Failed building wheel for llama-cpp-python`) only say *that* it failed.
Capture the log to see why:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python --no-cache-dir -v 2>&1 | tee $PROJECT/caches/llama_build.log
grep -n -iE "error|unsupported|not found|required" $PROJECT/caches/llama_build.log | head -30
```

| Message in the log | Cause | Fix |
|---|---|---|
| `parameter packs not expanded with '...'` (in `std_function.h`) | `nvcc` 11.5 is incompatible with gcc 11's C++ headers (**our server**) | use the prebuilt version (4a), or a CUDA ≥ 11.6 toolkit |
| `Unsupported gpu architecture 'compute_…'` | the default build includes GPU generations this toolkit doesn't know | `-DCMAKE_CUDA_ARCHITECTURES=<your GPU>` (see 4b) |
| `unsupported GNU version` | `gcc` newer than the toolkit accepts | add `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` to `CMAKE_ARGS` |
| `No CMAKE_CUDA_COMPILER could be found` | `nvcc` not found | `export CUDACXX=/path/to/nvcc` |
| `CMake 3.… or higher is required` | `cmake` too old | `python -m pip install -U cmake` (inside the environment) |
| `No space left on device` | temporary space full | `df -h $PROJECT/caches/tmp` |

**Effect on the server (4a–4c).** Everything is installed **only inside `.mosaicvenv`**. Compiling
uses the system's compilers and CUDA toolkit but changes nothing on the system, and no `sudo` is
needed.

**After step 5,** run the `llama_cpp` check again: `requirements.txt` also lists `llama-cpp-python`.
pip normally keeps the installed version. If `GPU offload` turns `False`, repeat 4a.

Steps 5–7 don't depend on step 4, so continue with them while sorting out step 4 if needed.

---

## Step 5. MOSAIC's requirements

```bash
cd $PROJECT/MOSAIC                              # requirements.txt is here (or use -r $PROJECT/MOSAIC/requirements.txt)
python -m pip install -r requirements.txt
python -m pip install optuna langdetect python-dotenv pyyaml   # used by MOSAIC but missing from requirements.txt
python -m pip check                             # ideally "No broken requirements found."
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # still …+cu126 True?
```

- **The packages MOSAIC forgot.** `requirements.txt` doesn't list everything MOSAIC imports. `optuna` is used by the search script. `langdetect` and `python-dotenv` are used by MOSAIC's own preprocessing module, which `src/mosaic/__init__.py` loads whenever anything from `mosaic` is imported, so they're needed even though we don't use that preprocessing. `pyyaml` is usually installed already; listing it does no harm.
- **`bitsandbytes` errors** can be ignored. It's only used for experiments we don't run.
- **If the last line now says `False`**, another package replaced PyTorch: repeat step 3.
- **Don't run `pip install -e .`.** It fails below Python 3.12, and `PYTHONPATH` replaces it.

---

## Step 6. Language resources

*(Any folder; the environment must be active.)*

```bash
python -m nltk.downloader stopwords punkt punkt_tab     # small, stored in ~/nltk_data
python -m spacy download en_core_web_sm                 # only for the optional lemmatised keywords
```

---

## Step 7. Download the models

*(Any folder; the environment must be active, so that `HF_HOME` points to `caches/hf_cache`.)*

MOSAIC's search script never downloads anything (it runs offline), so this must happen first:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
huggingface-cli download NousResearch/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
du -sh $PROJECT/caches/hf_cache                         # ~6 GB
```

| Model | Size | Used for |
|---|---|---|
| Qwen3-Embedding-0.6B | ~1.2 GB | turning each answer into a vector (MOSAIC's default embedding model) |
| Meta-Llama-3-8B-Instruct, Q4_K_M | ~4.9 GB | topic labels (the model MOSAIC's notebooks use) |

- **Command name:** in newer versions of the Hugging Face tools, the command is `hf download …`.
- **Login:** if a login is requested, run `huggingface-cli login`.

**Other Llama options** (GGUF, all run with `llama-cpp-python`):

| Model | Size (Q4_K_M) | GPU memory | Notes |
|---|---|---|---|
| **Meta-Llama-3-8B-Instruct** | ~4.9 GB | ~6–8 GB | MOSAIC's default, recommended |
| Meta-Llama-3.1-8B-Instruct | ~4.9 GB | ~6–8 GB | used in another MOSAIC script; safe alternative |
| Llama-3.2-3B-Instruct | ~2 GB | ~3–4 GB | only if GPU memory is short; weaker labels |
| Llama-3.1/3.3-70B-Instruct | ~40 GB | ≥ 48 GB | better labels; overkill for preliminary results |

---

## Step 8. Test everything

Choose a free GPU first. Check `nvidia-smi` and pick one with little memory in use. This is set
per session, not stored, because which GPU is free changes from day to day.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0          # number of a free GPU

python -c "import mosaic.model, bertopic, optuna, llama_cpp; print('imports ok')"

python -c "
from sentence_transformers import SentenceTransformer
print(SentenceTransformer('Qwen/Qwen3-Embedding-0.6B').encode(['my chest got tight']).shape)"

python -c "
from huggingface_hub import hf_hub_download; from llama_cpp import Llama
p = hf_hub_download('NousResearch/Meta-Llama-3-8B-Instruct-GGUF', 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
llm = Llama(model_path=p, n_gpu_layers=-1, n_ctx=4096)
print(llm('Give a 3-word title for: my chest got tight; a knot in my stomach. Title:', max_tokens=12)['choices'][0]['text'])"
```

**Expected output:**

- `imports ok`;
- `(1, 1024)`;
- a short title (e.g. "Tense Moment Approaches"), possibly followed by more text. The raw test prompt has no stop signal, so the model writes until the 12-token limit; MOSAIC's labeller uses a chat prompt with stop tokens.
- Lines mentioning `CUDA0` (e.g. `CUDA0 compute buffer`) show that the GPU is used. `CUDA Graph … reused` lines are information only.

**Speed check.** The first run on a GPU is slow (on our server: ~30 s), because the driver translates the
prebuilt library for this GPU once and stores the result in `~/.nv/ComputeCache`. Run the test again,
silent and timed:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv    # ~7 GB free on your GPU?
python -c "
from huggingface_hub import hf_hub_download; from llama_cpp import Llama
p = hf_hub_download('NousResearch/Meta-Llama-3-8B-Instruct-GGUF', 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
llm = Llama(model_path=p, n_gpu_layers=-1, n_ctx=4096, verbose=False)
import time; t = time.time()
print(llm('Give a 3-word title for: my chest got tight; a knot in my stomach. Title:', max_tokens=12)['choices'][0]['text'])
print(f'{time.time() - t:.1f} s')"
```

**`n_gpu_layers`** sets how many of the model's 33 layers run on the GPU:

- **`-1`, all 33:** fastest, about 1–3 s for this test on an RTX 2080 Ti. It needs ~6 GB of free GPU memory, which fits on an 11 GB card that isn't busy.
- **A lower value such as `20`:** use it if the GPU is busy or reports "out of memory". The rest runs on the CPU, which is slower, but still fine for labelling 10–20 topics.
- **`verbose=False`:** hides llama.cpp's information messages.

---

## Step 9. Put the pipeline in place

**9a. Scripts** (the four must stay in one folder):

```bash
mkdir -p $PROJECT/scripts
chmod +x $PROJECT/scripts/run_preproc.sh
ls $PROJECT/scripts        # preprocessing.py  run_preproc.sh  quality_control.py  analyse_gutslei.py
```

**9b. Configs.** Keep the master copy in `mosaic_configs/` and link it into MOSAIC:

```bash
rm -f $PROJECT/MOSAIC/src/mosaic/configs/gutslei*.py                  # remove old copies, if any
ln -s $PROJECT/mosaic_configs/gutslei*.py $PROJECT/MOSAIC/src/mosaic/configs/
ls -l $PROJECT/MOSAIC/src/mosaic/configs/ | grep gutslei              # 10 lines, each "-> …/mosaic_configs/…"
```

**9c. Keep MOSAIC's own `git status` clean** with git's local exclude file, which doesn't change MOSAIC:

```bash
cat >> $PROJECT/MOSAIC/.git/info/exclude <<'EOF'
*.egg-info/
src/mosaic/configs/gutslei*.py
DATA/derivatives/
DATA/preprocessed/
results/
EOF
git -C $PROJECT/MOSAIC status --short        # empty
```

**9d. Participant list:** `$PROJECT/participants.txt`, one ID per line (e.g. `sub-gutslei0169`).
Blank lines, `#` comments and a header are ignored; a CSV first column also works.

---

## Step 10. Preprocess and check

```bash
$PROJECT/scripts/run_preproc.sh
```

The default paths are set at the top of `run_preproc.sh` (`TRANSCRIPTS_DIR`, `PARTICIPANTS_FILE`,
`MOSAIC_DATA_DIR`), so on our server no options are needed. Elsewhere, or for other folders, give them
explicitly (or edit the defaults):

```bash
$PROJECT/scripts/run_preproc.sh -i /path/to/the/transcripts -l $PROJECT/participants.txt -m $PROJECT/MOSAIC/DATA
```

| Option | Meaning |
|---|---|
| `-i` | transcripts folder; only files matching `*_task-emtint.txt` are read (e.g. `sub-gutslei0169_ses-01a_task-emtint.txt`); another naming: `-g '<pattern>'` |
| `-l` | participant list |
| `-m` | MOSAIC's `DATA` folder (MOSAIC only looks there) |
| `-- <options>` | passed to the Python scripts, e.g. `-- --min-words 4`; a mistyped option stops the run immediately |
| `-h` | help (the script's header) |

**What you see:** one line per participant (OK / CHECK / EXCLUDED / ID MISMATCH / FAILED / MISSING / SKIPPED), then a summary.

**Check, in this order:**

1. `MOSAIC/DATA/derivatives/run_summary.txt`: what happened to each participant.
2. `MOSAIC/DATA/derivatives/qc/qc_summary.txt`: overview of all checks.
3. `MOSAIC/DATA/derivatives/qc/qc_flags.csv`: every issue, ERRORs first.
4. `MOSAIC/DATA/derivatives/qc/qc_moments.csv`: `first_answer`, moment start times and scenes, per interviewer.
5. `MOSAIC/DATA/derivatives/participants/<ID>/<ID>_full.txt`: compare a few participants' cleaned answers with their transcripts.

Fix ERRORs (add wording to a pattern in `preprocessing.py`, or correct a transcript) and rerun.
Each run replaces the previous results.

**Result:** `MOSAIC/DATA/preprocessed/gutslei_answers_full_preprocessed.csv`, one row per answer, text column `cleaned_reflection`.

---

## Step 11. MOSAIC smoke test

```bash
cd $PROJECT/MOSAIC
export CUDA_VISIBLE_DEVICES=0                     # a free GPU
python src/mosaic/optuna_search.py --dataset gutslei_answers_full --use-config --n_trials 3
rm -rf results/optuna/gutslei_answers_full
```

- **What the command does:** it loads the dataset and its config (`--use-config`), keeps answers whole (no `--sentences`), and runs 3 trials.
- **Why delete the results:** MOSAIC **continues** an earlier search of the same dataset. Delete its results folder after the smoke test, and whenever the data or settings change, so old trials never mix with new ones.

The real search is the same command without `--n_trials 3` (default 100 trials). Run it in `tmux`,
and tell colleagues if it will occupy the machine for a while.

---

## Step 12. Jupyter for MOSAIC's notebook

On the server (in `tmux`, environment active):

```bash
cd $PROJECT/MOSAIC
which jupyter                                     # must point into .mosaicvenv
export CUDA_VISIBLE_DEVICES=0                     # a free GPU
jupyter lab --no-browser --port 8888
```

On your laptop:

```bash
ssh -L 8888:localhost:8888 <user>@<server-address>
```

Then open the `http://localhost:8888/…?token=…` link printed by Jupyter.

**In `notebooks/1_Run_pipeline/MOSAIC_pipeline.ipynb`:**

- **Data path:** `DATA/preprocessed/gutslei_answers_full_preprocessed.csv`.
- **Sentence splitting:** remove the `split_sentences(...)` line, so answers stay whole.
- **Import error:** if `from src.llama_CPP_custom import *` fails, use `from mosaic.llama_CPP_custom import *`.

**When done:** *Kernel → Shut Down Kernel*, then stop Jupyter with `Ctrl-c` twice. An idle
notebook keeps its GPU memory reserved.

---

## Step 13. Analysis commands

Run them from the MOSAIC folder, because they look for `DATA/…` there:

```bash
cd $PROJECT/MOSAIC
python $PROJECT/scripts/analyse_gutslei.py sample --n 80 --out validation          # blind hand-coding sample
python $PROJECT/scripts/analyse_gutslei.py agreement --coded validation/sample_to_code.csv --key validation/sample_key.csv
python $PROJECT/scripts/analyse_gutslei.py filter                                  # dataset without purely narrative answers
python $PROJECT/scripts/analyse_gutslei.py topics --topics topics.csv --topic-info topic_info.csv --out topics_table.csv
```

- **Row order:** MOSAIC returns topics in the order of the dataset's rows, so don't edit the dataset between modelling and `topics`.
- **These outputs contain interview text.** Keep them inside `MOSAIC/`, which is never on GitHub.

---

## Step 14. GitHub

**14a. `.gitignore`** (in `$PROJECT`):

```gitignore
# MOSAIC: separate repository, plus its environment, data and results inside it
MOSAIC/
# downloaded models, pip cache, temporary files
caches/
# anything derived from the interviews - never on GitHub
**/DATA/
**/derivatives/
**/preprocessed/
*_preprocessed.csv
validation/
topics*.csv
*.db
# Python, Jupyter and system clutter
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
.mosaicvenv/
.DS_Store
```

If the data management plan treats participant IDs as personal data, add `participants.txt` too.

**14b. Record the MOSAIC version and package versions.** Repeat this after updating MOSAIC or
packages, and before reporting results:

```bash
cd $PROJECT
cat >> README.md <<EOF

## MOSAIC version

- Repository: $(git -C MOSAIC remote get-url origin)
- Branch: $(git -C MOSAIC rev-parse --abbrev-ref HEAD)
- Commit: $(git -C MOSAIC rev-parse HEAD)
- Commit date: $(git -C MOSAIC log -1 --format='%cd' --date=short)
- Python: $(python --version 2>&1)
- Recorded on: $(date +%Y-%m-%d)
EOF
git -C MOSAIC status --short --untracked-files=no       # empty = MOSAIC's code unchanged
python -m pip freeze > requirements-lock.txt            # exact package versions
```

**14c. Commit and push** (to a **private** repository):

```bash
git add -n .                       # dry run: check that NO data, MOSAIC or caches appear
git add .gitignore README.md SETUP.md requirements-lock.txt scripts/ mosaic_configs/
git status
git commit -m "<what changed>"
git push
```

First time only:

- run `git init` first;
- set your identity: `git config --global user.name "…"` and `git config --global user.email "…"`;
- create an empty private repository on GitHub;
- connect it: `git remote add origin git@github.com:<account>/<repo>.git`, then `git branch -M main` and `git push -u origin main`;
- pushing over SSH needs an SSH key on the server, added to GitHub; alternatively use HTTPS with a personal access token.

**Before every push:**

- **`git status`** should list only scripts, configs and documentation.
- **`git check-ignore -v <file>`** confirms a file is ignored; empty output means it is *not*.
- **Something private committed by mistake:** remove it with `git rm --cached <file>` *before* pushing.

---

## Every time you work on this

### Start of the day (5 min)

```bash
# 1. session: continue yesterday's, or start a new one
tmux attach -t mosaic || tmux new -s mosaic

# 2. environment: sets PROJECT, PYTHONPATH, HF_HOME, caches and thread limits (step 2)
source /data00/GUTS/francisca/interview_preliminary_analysis/NLP_interview_analysis/MOSAIC/.mosaicvenv/bin/activate

# 3. check that everything is in place
which python                                   # …/MOSAIC/.mosaicvenv/bin/python
echo $PROJECT $HF_HOME                         # both set (empty = environment not active)
python -c "import mosaic, bertopic, torch; print('environment ok | GPU:', torch.cuda.is_available())"

# 4. project state: anything left uncommitted yesterday?
cd $PROJECT && git status --short

# 5. before GPU work only: pick a free GPU (little memory in use)
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=0                  # number of the free GPU

# 6. work from the MOSAIC folder (MOSAIC commands, Jupyter, analysis)
cd $PROJECT/MOSAIC
```

The environment, `CUDA_VISIBLE_DEVICES` and `cd` apply to **one terminal**: repeat steps 2, 5 and 6
in every new `tmux` window (`Ctrl-b`, then `c`).

### End of the day (5 min)

```bash
# Jupyter: Kernel -> Shut Down Kernel for each notebook, then Ctrl-c twice in its window (frees GPU memory)
nvidia-smi                                     # none of your processes should still hold GPU memory
cd $PROJECT && git status                      # commit scripts/configs/docs you changed (never data)
git add scripts/ mosaic_configs/ SETUP.md README.md && git commit -m "<what changed>" && git push
# leave tmux running with Ctrl-b, then d (long runs continue)
```

---

## Working on the shared server

| Resource | Good practice |
|---|---|
| installing | only inside `.mosaicvenv`, with `python -m pip`; never `sudo pip` |
| compiling (step 4b) | `nice -n 10` and `CMAKE_BUILD_PARALLEL_LEVEL`; on our server the prebuilt version (4a) needs no compiling |
| GPU choice | check `nvidia-smi`; set `CUDA_VISIBLE_DEVICES` to a free GPU |
| GPU memory | shut down notebook kernels and finished processes; PyTorch keeps memory reserved until the process ends |
| CPU cores | the thread limits from step 2 are on by default |
| long runs | in `tmux`; tell colleagues, or use the lab's scheduler if there is one; lab rules take precedence |
| disk | everything lives in the project folder on `/data00`, ~15–20 GB in total |

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `bad interpreter: No such file or directory` | the environment was moved: redo step 2 (delete and recreate), then steps 3–5 |
| `mosaic 0.1.0 requires …, which is not installed` | harmless before step 5; or remove `MOSAIC/src/mosaic.egg-info` |
| `Failed to initialize NumPy` (PyTorch warning) | harmless before step 5 (NumPy is installed there) |
| downloads hang | proxy: ask IT, then `export https_proxy=…` |
| MOSAIC: model not found | environment not active (different `HF_HOME`), or step 7 skipped |
| `torch.cuda.is_available()` is `False` | repeat step 3 (fallback: torch 2.6.0 cu124) |
| step 4: `Failed building wheel for llama-cpp-python` | see step 4c; on our server (`nvcc` 11.5 + gcc 11): use the prebuilt cu125 version (4a) |
| `libcudart.so.12` / `libcublas.so.12: cannot open shared object file` | prebuilt Llama version needs CUDA 12 libraries: the `LD_LIBRARY_PATH` line in step 4a |
| `llama_supports_gpu_offload()` is `False` | a CPU-only version is installed (e.g. by step 5): repeat step 4a |
| GPU out of memory | another GPU via `CUDA_VISIBLE_DEVICES`; lower `n_gpu_layers`; close old notebooks |
| Llama very slow (first run ~30 s) | normal on the first run (one-time GPU translation); if still slow: `n_gpu_layers=-1` and check `nvidia-smi` for other users of the GPU |
| Optuna results look odd after changes | old trials mixed in: `rm -rf results/optuna/gutslei_answers_full` |
| `Could not open requirements file` / `No such file or directory` | run from `$PROJECT/MOSAIC` (steps 5, 11–13) or give the full path |
| `No module named 'langdetect'` / `'dotenv'` / `'optuna'` | MOSAIC imports packages missing from its requirements: `python -m pip install optuna langdetect python-dotenv pyyaml` (step 5) |
| warning that `google.generativeai` is deprecated | harmless: part of MOSAIC's optional Gemini preprocessing, which we don't use |
| `import mosaic` fails | environment not active, or wrong `PYTHONPATH` line in `.mosaicvenv/bin/activate` |
| `run_preproc.sh`: "Unknown option" | typo in an option after `--` (see `--help`) |
| participant SKIPPED (several transcripts) | more than one file matches the ID and the pattern: adjust `-g` or remove the extra file |
| participant ID MISMATCH | the speaker label in the transcript differs from the ID in the file name |
| an installation fails because of the Python version | Python 3.12 via `uv` (no sudo): `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then `uv python install 3.12`, then `uv venv --python 3.12 .mosaicvenv`, then redo step 2b onwards |