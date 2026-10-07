# Answer-level GUTSLEI dataset without short low-content answers AND without purely narrative answers
# (created by: analyse_gutslei.py content --drop-narrative). One answer per unit (run optuna_search.py WITHOUT --sentences).
from .gutslei import GutsleiConfig

config = GutsleiConfig(name="gutslei_answers_content_nonarrative", split_sentences=False)