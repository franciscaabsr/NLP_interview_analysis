# Answer-level GUTSLEI dataset without purely narrative answers (sensitivity analysis;
# created by: analyse_gutslei.py filter). One answer per unit (run optuna_search.py WITHOUT --sentences).
from .gutslei import GutsleiConfig

config = GutsleiConfig(name="gutslei_answers_full_nonarrative", split_sentences=False)