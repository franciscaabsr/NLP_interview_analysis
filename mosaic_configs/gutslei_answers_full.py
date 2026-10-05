# Answer-level GUTSLEI dataset: one answer per unit (run optuna_search.py WITHOUT --sentences)
from .gutslei import GutsleiConfig

config = GutsleiConfig(name="gutslei_answers_full", split_sentences=False)