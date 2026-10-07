# Answer-level GUTSLEI dataset without short low-content answers (created by: analyse_gutslei.py content).
# One answer per unit (run optuna_search.py WITHOUT --sentences).
from .gutslei import GutsleiConfig

config = GutsleiConfig(name="gutslei_answers_content", split_sentences=False)