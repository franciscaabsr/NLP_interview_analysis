# MOSAIC loads the config module named after the dataset; all GUTSLEI datasets share gutslei.py
# dataset where each row corresponds to all answers (of moment 1) of one participant - sentence-splitting used
from .gutslei import GutsleiConfig

config = GutsleiConfig(name="gutslei_moment1")