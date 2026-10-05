# coding=utf-8
# ============================================================================================================================================================
# title           : gutslei.py
# description     : MOSAIC dataset configuration for the interviews of GUTSLEI project
#                   - shared settings for all GUTSLEI datasets (extra config modules for specific datasets types such as for answer-level dataset and participant-
#                   level datasets)
#                   - Embedding and clustering: 
#                       preprocessing done pre answers embedded with transformer model and clustered (UMAP+HDBSCAN) - text not lowercased, lemmatised or 
#                   stripped of stopwords (preprocessing done previously by preprocessing.py);
#                   - Topic representation: 
#                       keywords describing each topic are computed (CountVectorizer + c-TF-IDF) - lowercasing, stopwords removal, n-grams and document-
#                   frequency thresholds act here
#                   - Stopwords: NLTK english stopwords (without negations), FILLERS (hesitations, discourse markers), HEDGES_AND_VAGUE (frequent, low information, 
#                   spontaneous speech), INTERVIEW_FRAME (referring to interview or film-viewing itself)
#                       - words describing experience and film content are not removed
#                       - OPTIONAL_HIGH_FREQ lists experience verbs that may appear in nearly every topic - enable it if qc_words_content.csv shows they are used by 
#                       most participants and they crowd out more specific keywords
#                   - Lemmatisation (optional): inflected forms split word counts, which weakens keywords (important for small datasets)
#                       - can be done to replace topic representation after fitting, without refitting mordel (requires spacy, download en_core_web_sm)
# usage           : (lemmatisation)
#                   from mosaic.configs.gutslei import config, lemma_vectorizer
#                   topic_model.update_topics(docs, vectorizer_model=lemma_vectorizer())
#
# dependencies    : Python  : >= 3.10 ; pandas and nltk installation
# author          : Ayres Ribeiro, Francisca (f.ayres-ribeiro@nin.knaw.nl) (with support from Claude)
# date            : 2026-10-05
# version         : 1.0
# ============================================================================================================================================================

from nltk.corpus import stopwords

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 0 - Stopword sets (representation only; see header)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FILLERS = {
    "um", "uh", "uhm", "umm", "erm", "er", "hmm", "mm", "mhm", "oh", "ah", "eh",
    "yeah", "yes", "okay", "ok", "like", "well", "actually", "basically", "guess",
    "know", "mean", "kind", "sort",                    
}
HEDGES_AND_VAGUE = {
    "really", "just", "maybe", "probably", "quite", "bit", "little", "lot",
    "thing", "things", "something", "stuff", "also", "even", "much", "way",
    "think", "say", "said", "get", "got", "go", "going", "gonna", "would",
    "could", "one", "still", "anything", "everything", "sometimes",
}
INTERVIEW_FRAME = {
    "movie", "film", "scene", "scenes", "moment", "moments", "part", "parts",
    "watching", "watched", "watch", "video", "screen", "time",
    "first", "second", "third", "last", "final", "beginning", "end",
}
# Lists experience verbs that may appear in nearly every topic - enable it if qc_words_content.csv shows they are used by most participants and they crowd out more specific keywords
OPTIONAL_HIGH_FREQ = {
    "feel", "felt", "feeling", "feelings",
}
USE_OPTIONAL_HIGH_FREQ = False

# TODO - can add extra stopwords based on the quality control, e.g. words used by large % of participants that only reflect film or interview
# (decide BEFORE the Optuna search; after any change, delete results/optuna/<dataset> before searching again)

# Negations are KEPT in the topic keywords (removed from NLTK's stopword list): absence and contrast are part of how an experience is described ("didn feel", "not tense", "no pain" appear as two-word keywords). 
NEGATIONS = {
    "no", "nor", "not", "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven",
    "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't",
    "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
}
KEEP_NEGATIONS = True  # set KEEP_NEGATIONS = False to use NLTK's full list again.

# Hesitations  removed from the TEXT itself (before embedding) - for MOSAIC's own preprocessing. NOT USED FOR GUTSLEI. 
# They carry no meaning, so removing them is safe. Do not add 'like' here: it is often a comparison ("like a heaviness in my stomach").
HESITATIONS = ["um", "uh", "uhm", "umm", "erm", "hmm", "mm", "mhm", "mm-hmm"]

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 1 - Config module for GUTSLEI dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class GutsleiConfig:
    def __init__(self, name="gutslei", split_sentences=True):
        # --- Stopwords (topic representation only)
        self.stop_words = set(stopwords.words("english")) # NLTK English stopword
        if KEEP_NEGATIONS:
            self.stop_words -= NEGATIONS        # negations stay in the topic keywords
        self.reduced_custom_stopwords = FILLERS | HEDGES_AND_VAGUE | INTERVIEW_FRAME
        if USE_OPTIONAL_HIGH_FREQ:
            self.reduced_custom_stopwords |= OPTIONAL_HIGH_FREQ
        # final list combining all - MOSAIC reads for its vectorizer
        self.extended_stop_words = self.stop_words.union(self.reduced_custom_stopwords)

        # --- Dataset (label only: MOSAIC finfd config by filename)
        self.name = name
        self.transformer_model = "Qwen/Qwen3-Embedding-0.6B" # MOSAIC default model

        # --- Text cleaning applied by MOSAIC's basic preprocessing (acts on text to be embedded) - NOT FOR GUTSLEI
        # (optuna_search.py ignores these if sentence splitting left out for answer-level datasets)
        self.words_to_remove = HESITATIONS                       # already removed by preprocessing.py; unused
        self.patterns_to_remove = [r"\([^)]*\)", r"\[[^\]]*\]"]  # already removed by preprocessing.py; unused
        self.split_sentences = split_sentences                   # False for answer-level datasets (gutslei_answers_*)
        self.min_word_count = 3                                  # already accounted in preprocessing.py; unused

        # --- Vectorizer (topic representation only)
        self.ngram_range = (1, 2)       # topic keywords single word or two-word
        self.max_df = 0.95              # words in more than 95% of texts ignored
        self.min_df = 2                 # words in fewer than 2 texts ignored
        self.top_n_words = 15           # number of keywords to describe each topic

        # --- Optuna search space. Ranges suit the preliminary sample (16 interviews, a few hundred sentences); 
        # TODO - widen them for the full sample of 144 (e.g. min_cluster_size (10, 30)).
        self.search_space = {
            "n_components": (5, 10),     # how many dimensions to reduce to (UMAP)
            "n_neighbors": (5, 15),      # how many nearby texts each text considers (UMAP)
            "min_dist": (0.0, 0.05),     # how tightly points can be packed (UMAP)
            "min_cluster_size": (5, 10), # smallest group that counts as topic (HDBSCAN)
            "min_samples": (3, 5),       # how strict clustering is
        }

    # Interface expected by MOSAIC scripts - default parameters, when no optuna_search.py
    def get_default_params(self, condition=None): # single condition for this study
        return {"n_neighbors": 10, "n_components": 5, "min_dist": 0.0,
                "min_cluster_size": 5, "min_samples": 3, "top_n_words": self.top_n_words}
    # Interface expected by MOSAIC scripts - alternative grid of values to try for MOSAIC's grid search (alternative to optuna_search.py)
    def get_params(self, condition=None, reduced=False):
        s = self.search_space
        if reduced: # small grid for quick test (2 combinations)
            return {"umap_params": {"n_components": [5], "n_neighbors": [10], "min_dist": [0.0]},
                    "hdbscan_params": {"min_cluster_size": [5, 8], "min_samples": [3]}}
        return {"umap_params": {"n_components": list(range(s["n_components"][0], s["n_components"][1] + 1, 5)),
                                "n_neighbors": list(range(s["n_neighbors"][0], s["n_neighbors"][1] + 1, 5)),
                                "min_dist": [0.0, 0.05]},
                "hdbscan_params": {"min_cluster_size": list(range(s["min_cluster_size"][0], s["min_cluster_size"][1] + 1)),
                                   "min_samples": list(range(s["min_samples"][0], s["min_samples"][1] + 1))}}


def lemma_vectorizer(cfg=None):
    """CountVectorizer that lemmatises with spaCy (topic representation only)."""
    import spacy
    from sklearn.feature_extraction.text import CountVectorizer

    cfg = cfg or config
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # lemmatise the stopword list too, so e.g. 'felt' and 'feeling' are caught by 'feel'
    stop = {t.lemma_.lower() for w in cfg.extended_stop_words for t in nlp(w)} | {
        w.lower() for w in cfg.extended_stop_words}

    def tokenize(text):
        # "n't" (from "didn't", "wasn't") is kept as "not", so negations survive lemmatisation
        return ["not" if t.lower_ in ("n't", "n’t") else t.lemma_.lower() for t in nlp(text)
                if (t.is_alpha or t.lower_ in ("n't", "n’t"))
                and ("not" if t.lower_ in ("n't", "n’t") else t.lemma_.lower()) not in stop
                and (len(t) > 1 or t.lower_ in ("n't", "n’t"))]

    return CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           ngram_range=cfg.ngram_range, min_df=cfg.min_df, max_df=cfg.max_df)


config = GutsleiConfig()