"""Final ensemble configuration (chosen on OOF P@R>=0.70, see README_SOLUTION.md)."""
HIST_DAYS = 7          # labelled history length for item bot-pool features
SEEDS = (0, 1, 2, 3, 4)
MEMBERS = [
    dict(name="catboost", model="cat", params=None, weight=0.6),
    dict(name="extratrees", model="et", params=dict(min_samples_leaf=3), weight=0.2),
    dict(name="xgboost", model="xgb", params=None, weight=0.2),
]
