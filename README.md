# Ranking reliability of classical learners on small tabular tasks

Code and results for a study of how reliably train/test splits rank nine classical learners
(logistic regression, k-NN, SVM, random forest, extra trees, histogram gradient boosting,
XGBoost, LightGBM, MLP) on 37 small public classification tasks (100 stratified 70/30 splits each).

## Files
- `build_datasets.py`, `fetch_rd.py`, `final_build.py`: build the task set from public data (Rdatasets, PyDataset, scikit-learn); `manifest_final.json` lists the tasks.
- `run_exp.py`: fits the nine learners on every task and split; writes `raw_results.csv` (task, rep, model, acc, n_test, sec; 35,100 rows).
- `analyze_exp.py`, `extra.py`, `adaptive.py`: reliability analysis (Kendall tau, top-learner agreement, pairwise reversals, aggregation across datasets, sequential stopping).
- Remaining `.csv` and `.txt` files: result tables and analysis logs used in the paper.

## Reproduce
`python run_exp.py` (needs scikit-learn, xgboost, lightgbm), then `python analyze_exp.py`, `python extra.py`, `python adaptive.py`.
The raw Rdatasets files are downloaded by `fetch_rd.py` from https://github.com/vincentarelbundock/Rdatasets.
