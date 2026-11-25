"""Fixed, rule-based construction of classification tasks. Rule written BEFORE any model was run.

Source: R datasets bundled in the `pydataset` package, plus four scikit-learn toy datasets.
Rule R1: 200 <= n <= 20000 rows. Target = the only column that is categorical (object/category/bool) or an integer
         with 2..8 distinct values. If there are several candidates or none, the dataset is skipped (target ambiguous).
Rule R2: features = all other columns; drop ID-like text columns (>50 distinct values and >50% unique); drop columns
         with >30% missing; median/mode imputation; one-hot for categoricals with <=20 levels, drop others.
Rule R3: keep classes with >=30 rows; need >=2 classes left; need >=4 feature columns after encoding.
Rule R4: exclude if majority class >= 95% of rows, or any single-feature depth-1 stump reaches >=99% accuracy (leak/trivial).
"""
import json, warnings, sys
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")
from pydataset import data as pdata

def prep(df, name):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    n = len(df)
    if not (200 <= n <= 20000): return None, "size"
    cands = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype) == "category" or s.dtype == bool:
            if s.nunique(dropna=True) <= 10 and s.nunique() >= 2: cands.append(c)
        elif pd.api.types.is_integer_dtype(s) and 2 <= s.nunique() <= 8:
            cands.append(c)
    if len(cands) != 1: return None, f"target_ambiguous({len(cands)})"
    t = cands[0]
    y = df[t]; X = df.drop(columns=[t])
    keep = []
    for c in X.columns:
        s = X[c]
        if s.isna().mean() > 0.30: continue
        if s.dtype == object:
            if s.nunique() > 50 and s.nunique() / max(1, s.notna().sum()) > 0.5: continue
        keep.append(c)
    X = X[keep]
    ok = y.notna(); X, y = X[ok], y[ok]
    vc = y.value_counts(); good = vc[vc >= 30].index
    m = y.isin(good); X, y = X[m], y[m]
    if y.nunique() < 2: return None, "classes"
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype == bool or str(X[c].dtype) == "category":
            X[c] = X[c].astype(object).where(X[c].notna(), X[c].mode().iloc[0] if X[c].notna().any() else "NA")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce"); X[c] = X[c].fillna(X[c].median())
    cat = [c for c in X.columns if X[c].dtype == object or X[c].dtype == bool]
    drop = [c for c in cat if X[c].nunique() > 20]
    X = X.drop(columns=drop); cat = [c for c in cat if c not in drop]
    X = pd.get_dummies(X, columns=cat, dtype=float)
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 4: return None, "features"
    yy, _ = pd.factorize(y)
    if np.bincount(yy).max() / len(yy) >= 0.95: return None, "majority>=95%"
    # leak/trivial stump check
    for j in range(X.shape[1]):
        st = DecisionTreeClassifier(max_depth=1).fit(X.iloc[:, [j]], yy)
        if (st.predict(X.iloc[:, [j]]) == yy).mean() >= 0.99: return None, "stump>=99%"
    return (X.astype(float).values, yy, t), "ok"

if __name__ == "__main__":
    out, log = {}, []
    for name in pdata().dataset_id:
        try:
            df = pdata(name)
            if not isinstance(df, pd.DataFrame): continue
            r, why = prep(df, name)
        except Exception as e:
            r, why = None, "error:" + type(e).__name__
        log.append((name, why))
        if r is not None: out["pyd_" + name] = r
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
    for nm, f in [("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer), ("digits", load_digits)]:
        b = f(); out["skl_" + nm] = (b.data.astype(float), b.target, "target")
    np.savez_compressed("datasets.npz", **{k + "__X": v[0] for k, v in out.items()}, **{k + "__y": v[1] for k, v in out.items()})
    man = [{"name": k, "n": int(v[0].shape[0]), "d": int(v[0].shape[1]), "classes": int(len(set(v[1]))),
            "target": v[2], "majority_share": round(float(np.bincount(v[1]).max() / len(v[1])), 3)} for k, v in out.items()]
    json.dump(man, open("manifest.json", "w"), indent=1)
    from collections import Counter
    print("qualifying tasks:", len(out)); print(Counter(w.split("(")[0] for _, w in log))
