"""Final freeze. Rules R1-R4 (build_datasets.prep) plus, declared before any model was run:
R5 target-name filter (year, period, id, ...), R8 target must be a non-numeric column or a 0/1 integer/logical column,
R6 at most 3 tasks per Rdatasets package (seeded shuffle, seed 0), duplicates removed by (n,d,classes,majority share)."""
import glob, re, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from build_datasets import prep
from pydataset import data as pdata
from collections import Counter
bad_t=re.compile(r"^(year|yr|period|time|date|month|week|day|id|index|obs|rownames|rep|block|plot|trial)$",re.I)
def r8(s):
    if s.dtype==object or str(s.dtype)=="category" or s.dtype==bool: return True
    return set(pd.Series(s).dropna().unique())<= {0,1}
log=Counter(); out={}; seen=set()
fp=lambda X,y:(X.shape[0],X.shape[1],len(set(y)),round(float(np.bincount(y).max()/len(y)),3))
def handle(name,df,src):
    df=df.loc[:,~df.columns.duplicated()]
    r,why=prep(df,name)
    if r is None: log[why.split("(")[0]]+=1; return None
    X,y,t=r
    if bad_t.match(str(t)): log["R5"]+=1; return None
    if not r8(df[t]): log["R8_numeric_target"]+=1; return None
    k=fp(X,y)
    if k in seen: log["duplicate"]+=1; return None
    seen.add(k); return (X,y,t)
py={}
for name in pdata().dataset_id:
    try:
        df=pdata(name)
        if not isinstance(df,pd.DataFrame): continue
        r=handle(name,df,"pyd")
        if r: py["pyd_"+name]=r
    except Exception: log["error"]+=1
rd={}
for f in sorted(glob.glob("rd_cache/*.csv")):
    pk,it=f.split("/")[-1][:-4].split("__")
    try:
        df=pd.read_csv(f)
        if df.columns[0] in("rownames","Unnamed: 0","") : df=df.drop(columns=df.columns[0])
        r=handle(it,df,"rd")
        if r: rd[f"rd_{pk}_{it}"]=r
    except Exception: log["error"]+=1
print(log); print(len(py),len(rd))
rng=np.random.default_rng(0); bypk={}
for k in sorted(rd): bypk.setdefault(k.split("_")[1],[]).append(k)
sel=[]
for pk in sorted(bypk): sel+=list(rng.permutation(bypk[pk]))[:3]
out.update(py); out.update({k:rd[k] for k in sel})
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
for nm,f in [("iris",load_iris),("wine",load_wine),("breast_cancer",load_breast_cancer),("digits",load_digits)]:
    b=f(); out["skl_"+nm]=(b.data.astype(float),b.target,"target")
np.savez_compressed("datasets_final.npz",**{k+"__X":v[0] for k,v in out.items()},**{k+"__y":v[1] for k,v in out.items()})
man=[{"name":k,"n":int(v[0].shape[0]),"d":int(v[0].shape[1]),"classes":int(len(set(v[1]))),"target":str(v[2]),"majority_share":round(float(np.bincount(v[1]).max()/len(v[1])),3)} for k,v in out.items()]
json.dump(man,open("manifest_final.json","w"),indent=1)
print(len(man)); 
for m in man: print(m["name"],m["n"],m["d"],m["classes"],m["target"],m["majority_share"])
