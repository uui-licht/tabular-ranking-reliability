import json, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
R=int(sys.argv[1]); NCAP=3000
def models(seed):
    return {
     "LR": make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
     "kNN": make_pipeline(StandardScaler(),KNeighborsClassifier(5)),
     "SVM": make_pipeline(StandardScaler(),SVC()),
     "RF": RandomForestClassifier(200,random_state=seed,n_jobs=1),
     "ET": ExtraTreesClassifier(200,random_state=seed,n_jobs=1),
     "HGB": HistGradientBoostingClassifier(random_state=seed),
     "XGB": XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,n_jobs=1,random_state=seed,verbosity=0),
     "LGBM": LGBMClassifier(n_estimators=200,learning_rate=0.1,n_jobs=1,random_state=seed,verbose=-1),
     "MLP": make_pipeline(StandardScaler(),MLPClassifier((64,),max_iter=300,random_state=seed)),
    }
names=[m["name"] for m in json.load(open("manifest_final.json"))]
def job(name,r):
    Z=np.load("datasets_final.npz"); X=Z[name+"__X"]; y=Z[name+"__y"]
    if len(y)>NCAP:
        s=StratifiedShuffleSplit(1,train_size=NCAP,random_state=12345); i,_=next(s.split(X,y)); X,y=X[i],y[i]
    s=StratifiedShuffleSplit(1,test_size=0.3,random_state=1000+r); tr,te=next(s.split(X,y))
    rows=[]
    for mn,m in models(r).items():
        t=time.time()
        try:
            m.fit(X[tr],y[tr]); acc=float((m.predict(X[te])==y[te]).mean())
        except Exception as e: acc=float("nan")
        rows.append((name,r,mn,acc,len(te),time.time()-t))
    return rows
if __name__=="__main__":
    tasks=[(n,r) for r in range(R) for n in names]
    t0=time.time(); res=Parallel(n_jobs=2,verbose=0)(delayed(job)(n,r) for n,r in tasks)
    df=pd.DataFrame([x for rr in res for x in rr],columns=["task","rep","model","acc","n_test","sec"])
    df.to_csv(sys.argv[2],index=False); print("done",time.time()-t0,df.acc.isna().sum())
