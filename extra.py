import numpy as np, pandas as pd
exec(open("analyze_exp.py").read().split("# --- descriptive")[0])
def ranks(m): return (-jit(m)).argsort(1).argsort(1)+1
rows=[]
for d in range(400):
    p=rng.permutation(R); ref=p[:50]; pool=p[50:]
    for D in [3,5,10,20,T]:
        sel=rng.permutation(T)[:D]
        rr=ranks(A[sel][:,ref].mean(1)).mean(0)
        for k in [1,3,10,30]:
            er=ranks(A[sel][:,pool[:k]].mean(1)).mean(0)
            rows.append((D,k,tau(jit(er),jit(rr)),int(np.argmin(jit(er))==np.argmin(jit(rr)))))
C=pd.DataFrame(rows,columns=["D","k","tau","top1"]).groupby(["D","k"]).mean().round(3).reset_index()
C.to_csv("out/aggregate_samedata.csv",index=False)
print(C.pivot(index="D",columns="k",values="tau"));print(C.pivot(index="D",columns="k",values="top1"))
