import numpy as np, pandas as pd, json, sys
from scipy.stats import kendalltau
rng=np.random.default_rng(2026)
df=pd.read_csv("raw_results.csv")
DROP=["rd_OncoDataSets_BreastCancerWI_df","rd_mosaicData_HELPmiss"]  # same data as skl_breast_cancer / near-duplicate of HELPrct
df=df[~df.task.isin(DROP)]
tasks=sorted(df.task.unique()); models=["LR","kNN","SVM","RF","ET","HGB","XGB","LGBM","MLP"]
A=np.full((len(tasks),df.rep.max()+1,len(models)),np.nan)
for r in df.itertuples():
    A[tasks.index(r.task),r.rep,models.index(r.model)]=r.acc
assert not np.isnan(A).any(); T,R,M=A.shape; print(T,R,M)
jit=lambda x: x+rng.normal(0,1e-9,x.shape)
def tau(a,b): return kendalltau(a,b)[0]
# --- descriptive
mean=A.mean(1); sd=A.std(1,ddof=1)
desc=pd.DataFrame({"task":tasks,"best_model":[models[i] for i in mean.argmax(1)],"best_acc":mean.max(1),
   "gap_top2":[np.sort(m)[-1]-np.sort(m)[-2] for m in mean],"mean_sd_single_split":sd.mean(1),"n_test":df.groupby("task").n_test.first().reindex(tasks).values})
desc.to_csv("out/table_tasks.csv",index=False)
glob=pd.DataFrame({"model":models,"mean_acc":mean.mean(0),"mean_rank":(-mean).argsort(1).argsort(1).mean(0)+1,
  "n_tasks_best":[(mean.argmax(1)==j).sum() for j in range(M)]}); glob.to_csv("out/table_models.csv",index=False)
print(desc.round(3).to_string()); print(glob.round(3))
# --- A. per-task reliability vs k (reference = disjoint half of reps)
KS=[1,2,5,10,20,40]; NDRAW=300; per=[]
for t in range(T):
    for d in range(NDRAW):
        p=rng.permutation(R); ref=A[t,p[:50]].mean(0); pool=p[50:]
        for k in KS:
            est=A[t,pool[:k]].mean(0)
            per.append((t,d,k,tau(jit(est),jit(ref)),int(np.argmax(jit(est))==np.argmax(jit(ref)))))
P=pd.DataFrame(per,columns=["t","d","k","tau","top1"])
pt=P.groupby(["t","k"])[["tau","top1"]].mean().reset_index(); pt.to_csv("out/per_task_reliability.csv",index=False)
sumA=pt.groupby("k")[["tau","top1"]].agg(["mean","median","min"]).round(3); sumA.to_csv("out/reliability_by_k.csv"); print(sumA)
fr=pt[pt.k==1].tau; print("tasks with single-split tau<0.5:",(pt[pt.k==1].tau<0.5).sum(),"of",T)
# relationship of single-split tau with gap and n_test
pt1=pt[pt.k==1].set_index("t"); desc["tau_k1"]=pt1.tau.values; desc["top1_k1"]=pt1.top1.values
from scipy.stats import spearmanr
print("spearman tau_k1 vs ntest",spearmanr(desc.tau_k1,desc.n_test)); print("spearman top1_k1 vs gap",spearmanr(desc.top1_k1,desc.gap_top2))
desc.to_csv("out/table_tasks.csv",index=False)
# --- B. pairwise reversal rate single split vs gap/sd
rows=[]
for t in range(T):
    ref=A[t].mean(0)
    for i in range(M):
        for j in range(i+1,M):
            g=ref[i]-ref[j]
            if g==0: continue
            rev=((A[t,:,i]-A[t,:,j])*np.sign(g)<0).mean()
            rows.append((t,i,j,abs(g),sd[t].mean(),rev))
PW=pd.DataFrame(rows,columns=["t","i","j","gap","sd","rev"]); PW["z"]=PW.gap/PW.sd
bins=[0,0.005,0.01,0.02,0.05,1]; PW["bin"]=pd.cut(PW.gap,bins)
pw=PW.groupby("bin").rev.agg(["mean","count"]).round(3); pw.to_csv("out/pairwise_reversal_by_gap.csv"); print(pw)
# --- C. aggregation across datasets: D x k
def ranks(m): return (-jit(m)).argsort(1).argsort(1)+1
DS=[3,5,10,19]; KS2=[1,3,10,30]; res=[]
for d in range(400):
    p=rng.permutation(R); refpool=p[:50]; pool=p[50:]
    allt=rng.permutation(T); holdT=allt[:T//2]; poolT=allt[T//2:]   # disjoint task halves for generalisation to new datasets
    refrank=ranks(A[holdT][:,refpool].mean(1)).mean(0)
    for D in DS:
        if D>len(poolT): continue
        sel=poolT[:D]
        for k in KS2:
            est=A[sel][:,pool[:k]].mean(1)
            er=ranks(est).mean(0)
            res.append((D,k,tau(jit(er),jit(refrank)),int(np.argmin(jit(er))==np.argmin(jit(refrank)))))
C=pd.DataFrame(res,columns=["D","k","tau","top1"]); ct=C.groupby(["D","k"]).agg(tau=("tau","mean"),top1=("top1","mean")).round(3).reset_index()
ct.to_csv("out/aggregate_D_by_k.csv",index=False); print(ct.pivot(index="D",columns="k",values="tau")); print(ct.pivot(index="D",columns="k",values="top1"))
# --- D. min k for tau>=0.9 mean
print("per-task tau_k by task (k=1,5,20)"); print(pt.pivot(index="t",columns="k",values="tau").round(2).describe().round(2))
pd.DataFrame({"task":tasks}).to_csv("out/tasks_order.csv",index=False)
