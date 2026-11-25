import numpy as np, pandas as pd
exec(open("analyze_exp.py").read().split("# --- descriptive")[0])
res=[]
def run(t,p,z,kmin=5,kmax=50):
    ref=A[t,p[:50]].mean(0); pool=p[50:]; X=A[t,pool]
    for k in range(kmin,kmax+1):
        m=X[:k].mean(0); o=np.argsort(-jit(m)); a,b=o[0],o[1]
        d=X[:k,a]-X[:k,b]; se=d.std(ddof=1)/np.sqrt(k)
        if se>0 and d.mean()/se>=z or k==kmax: break
    return k,int(a==np.argmax(jit(ref))),int(k<kmax)
for z in [1.0,1.5,2.0,2.5,3.0]:
    ks=[];hit=[];stop=[]
    for t in range(T):
        for d in range(300):
            p=rng.permutation(R); k,h,s=run(t,p,z); ks.append(k);hit.append(h);stop.append(s)
    res.append((z,np.mean(ks),np.mean(hit),np.mean(stop)))
    # fixed-budget comparison at same mean k
    kf=int(round(np.mean(ks))); fh=[]
    for t in range(T):
        for d in range(300):
            p=rng.permutation(R); ref=A[t,p[:50]].mean(0); est=A[t,p[50:][:kf]].mean(0); fh.append(int(np.argmax(jit(est))==np.argmax(jit(ref))))
    res[-1]+=(kf,np.mean(fh))
df=pd.DataFrame(res,columns=["z","mean_k","top1_adaptive","share_stopped_early","fixed_k","top1_fixed"]).round(3); print(df.to_string()); df.to_csv("out/adaptive.csv",index=False)
