import pandas as pd, requests, os, re
from concurrent.futures import ThreadPoolExecutor
d=pd.read_csv('rd_index.csv')
s=d[(d.Rows>=200)&(d.Rows<=20000)&(d.Cols<=100)]
def go(r):
    item=re.sub(r"\s*\(.*\)$","",r.Item)
    fn=f"rd_cache/{r.Package}__{item}.csv"
    if os.path.exists(fn): return "cached"
    u=f"https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/{r.Package}/{item}.csv"
    try:
        x=requests.get(u,timeout=60)
        if x.status_code==200: open(fn,"wb").write(x.content); return "ok"
        return str(x.status_code)
    except Exception as e: return "err"
with ThreadPoolExecutor(8) as ex: res=list(ex.map(go,s.itertuples()))
from collections import Counter; print(Counter(res))
