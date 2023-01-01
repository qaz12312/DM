from math import log2,sqrt
import json
import os
def normalize(dc:dict):
    ss=0
    for _,v in dc.items():
        ss+=v
    
    ss/=len(dc)
    for k in dc:
        dc[k]/=ss

alpha=0.09
dsp='dev'
algos=['bm25_0.13_h1000','mDPR_h1000','bm25_0.08_h1000']
langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
for lang in langs:

    Runfiles=[]
    for algo in algos:
        runfilepath=f'data/{algo}/{lang}_{dsp}.txt'
        
        runfile={}
        try:
            with open(runfilepath,'r',encoding='utf-8') as f:
                for li in f.readlines():
                    tid,_,docid,_,rel,_=li.split(' ')
                    rel=float(rel)
                    
                    if tid not in runfile:
                        runfile[tid]={}

                    runfile[tid][docid]=rel
        except:
            pass

        for tid in runfile:
            normalize(runfile[tid])
        
        Runfiles.append(runfile)

    Runfile={}
    for i,runfile in enumerate(Runfiles):
        for tid in runfile:
            for docid in runfile[tid]:
                if tid not in Runfile:
                    Runfile[tid]={}
                if docid not in Runfile[tid]:
                    Runfile[tid][docid]=[0.0 for _ in range(len(algos))]
                Runfile[tid][docid][i]=runfile[tid][docid]

    Result=[]
    X=[]
    L=[]
    for tid in Runfile:
        for docid in Runfile[tid]:
            X.append(Runfile[tid][docid])
            L.append((tid,docid))
        
    try:
        os.makedirs(f'candidates/{dsp}')
    except:
        pass
    with open(f'candidates/{dsp}/{lang}.json','w') as f:
        json.dump((X,L),f)