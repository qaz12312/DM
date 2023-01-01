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

    
    X=[]
    Y=[]
    qrelpath=f'miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv'
    try:
        with open(qrelpath,'r',encoding='utf-8') as f:
            for li in f.readlines():
                tid,_,docid,rel=li.split('\t')
                rel=int(rel)
                        
                if tid not in runfile:
                    runfile[tid]={}
                    
                x=[]
                for i in range(len(algos)):
                    if len(Runfiles[i])==0:
                        x.append(0.0)
                    elif docid in Runfiles[i][tid]:
                        x.append(Runfiles[i][tid][docid])
                    else:
                        x.append(0.0)
                X.append(x)
                Y.append(rel)    
    except:
        pass
    try:
        os.makedirs(f'datasets/{dsp}')
    except:
        pass
    with open(f'datasets/{dsp}/{lang}.json','w') as f:
        json.dump((X,Y),f)