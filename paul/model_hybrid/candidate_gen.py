from math import log2,sqrt
import json
import os
import numpy as np
def candidate_gen(dsp,algos,langs,datasetname):
    def normalize(dc:dict):
        ss=0
        for _,v in dc.items():
            ss+=v
        
        ss/=len(dc)
        for k in dc:
            dc[k]/=ss

    for lang in langs:
        Runfiles=[]
        for algo in algos:
            runfilepath=f'runfiles/{algo}/{lang}_{dsp}.txt'
            
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
            
            """
            for tid in runfile:
                normalize(runfile[tid])
            """
            
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
        
        datasetpath=f'dataset/{datasetname}'
        try:os.makedirs(f'{datasetpath}/pair')
        except:pass
        try:os.makedirs(f'{datasetpath}/candidate')
        except:pass
        np.savetxt(f'{datasetpath}/pair/{lang}_{dsp}.txt',np.array(L,dtype=np.str_),delimiter=',',fmt='%s',newline='\n')
        np.savetxt(f'{datasetpath}/candidate/{lang}_{dsp}.txt',np.array(X),delimiter=',',newline='\n')

if __name__=='__main__':
    dsp='dev'
    algos=['mdpr','bm25','qld','rm3','rocchio','rocchio-nonrelevant']
    langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
    datasetname='all'
    candidate_gen(dsp,algos,langs,datasetname)