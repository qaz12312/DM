from math import log2,sqrt
import json
import os
import numpy as np
def frel_gen(dsp,algos,langs,datasetname):
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

        datasetpath=f'dataset/{datasetname}'
        try:os.makedirs(f'{datasetpath}/rel')
        except:pass
        try:os.makedirs(f'{datasetpath}/feature')
        except:pass
        np.savetxt(f'{datasetpath}/feature/{lang}_{dsp}.txt',np.array(X),delimiter=',',newline='\n')
        np.savetxt(f'{datasetpath}/rel/{lang}_{dsp}.txt',np.array(Y),delimiter=',',fmt='%d',newline='\n')

if __name__=='__main__':
    dsp='train'
    algos=['mdpr','mdpr_tydi','bm25','qld','rm3','rocchio','rocchio-nonrelevant']
    langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
    datasetname='mdpr_tydi_all'
    frel_gen(dsp,algos,langs,datasetname)
