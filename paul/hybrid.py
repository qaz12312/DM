from math import log2,sqrt
import os
from .evalute import *
def hybrid(dsp,algos,weights,langs,methodname,hits):
    def normalize(dc:dict):
        ss=0
        for _,v in dc.items():
            ss+=v
        
        ss/=len(dc)
        for k in dc:
            dc[k]/=ss

    for lang in langs:
        algo_dict=[{}for _ in range(len(algos))]
        
        for i,algo in enumerate(algos):
            try:
                runfilepath=f'runfiles/{algo}/{lang}_{dsp}.txt'
                    
                with open(runfilepath,'r',encoding='utf-8') as f:
                    for li in f.readlines():
                        tid,_,docid,_,rel,_=li.split(' ')
                        rel=float(rel)
                            
                        if tid not in algo_dict[i]:
                            algo_dict[i][tid]={}

                        algo_dict[i][tid][docid]=rel
            except:
                pass

        for algo in algo_dict:
            for tid in algo:
                normalize(algo[tid])

        Result={}
        for i,algo in enumerate(algo_dict):
            for tid in algo:
                for docid in algo[tid]:
                    if tid not in Result:
                        Result[tid]={}
                    if docid not in Result[tid]:
                        Result[tid][docid]=algo[tid][docid]*weights[i]
                    else:
                        Result[tid][docid]+=algo[tid][docid]*weights[i]
        
        R=[]
        for tid in Result:
            r=sorted(Result[tid].items(),key=lambda kv:-kv[1])
            R.append((tid,r[:hits]))

        mdir=f'runfiles/{methodname}'
        try:
            os.makedirs(mdir)
        except:
            pass
        with open(f'{mdir}/{lang}_{dsp}.txt','w') as f:
            for tid,res in R:
                rank=1
                for docid,rel in res:
                    f.write(f'{tid} Q0 {docid} {rank} {rel} {methodname}\n')
                    rank+=1

if __name__=='__main__':
    langs=['ar','bn','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
    hybrid('dev',['mdpr','mdpr_tydi','qld'],[0.1,0.8,0.1],langs,'temp',100)
    hybrid('dev',['bm25_0.13'],[1],['en'],'temp',100)
    evalute('dev','temp',['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh'])
    hybrid('test-a',['mdpr','mdpr_tydi','qld'],[0.1,0.8,0.1],langs,'temp',100)
    hybrid('test-a',['bm25_0.13'],[1],['en'],'temp',100)

    #hybrid('dev',['mdpr','bm25_0.13'],[0.91,0.09],langs,'mdpr_bm25_0.13',100)
    #hybrid('dev',['bm25_0.13'],[1],['en'],'mdpr_bm25_0.13',100)
    #evalute('dev','mdpr_bm25_0.13',['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh'])
    