from math import log2,sqrt
def normalize(dc:dict):
    ss=0
    for _,v in dc.items():
        ss+=v
    
    ss/=len(dc)
    for k in dc:
        dc[k]/=ss

alpha=0.09
dsp='test-a'
for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    bm25_file=f'data/bm25_0.13_h1000/{lang}_{dsp}.txt'
    mDPR_file=f'data/mDPR_h1000/{lang}-{dsp}.txt'
    bm25_dict={}
    mDPR_dict={}
    try:
        with open(bm25_file,'r',encoding='utf-8') as f:
            for li in f.readlines():
                tid,_,docid,_,rel,_=li.split(' ')
                rel=float(rel)
                
                if tid not in bm25_dict:
                    bm25_dict[tid]={}

                bm25_dict[tid][docid]=rel
        
        with open(mDPR_file,'r',encoding='utf-8') as f:
            for li in f.readlines():
                tid,_,docid,_,rel,_=li.split(' ')
                rel=float(rel)
                
                if tid not in mDPR_dict:
                    mDPR_dict[tid]={}

                mDPR_dict[tid][docid]=rel
    except:
        continue
    
    results=[]
    for tid in bm25_dict:
        normalize(bm25_dict[tid])
        normalize(mDPR_dict[tid])

        result={}

        for docid in bm25_dict[tid]:
            result[docid]=bm25_dict[tid][docid]*alpha

        for docid in mDPR_dict[tid]:
            if docid in result:
                result[docid]+=mDPR_dict[tid][docid]*(1-alpha)
            else:
                result[docid]=mDPR_dict[tid][docid]*(1-alpha)

        result=sorted(result.items(),key=lambda kv:-kv[1])
        results.append((tid,result[:100]))

    
    with open(f'data/hybrid/{lang}_{dsp}.txt','w') as f:
        for tid,res in results:
            rank=1
            for docid,rel in res:
                f.write(f'{tid} Q0 {docid} {rank} {rel} hybrid\n')
                rank+=1