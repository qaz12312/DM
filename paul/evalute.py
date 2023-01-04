import os

def evalute(dsp,methodname,langs):
    runfiles_dir=f'runfiles/{methodname}'
    N=0
    Sndcg=0
    Srecall=0
    print('||Recall@100|nDCG@10|')
    print('|-|-|-|')
    R={'lang':[],'recall':[],'ndcg':[]}
    for lang in langs:
        try:
            # evaluate recall@100
            temp=os.popen(f'python -m pyserini.eval.trec_eval \
                -c -m recall.100 miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv \
                {runfiles_dir}/{lang}_{dsp}.txt').readlines()[5]
            recall=temp.split('\t')[-1].replace('\n','')

            # evaluate nDCG@10
            temp=os.popen(f'python -m pyserini.eval.trec_eval \
                -c -M 100 -m ndcg_cut.10 miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv \
                {runfiles_dir}/{lang}_{dsp}.txt').readlines()[5]
            ndcg=temp.split('\t')[-1].replace('\n','')
        except:
            continue
        
        N+=1
        recall=float(recall)
        ndcg=float(ndcg)
        Sndcg+=ndcg
        Srecall+=recall
        R['lang'].append(lang)
        R['recall'].append(recall)
        R['ndcg'].append(ndcg)
        print(f'|{lang}|{round(recall,4)}|{round(ndcg,4)}|')

    print(f'|Avg|{round(Srecall/N,4)}|{round(Sndcg/N,4)}|')
    R['Avg']={'recall':Srecall/N,'ndcg':Sndcg/N}
    with open(f'{methodname}_{dsp}.json','w') as f:
        import json
        json.dump(R,f)

if __name__=='__main__':
    dsp='dev'
    methodname=f'model_hybrid'
    langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
    evalute(dsp,methodname,langs)