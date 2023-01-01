import os
dsp='dev'
runfiles_dir='data/model_hybrid'

print('|lang|recall@100|nDCG@10|\n|-|-|-|')
langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
N=0
Sndcg=0
Srecall=0
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
    Sndcg+=float(ndcg)
    Srecall+=float(recall)
    print(f'|{lang}|{recall}|{ndcg}|')

print(f'|Avg|{round(Srecall/N,4)}|{round(Sndcg/N,4)}|')