import os
import sys
data_split=sys.argv[1]
if not os.path.isdir('evaluations'):
    os.makedirs('evaluations')
for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    os.system(f'python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 ./miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{data_split}.tsv ./runfiles/{lang}_{data_split}.txt > ./evaluations/{lang}_{data_split}.txt')