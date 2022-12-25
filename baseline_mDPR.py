"""
This script is used to run mDPR on MIRACL dataset.
"""
import os

RESULTS_DIR = './results/mDPRs'
HITS = 1000

for lang in ['ar','bn','en', 'es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th', 'zh']:
    ans_path = f'miracl-v1.0-{lang}-dev'
    ret_path = f'{RESULTS_DIR}/{lang}-dev.txt'

    # runfiles
    os.system(f'python -m pyserini.search.faiss \
        --encoder-class auto \
        --encoder castorini/mdpr-tied-pft-msmarco \
        --topics {ans_path} \
        --index miracl-v1.0-{lang}-mdpr-tied-pft-msmarco \
        --output {ret_path} \
        --batch 128 --threads 16 --hits {HITS}')

    # evaluate recall@HITS
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -m recall.{HITS} {ans_path} \
        {ret_path}')

    # evaluate nDCG@10
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -M {HITS} -m ndcg_cut.10 {ans_path} \
        {ret_path}')