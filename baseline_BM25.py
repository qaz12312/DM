"""
This script runs BM25 baseline on MIRACL v1.0.
"""
import os
import numpy as np

RESULTS_DIR = './results/BM25s'

HITS = 1000

for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    ans_path = f'miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv'
    for k1 in [0]:# np.arange(1.2, 2.0, 0.2): # 詞頻對文檔相關度的影響
        for b in [0]:# np.arange(0.6, 0.9, 0.1): # 文檔長度對文檔相關度的影響
            ret_path = f'{RESULTS_DIR}/{lang}-dev_k1{k1}_b{b}.txt'

            # runfiles
            bm25_params = f'--k1 {k1} --b {b} ' if k1 > 0 and b > 0 else '' 
            os.system(f'python -m pyserini.search.lucene \
                --language {lang} \
                --topics miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv \
                --index miracl-v1.0-{lang} \
                --output {ret_path} \
                --batch 128 --threads 16 --bm25 {bm25_params}--hits {HITS}')
            
            # evaluate recall@HITS
            os.system(f'python -m pyserini.eval.trec_eval \
                -c -m recall.{HITS} {ans_path} \
                {ret_path}')
            
            # evaluate nDCG@10
            os.system(f'python -m pyserini.eval.trec_eval \
                -c -M {HITS} -m ndcg_cut.10 {ans_path} \
                {ret_path}')