"""
This script is used to run the baseline models using different algorithms on MIRACL dataset.
"""
import os
from collections import defaultdict
import json
from utils import *

HITS = 1000

k1 = 0 # 1.6
b = 0 # 0.75
bm25_params = f'--k1 {k1} --b {b} ' if k1 > 0 and b > 0 else ''
ALGORITHMS = {
    'bm25' : f'--bm25 {bm25_params}', # BM25
    'rm3' : '--rm3', # RM3
    'rocchio' : '--rocchio',
    'rocchio_n' : '--rocchio-use-negative', # Use nonrelevant labels in Rocchio
    'qld': '--qld', # Use QLD
    'ql' : '--ql',   # Query Likelihood
    ################ Another prefix #############################
    'mDPR' : '',
}

for k in ALGORITHMS.keys():
    curr_alg = k
    log_path = f'./{curr_alg}s/results.json'
    all_lang_logs = defaultdict(dict)

    for lang in LANGUAGES:
        for dataset_type in DATASET_PATHS[lang].keys():
            results_dir_path = f'./{curr_alg}s/{dataset_type}'
            target_path = f'{results_dir_path}/{lang}-{dataset_type}.txt'

            # runfiles
            if curr_alg == 'mDPR':
                os.system(f'python -m pyserini.search.faiss \
                    --encoder-class auto \
                    --encoder castorini/mdpr-tied-pft-msmarco-ft-all \
                    --index miracl-v1.0-{lang}-mdpr-tied-pft-msmarco \
                    --topics {DATASET_PATHS[lang][dataset_type][0]} \
                    --output {target_path} \
                    --batch 128 --threads 16 \
                    --hits {HITS}')
            else:
                os.system(f'python -m pyserini.search.lucene \
                    --index miracl-v1.0-{lang} \
                    --language {lang} \
                    --topics {DATASET_PATHS[lang][dataset_type][0]} \
                    --output {target_path} \
                    --batch 128 --threads 16 \
                    {ALGORITHMS[curr_alg]} \
                    --hits {HITS}')
            
            if len(DATASET_PATHS[lang][dataset_type]) == 1:
                print(f'ans file not exists. [{dataset_type} of {lang}]')
            else:
                try:
                    ans_path = DATASET_PATHS[lang][dataset_type][1]
                    # recall@100
                    temp = os.popen(f'python -m pyserini.eval.trec_eval \
                        -c -m recall.100 {ans_path} \
                        {target_path}').readlines()[5]
                    recall = temp.split('\t')[-1].replace('\n','')

                    # nDCG@10
                    temp = os.popen(f'python -m pyserini.eval.trec_eval \
                        -c -M 100 -m ndcg_cut.10 {ans_path} \
                        {target_path}').readlines()[5]
                    ndcg = temp.split('\t')[-1].replace('\n','')
                    print(f'[{dataset_type} of {lang}] done.')
                except:
                    print(f'[{dataset_type} of {lang}] failed.')
                    recall = 0
                    ndcg = 0
                all_lang_logs[lang][dataset_type] = {'recall' : recall, 'ndcg' : ndcg}

    # with open(log_path, 'a') as f:
    #     f.write('\t|lang|recall@100|nDCG@10|\n\t|-|-|-|\n')
    #     for item in all_lang_logs:
    #         f.write(f"\t{item}\n")
    #     # average of all languages
    #     recall = sum([float(item.split('|')[2]) for item in all_lang_logs]) / len(all_lang_logs)
    #     ndcg = sum([float(item.split('|')[3]) for item in all_lang_logs]) / len(all_lang_logs)
    #     f.write(f'\t|Avg|{recall:.4f}|{ndcg:.4f}|\n')

    with open(log_path, 'w') as fp:
        json.dump(all_lang_logs, fp)