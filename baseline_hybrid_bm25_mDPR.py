"""
Hybrid BM25 and mDPR results for all languages and all alphas.
formula: alpha * BM25 + (1-alpha) * mDPR
"""
import os
import sys
import numpy as np
from collections import defaultdict
dsp=sys.argv[1] # 'dev' or 'test-a'

BM25_DIR = './results/BM25s'
MDPR_DIR = './results/mDPRs'
HYBRIDS_DIR = './results/hybrids'
LOGGING_PATH = './results/all_lang_results.log'


def normalize_avg(qid_docids:dict)->None:
    """
    Normalize the scores of the documents for a given query by the average score of the documents for that query. 
    """
    avg_val = sum(qid_docids.values()) / len(qid_docids)
    for k in qid_docids:
        qid_docids[k] /= avg_val


def read_qrels(qrels_path:str, qid_docids:dict)->dict:
    """
    Read qrels file.  
    Return a dict of dicts, format: {query id: {document id:relevance score, ...}, ...}
    The relevance score is normalized by the average relevance score for that query.
    """
    qrels = defaultdict(dict)
    with open(qrels_path, 'r') as f:
        for li in f.readlines():
            qid, _, docid, rank, rel, *_ = li.split(' ')
            qid_docids[qid].append(docid)
            qrels[qid][docid] = float(rel)
    for qid in qrels:
        normalize_avg(qrels[qid])
    return qrels


for alpha in np.arange(0.0, 0.11, 0.01): # alpha * BM25 + (1-alpha) * mDPR
    all_lang_results = list() # for logging
    qid_docids = defaultdict(list) # qid -> [docid1, docid2, ...]
    for lang in ['ar','bn','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th', 'zh']: # ,'en'
        mdpr_results = read_qrels(f'{MDPR_DIR}/{lang}-{dsp}.txt', qid_docids)
        bm25_results = read_qrels(f'{BM25_DIR}/{lang}_{dsp}.txt', qid_docids)
        """
        Hybrid
        """
        target_path = f'{HYBRIDS_DIR}/{lang}_{dsp}_{alpha}.txt'
        with open(target_path, 'w') as f:
            for qid in qid_docids:
                rank_list = list()
                for docid in list(set(qid_docids[qid])):
                    # 加權平均(BM25 更加關注詞頻，mDPR 則更加關注詞向量之間的相似度)
                    score = alpha * bm25_results[qid].get(docid, 0) + (1-alpha) * mdpr_results[qid].get(docid, 0)
                    rank_list.append((docid, score))
                
                rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True) # 由大到小排序
                for i in range(len(rank_list)):
                    if i >= 100:
                        break
                    docid, score = rank_list[i]
                    f.write(f'{qid} Q0 {docid} {i+1} {score} hybrid\n')
        """
        Evaluate
        """
        ans_path = f'miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv'
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

        all_lang_results.append(f'|{lang}|{recall}|{ndcg}|')
        print(f'{lang} done')

    """
    Logging
    """
    with open(LOGGING_PATH, 'a') as f:
        f.write(f'\n+ lang={lang} alpha={alpha}\n')
        f.write('\t|lang|recall@100|nDCG@10|\n\t|-|-|-|\n')
        for item in all_lang_results:
            f.write(f"\t{item}\n")
        # average of all languages
        recall = sum([float(item.split('|')[2]) for item in all_lang_results]) / len(all_lang_results)
        ndcg = sum([float(item.split('|')[3]) for item in all_lang_results]) / len(all_lang_results)
        f.write(f'\t|Avg|{recall}|{ndcg}|\n')