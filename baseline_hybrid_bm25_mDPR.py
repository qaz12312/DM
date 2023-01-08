import os
import sys
import numpy as np
from collections import defaultdict
from utils import *
dataset_type=sys.argv[1] # 'dev' or 'test-a'

BM25_DIR = f'./bm25s/{dataset_type}'
MDPR_DIR = f'./mDPRs/{dataset_type}'
HYBRIDS_DIR = f'./hybrids/{dataset_type}'
LOGGING_PATH = './hybrids/results.json'


def normalize_avg(qid_docids:dict,qid)->None:
    """
    Normalize the scores of the documents for a given query by the average score of the documents for that query. 
    """
    avg_val = sum(qid_docids.values()) / len(qid_docids)
    for docid in qid_docids:
        qid_docids[docid] /= avg_val
        if qid_docids[docid] == 0:
            print(f"qid={qid} / docid={docid}")


def read_qrels(qrels_path:str, qid_docids:dict)->dict:
    """
    Read qrels file.  
    Return a dict of dicts, format: {query id: {document id:relevance score, ...}, ...}
    The relevance score is normalized by the average relevance score for that query.
    """
    qrels = defaultdict(dict)
    try:
        with open(qrels_path, 'r') as f:
            for li in f.readlines():
                qid, _, docid, rank, rel, *_ = li.split(' ')
                qid_docids[qid].append(docid)
                qrels[qid][docid] = float(rel)
        for qid in qrels:
            normalize_avg(qrels[qid],qid)
    except FileNotFoundError:
        print(f'File [{qrels_path}] not found.')
    return qrels


for alpha in [0.09]:# np.arange(0.1, 0.13, 0.01): # alpha * BM25 + (1-alpha) * mDPR
    case_str = f'hybrid, alpha={alpha}'
    all_lang_logs = defaultdict(dict)
    for lang in LANGUAGES:
        qid_docids = defaultdict(list) # qid -> [docid1, docid2, ...]
        mdpr_results = read_qrels(f'{MDPR_DIR}/{lang}-{dataset_type}.txt', qid_docids)
        bm25_results = read_qrels(f'{BM25_DIR}/{lang}-{dataset_type}.txt', qid_docids)
        """
        Hybrid
        """
        target_path = f'{HYBRIDS_DIR}/{lang}-{dataset_type}.txt'
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
        print(f'{lang} done')

        all_lang_logs[lang][dataset_type] = {'recall' : recall, 'ndcg' : ndcg}

"""
Logging
"""
# with open(LOGGING_PATH, 'w') as f:
#     f.write('\t|lang|recall@100|nDCG@10|\n\t|-|-|-|\n')
#     for item in all_lang_results:
#         f.write(f"\t{item}\n")
#     # average of all languages
#     recall = sum([float(item.split('|')[2]) for item in all_lang_results]) / len(all_lang_results)
#     ndcg = sum([float(item.split('|')[3]) for item in all_lang_results]) / len(all_lang_results)
#     f.write(f'\t|Avg|{recall:.4f}|{ndcg:.4f}|\n')