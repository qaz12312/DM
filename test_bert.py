"""
Run this script to test the performance of BERT on MIRACL v1.0.
"""
import os
from collections import defaultdict
import datasets
import transformers
import torch
import textwrap
from scipy.spatial.distance import cosine
import numpy as np
from pyserini.eval.trec_eval import trec_eval
print('cuda: ', torch.cuda.is_available())

SURPRISE_LANGUAGES = ['de', 'yo']
NEW_LANGUAGES = ['es', 'fa', 'fr', 'hi', 'zh'] + SURPRISE_LANGUAGES
LANGUAGES = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh'] + SURPRISE_LANGUAGES
PROJECT_PATH = './transforms'
RESULT_PATH = F'{PROJECT_PATH}/results'
"""
{'en': {'train':[], 'dev':[], 'testB':[], 'testA':[]}}
"""
CORPUS_NAME = 'miracl/miracl-corpus'
DATASET_PATHS = {
    lang: {
        'dev': [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv',
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv',
        ],
        'testB': [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-b.tsv',
        ],
    } for lang in LANGUAGES
}
for lang in LANGUAGES:
    if lang not in SURPRISE_LANGUAGES:
        DATASET_PATHS[lang]['train'] = [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv',
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv',
        ]
    if lang not in NEW_LANGUAGES:
        DATASET_PATHS[lang]['testA'] = [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-a.tsv',
        ]


def load_topics(file_path:str)->dict:
    """
    Load topics from file.

    Return:
        {'query_id' : 'query', ...}
    """
    if os.path.exists(file_path) == False:
        raise ValueError(f'File not found ! [{file_path}]')
    qid2topic = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            qid, topic = line.strip().split('\t')
            qid2topic[qid] = topic # '5455987#0' : '月球到地球的距离是多少？'
    return qid2topic


def load_qrels(file_path:str)->dict:
    """
    Load qrels from file.

    Return:
        { 'query_id': { 'artical_passage_id': relevance , ...}, ...} 
    """
    if os.path.exists(file_path) == False:
        raise ValueError(f'File not found ! [{file_path}]')

    qrels = defaultdict(dict)
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t') # e.g., 1000222#0	Q0	453#50	1
            qrels[qid][docid] = int(rel) # '1000222#0': { '453#50': 1 , ...}
    return qrels


def load_corpus(lang:str)->dict:
    """
    Load corpus from dataset.

    Return:
        { 'artical_passage_id': ('artical_title', 'passage'), ...}
    """
    miracl_corpus = datasets.load_dataset(CORPUS_NAME, lang)['train'] # please check dataset's keys() before get corpus
    docid2doc_dict = {line['docid']: (line['title'], line['text']) for line in miracl_corpus}
    return docid2doc_dict


def show_results(lang:str, file_paths:list)->dict:
    """
    Args:
        filepaths: (topics_path, qrels_path)

    Return:
        { '5455987#0' : { 
            'query':'月球到地球的距离是多少？', 
            'positive_passages':[
                {
                    'docid':'453#50', 
                }, ...]},
            'negative_passages':[...]},
        , ... }
    """
    if len(file_paths) != 2:
        raise ValueError('Pass file_paths arg with 2 elements.')
    
    docid2doc_dict = load_corpus(lang)
    qid2topic_dict = load_topics(file_paths[0])
    qrels_2dict = load_qrels(file_paths[1])
        
    result = dict()
    for qid, topic in qid2topic_dict.items():
        data = dict()
        data['query'] = topic

        pos_docids, neg_docids = list(), list()
        for docid, rel in qrels_2dict[qid].items():
            if rel == 1:
                pos_docids.append(docid)
            else:
                neg_docids.append(docid)
        data['positive_passages'] = [docid for docid in pos_docids if docid in docid2doc_dict]
        data['negative_passages'] = [docid for docid in neg_docids if docid in docid2doc_dict]      
        result[qid] = data
    return result


class BSA():
    def __init__(self, model_name:str, device:str)->None:
        self.model = transformers.BertModel.from_pretrained(model_name)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name) # 將文本轉換為 token ids
        self.device = device
        self.model = self.model.to(self.device)

    def transform_input(self, input_text:str)->tuple:
        """
        Return:
            input_ids: torch.Size(batch_size, len(input_text+[CLS]))
            attention_mask: torch.Size(batch_size, len(input_text+[CLS]))
        """
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = input_ids.ne(0).long() # 將 tensor 中非 0 的元素轉換為 1
        return input_ids, attention_mask

    def forward(self, input_text:str)->tuple:
        """
        Return:
            last_hidden_state: torch.Size(batch_size, len(input_text+[CLS]), len(word_embedding))
            pooler_output: tanh(FCN(last_hidden_state)). torch.Size(batch_size, len(word_embedding))
        """
        parts = textwrap.wrap(input_text, width=510)
        last_hidden = None
        pooler = np.array([[]])
        for part in parts:
            input_ids, attention_mask = self.transform_input(part)
            inputs = (input_ids.to(self.device), attention_mask.to(self.device))
            outputs = self.model(*inputs)
            last_hidden_state, pooler_output = outputs.last_hidden_state, outputs.pooler_output

            last = last_hidden_state.to('cpu').detach().numpy()
            last_hidden = last if last_hidden is None else np.concatenate([last_hidden, last], axis=1)

            pooler = np.concatenate([pooler, pooler_output.to('cpu').detach().numpy()], axis=1)
        
        return last_hidden, pooler


def rank_rels(query_vec:list, passage_vec_dict:dict, top_k:int=10)->list:
    """
    Args:
        query_vec: np.array([1, LEN_QUERY])
        passage_vec_dict: { 'docid': np.array([1, LEN_PASSAGE]), ...}
    Return:
        rank_list: [(docid, score), ...]
    """
    results = list()
    
    for docid, p_vec in passage_vec_dict.items():
        score = 1 - cosine(query_vec.flatten(), p_vec.flatten())
        results.append((docid, score))
    
    rank_list = sorted(results, key=lambda x: x[1], reverse=True) # 由大到小排序
    return rank_list[:top_k]


def check_dir(path:str)->None:
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"{path} is created!")


def write_result(file_path:str, data:dict)->None:
    with open(file_path, 'w') as f:
        for qid in data:
            i = 0
            for docid, score in data[qid]:
                f.write(f'{qid} Q0 {docid} {i+1} {score} bert\n')
                i += 1


def evaluateNDCG(qrels_path:str, result_path:str, metric:str='ndcg')->float:
    """
    evaluate NDCG@10
    """
    ndcg = trec_eval(result_path, qrels_path, metric)
    # temp = os.popen(f'python -m pyserini.eval.trec_eval -c -M 10 -m ndcg_cut.10 {qrels_path} {result_path}')
    # ndcg = temp.readlines()[5].split('\t')[-1].replace('\n','') if len(temp) > 4 else 0
    return ndcg


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsa = BSA('bert-base-multilingual-cased', device)

    all_lang_ndcgs = defaultdict(dict) # for logging
    for test_lang in LANGUAGES: # ['sw']
        
        # docid2doc_dict = {
        #     '517773#1':('Kad網路', 'Kad Network利用UDP'),
        #     '13#0' : ('数学', "数学，是研究數量、结构以及空间等概念及其变化的一門学科，从某种角度看屬於形式科學的一種。數學利用抽象化和邏輯推理，從計數、計算、量度、對物體形狀及運動的觀察發展而成。數學家們拓展這些概念，以公式化新的猜想，以及從選定的公理及定義出發，嚴謹地推導出一些定理。"),
        #     '1' : ('維p', '是1939年至1945年爆發的全球軍事衝突'),
        #     '2' : ('維p', '二战是不好的'),
        # }
        docid2doc_dict = load_corpus(test_lang)

        passages_vec_dict = dict()
        for docid, (title, passage) in docid2doc_dict.items():
            # print(docid, (title, passage))
            passage_embedding, passage_pooler = bsa.forward(passage)
            passages_vec_dict[docid] = passage_pooler
        
        print(f'{test_lang} load_corpus done.')

        for dataset_type in DATASET_PATHS[test_lang].keys(): # ['dev']
            check_dir(f'{RESULT_PATH}/{dataset_type}')
            ret_path = f'{RESULT_PATH}/{dataset_type}/{test_lang}.txt'

            qid2topic_dict = load_topics(DATASET_PATHS[test_lang][dataset_type][0]) # qid2topic_dict = {"1719936#0":"二战是什么时候开始的？"}

            rets = dict()
            for qid, topic in qid2topic_dict.items():
                # print(qid, topic)
                query_embedding, query_pooler = bsa.forward(topic)

                ans = rank_rels(query_pooler, passages_vec_dict)

                rets[qid] = ans
            
            print(f'\t{dataset_type} rank_rels done.')
            write_result(ret_path, rets)

            # evaluate nDCG@10
            if len(DATASET_PATHS[test_lang][dataset_type]) > 1:
                ndcg = evaluateNDCG(DATASET_PATHS[test_lang][dataset_type][1], ret_path)
                all_lang_ndcgs[test_lang][dataset_type] = ndcg
            
    with open('results.log', 'w') as f:
        f.write(f'+ nDCG@10\n')
        for test_lang in all_lang_ndcgs:
            f.write(f'\t+ {test_lang}\n')
            for dataset_type, score in all_lang_ndcgs[test_lang].items():
                f.write(f'\t\t+ {dataset_type}:{score}\n')