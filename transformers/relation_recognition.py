import pandas as pd
from simpletransformers.classification import ClassificationArgs
from simpletransformers.classification import ClassificationModel
from pyserini.search.lucene import LuceneSearcher
import json
import sklearn
import os
import shutil

def get_dataset(lang:str,dsp:str,searcher:LuceneSearcher):
    tp_path=f'miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv'
    q_path=f'miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv'
    with open(tp_path,'r',encoding='utf-8') as f:
        tp=[t.split('\t') for t in f.readlines()]
        tp={tid:txt for tid,txt in tp}
            
    with open(q_path,'r',encoding='utf-8') as f:
        train_data=[]
        for li in f.readlines():
            tid,_,docid,rel=li.split('\t')
            doc=json.loads(searcher.doc(docid).raw())['text']
            topic=tp[tid]
            rel=int(rel)
            if rel==0:
                neg-=1
            else:
                neg=2
            if neg>=0:
                train_data.append([f'<s>{topic}</s>{doc}',rel])

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]
    return train_df

for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    searcher = LuceneSearcher.from_prebuilt_index(f'miracl-v1.0-{lang}')

    model_args = ClassificationArgs(
        overwrite_output_dir=True,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False
    )

    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
    )
    train_df=get_dataset(lang,'train',searcher)
    eval_df=get_dataset(lang,'dev',searcher)
    model.train_model(train_df)
    result, raw_outputs,wrongs=model.eval_model(eval_df)

    shutil.rmtree(f'./models/{lang}',True)
    shutil.move(f'./outputs',f'./models/{lang}')

    print(result)
    
    