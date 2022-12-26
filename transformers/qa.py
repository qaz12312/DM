from transformers import pipeline
from googletrans import Translator
import pandas as pd
import asyncio
import async_google_trans_new
from sklearn.metrics import classification_report
from pyserini.search.lucene import LuceneSearcher
import json

def ensure_future(func):
    def wrapper(*argv,**kargv):
        return asyncio.ensure_future(func(*argv,**kargv))
    return wrapper

@ensure_future
async def translate(text:str):
    g = async_google_trans_new.AsyncTranslator()
    return await g.translate(text,"en")

def run(tasks):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    return [t.result() for t in tasks]

def translates(li):
    return run([translate(lli) for lli in li])

def get_dataset(lang:str,dsp:str,searcher:LuceneSearcher):
    tp_path=f'miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv'
    q_path=f'miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{dsp}.tsv'
    t=Translator()
    with open(tp_path,'r',encoding='utf-8') as f:
        tp=[t.split('\t') for t in f.readlines()]
        tp={tid:txt for tid,txt in tp}

    topics=[]
    docs=[]
    rels=[] 
    tids=[]
    docids=[]       
    with open(q_path,'r',encoding='utf-8') as f:
        for li in f.readlines():
            tid,_,docid,rel=li.split('\t')
            tids.append(tid)
            docids.append(docid)
            doc=json.loads(searcher.doc(docid).raw())['text']
            topic=tp[tid]
            topics.append(topic)
            docs.append(doc)
            rels.append(int(rel))

    #topics=translates(topics)
    #docs=translates(docs)

    return list(zip(tids,topics,docids,docs,rels))

qa_model = pipeline("question-answering")

for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    searcher = LuceneSearcher.from_prebuilt_index(f'miracl-v1.0-{lang}')
    train_data=get_dataset(lang,'dev',searcher)
    results=[]
    trues=[]
    
    cnt=0
    for _,query,_,doc,rel in train_data:
        if qa_model(question = query, context = doc)['score']>0.3:
            results.append(1)
        else:
            results.append(0)
        
        trues.append(rel)
        cnt+=1
        print(cnt)

    print(f'{lang}:')
    print(classification_report(trues,results))
