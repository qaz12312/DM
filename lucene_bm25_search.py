import os
import sys
data_split=sys.argv[1]
from pyserini.search.lucene import LuceneSearcher

def lucene_bm25_search(lang:str,data_split:str,k1:float,b:float,hits:int):
    index=LuceneSearcher.from_prebuilt_index(f'miracl-v1.0-{lang}').index_dir
    os.system(f'python -m pyserini.search.lucene --index {index} --topics ./miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{data_split}.tsv --output ./runfiles/{lang}_{data_split}.txt --bm25 --k1 {k1} --b {b} --hits {hits}')

if not os.path.exists('./runfiles'):
    os.mkdir('./runfiles')

for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    lucene_bm25_search(lang,data_split,1.6,0.75,100)