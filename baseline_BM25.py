import os
for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    # BM25 runfiles
    os.system(f'python -m pyserini.search.lucene \
        --language {lang} \
        --topics miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv \
        --index miracl-v1.0-{lang} \
        --output run.miracl.bm25.{lang}.dev.txt \
        --batch 128 --threads 16 --bm25 --hits 100')
    
    # BM25 evaluate recall@100
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -m recall.100 miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv \
        run.miracl.bm25.{lang}.dev.txt')
    
    # BM25 evaluate nDCG@10
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -M 100 -m ndcg_cut.10 miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv \
        run.miracl.bm25.{lang}.dev.txt')