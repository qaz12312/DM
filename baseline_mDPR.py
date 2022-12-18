import os
for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
    # mDPR
    os.system(f'python -m pyserini.search.faiss \
        --encoder-class auto \
        --encoder castorini/mdpr-tied-pft-msmarco \
        --topics miracl-v1.0-{lang}-dev \
        --index miracl-v1.0-{lang}-mdpr-tied-pft-msmarco \
        --output run.miracl.mdpr-tied-pft-msmarco.{lang}.dev.txt \
        --batch 128 --threads 16 --hits 100')

    # mDPR evaluate recall@100
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -m recall.100 miracl-v1.0-{lang}-dev \
        run.miracl.mdpr-tied-pft-msmarco.{lang}.dev.txt')

    # mDPR evaluate nDCG@10
    os.system(f'python -m pyserini.eval.trec_eval \
        -c -M 100 -m ndcg_cut.10 miracl-v1.0-{lang}-dev \
        run.miracl.mdpr-tied-pft-msmarco.{lang}.dev.txt')
    