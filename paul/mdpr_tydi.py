import os
def eachf(dsp):
    for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
        os.system(f'python -m pyserini.search.faiss \
            --encoder-class auto \
            --encoder castorini/mdpr-tied-pft-msmarco-ft-all \
            --topics miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv \
            --index miracl-v1.0-{lang}-mdpr-tied-pft-msmarco \
            --output rufiles/mdpr_tydi/{lang}_{dsp}.txt \
            --batch 128 --threads 16 --hits 1000')

for dsp in ['train','dev','test-a']:
    eachf(dsp)


