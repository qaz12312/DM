import os

def eachf(dsp):
    for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
        os.system(f'python -m pyserini.search.lucene \
            --language {lang} \
            --topics miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv \
            --index miracl-v1.0-{lang} \
            --output runfiles/bm25/{lang}_{dsp}.txt \
            --batch 128 --threads 16 --bm25 --hits 1000')

for dsp in ['train','dev','test-a']:
    eachf(dsp)


