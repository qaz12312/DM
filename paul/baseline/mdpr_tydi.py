import os
#from ..evalute import evalute
def eachf(dsp,langs):
    for lang in langs:
        os.system(f'python -m pyserini.search.faiss \
            --encoder-class auto \
            --encoder castorini/mdpr-tied-pft-msmarco-ft-all \
            --topics miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv \
            --index miracl-v1.0-{lang}-mdpr-tied-pft-msmarco \
            --output runfiles/mdpr_tydi/{lang}_{dsp}.txt \
            --batch 128 --threads 16 --hits 1000')
    #evalute(dsp,'mdpr_tydi',langs)


for dsp in ['dev','test-a']:
    langs=['es','fr']
    eachf(dsp,langs)


