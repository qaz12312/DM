cmd='python -m pyserini.search.faiss --encoder-class auto --encoder castorini/mdpr-tied-pft-msmarco --topics miracl-v1.0-en-dev --index miracl-v1.0-en-mdpr-tied-pft-msmarco --output out.txt --batch 128 --threads 16 --hits 1000'
import os
os.system(cmd)