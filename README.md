# DM
WSDM https://project-miracl.github.io/index.html#overview
## Baseline
複製資料集
```
git lfs install
git clone https://huggingface.co/datasets/miracl/miracl
```
以 lucene bm25 演算法產生 runfiles 於 `./runfiles` 資料夾中
```
python lucene_bm25_search.py dev
python lucene_bm25_search.py train
python lucene_bm25_search.py test-a
```
產生評分 (recall@100、nDCG@10) 於 `./evaluations` 資料夾中
```
python evaluate.py dev
python evaluate.py train
```