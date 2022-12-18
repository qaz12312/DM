# DM
WSDM https://project-miracl.github.io/index.html#overview
## Baseline (BM25)
https://castorini.github.io/pyserini/2cr/miracl.html
```
git lfs install
git clone https://huggingface.co/datasets/miracl/miracl
python baseline_BM25.py
```
|lang|recall@100|nDCG@10|
|-|-|-|
|ar|0.8885|0.4809|
|bn|0.9088|0.5079|
|en|0.8190|0.3506|
|es|0.7018|0.3193|
|fa|0.7306|0.3334|
|fi|0.8910|0.5513|
|fr|0.6528|0.1832|
|hi|0.8679|0.4578|
|id|0.9041|0.4486|
|ja|0.8048|0.3689|
|ko|0.7831|0.4190|
|ru|0.6614|0.3342|
|sw|0.7008|0.3826|
|te|0.8307|0.4942|
|th|0.8874|0.4838|
|zh|0.5599|0.1801|
## BM25 (k1=1.6, b=0.75)
```
git lfs install
git clone https://huggingface.co/datasets/miracl/miracl
python baseline_BM25_k1-1.6_b-0.75.py
```
|lang|recall@100|nDCG@10|
|-|-|-|
|ar|0.8061|0.3773|
|bn|0.8796|0.4388|
|en|0.7044|0.2557|
|es|0.6085|0.2288|
|fa|0.6848|0.2676|
|fi|0.8552|0.4634|
|fr|0.4912|0.1354|
|hi|0.8261|0.4065|
|id|0.8142|0.3475|
|ja|0.7340|0.2751|
|ko|0.7526|0.3537|
|ru|0.5390|0.2209|
|sw|0.6468|0.3173|
|te|0.7341|0.3547|
|th|0.8477|0.4134|
|zh|0.4965|0.1348|
## Baseline (mDPR)
https://castorini.github.io/pyserini/2cr/miracl.html
```
git lfs install
git clone https://huggingface.co/datasets/miracl/miracl
python baseline_mDPR.py
```
|lang|recall@100|nDCG@10|
|-|-|-|
|ar|
|bn|
|en|
|es|
|fa|
|fi|
|fr|
|hi|
|id|
|ja|
|ko|
|ru|
|sw|
|te|
|th|
|zh|
