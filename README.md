# DM
WSDM https://project-miracl.github.io/index.html#overview
+ 修改 `utils.py` 裡的 `PROJECT_PATH`、`RESULT_PATH` 路徑
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

## Baseline (mDPR)
https://castorini.github.io/pyserini/2cr/miracl.html
```
git lfs install
git clone https://huggingface.co/datasets/miracl/miracl
python baseline_mDPR.py
```
|lang|recall@100|nDCG@10|
|-|-|-|
|ar|0.8407|0.4993|
|bn|0.8193|0.4427|
|en|||
|es|0.8643|0.4777|
|fa|0.8980|0.4800|
|fi|0.7877|0.4721|
|fr|0.9154|0.4352|
|hi|0.7755|0.3830|
|id|0.5734|0.2719|
|ja|0.8254|0.4390|
|ko|0.7369|0.4189|
|ru|0.7972|0.4073|
|sw|0.6158|0.2990|
|te|0.7619|0.3557|
|th|0.6783|0.3578|
|zh|0.9436|0.5116|


+ `en`: faiss MemoryError: std::bad_alloc
