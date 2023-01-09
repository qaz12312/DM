CONDA_ENV_NAME='dm_py38'

conda create -n $CONDA_ENV_NAME python=3.8
conda activate $CONDA_ENV_NAME

conda install -c conda-forge openjdk=11
conda install -c conda-forge maven # Use Maven as Java project management


git clone https://github.com/castorini/pyserini.git --recurse-submodules
cd pyserini
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../.. # 解開　tar 檔、顯示過程、指定檔名、顯示建立過程:解開壓縮檔的檔案到現行目錄
cd tools/eval/ndeval && make && cd ../../..
# Install project with pip in editable mode 在當前資料夾中尋找setup.py並在「編輯」或「開發」模式下安裝
## 編輯模式是指當改變本機的程式後，只需要重新安裝專案 就可以重新套用專案的設定檔
pip install -e .
python -m unittest
cd .. # back to project root # confirm everything is working by running the unit tests


# Install other packages
pip install torch
# faiss 有 cpu 跟 gpu 版本
conda install faiss-cpu -c pytorch # faiss 用於高效地進行類似性搜索的 C++ 庫


# Install Spacy English model
## m: 把模塊當成腳本來運行
python -m spacy download en_core_web_sm


# Install jdk and maven for buding Anserini for pyserini/resources/jars/…
sudo apt-get update && sudo apt-get install -y openjdk-11-jdk-headless maven -qq # -qq 選項是 "quiet" 的縮寫，用於指定在安裝軟體包時不顯示任何輸出信息


# Install dependency
pip install Cython pyjnius


# If all goes well, you should be able to see anserini-X.Y.Z-SNAPSHOT-fatjar.jar in target/:
git clone https://github.com/castorini/anserini.git --recurse-submodules
cd anserini
# mvn clean package appassembler:assemble -DskipTests -Dmaven.javadoc.skip=true ######
mvn clean package appassembler:assemble
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz
cd trec_eval.9.0.4 && make
cd ../../../..

# move the jar files to Pyserini project: target/anserini-X.Y.Z-SNAPSHOT-fatjar.jar into pyserini/resources/jars/
cp -r ./anserini/target/* ./pyserini/pyserini/resources/jars

# Install datasets
git clone https://huggingface.co/datasets/miracl/miracl