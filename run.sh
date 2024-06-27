#!/bin/bash

# download embedding model
git clone https://www.modelscope.cn/AI-ModelScope/bge-small-zh-v1.5.git demo/BAAI/bge-small-zh-v1.5

# download bge reranker model
git clone https://www.modelscope.cn/Xorbits/bge-reranker-base.git demo/BAAI/bge-reranker-base

# download nltk_data
git clone https://github.com/nltk/nltk_data.git
cp -r nltk_data/packages/. /root/

# download dataset
git clone https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset.git demo/dataset

mv demo/dataset/question.jsonl demo/question.jsonl

unzip demo/dataset/data.zip -d demo/

# print info 
echo "Please close current terminal and you can now play with demo!"