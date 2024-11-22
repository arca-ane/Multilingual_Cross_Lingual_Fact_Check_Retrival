#!/bin/bash

mkdir -p logs

# BERT
python src/train.py --model bert > logs/train_bert.log 2>&1
python src/test.py --model bert > logs/test_bert.log 2>&1

# XLM-RoBERTa
python src/train.py --model roberta > logs/train_roberta.log 2>&1
python src/test.py --model roberta > logs/test_roberta.log 2>&1

# LmBERT
python src/train.py --model lmbert > logs/train_lmbert.log 2>&1
python src/test.py --model lmbert > logs/test_lmbert.log 2>&1

# LmBERT with Contrastive Loss
python src/train.py --model lmbert_contrastive > logs/train_lmbert_contrastive.log 2>&1
python src/test.py --model lmbert_contrastive > logs/test_lmbert_contrastive.log 2>&1

