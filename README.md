# Multilingiual-Cross-Lingual-Fact-Check-Retrieval

#### Test 1 - Basic model execution and results (using default hyperparameters)
```
// Store the results from these model runs in a file clearly
// For training and testing BERT model
python src/train.py --model bert
python src/test.py --model bert

// For training and testing XLM-RoBERTa model
python src/train.py --model roberta
python src/test.py --model roberta

// For training and testing LmBERT model
python src/train.py --model lmbert
python src/test.py --model lmbert

// For training and testing LmBERT model with contrastive loss
python src/train.py --model lmbert_contrastive
python src/test.py --model lmbert_contrastive
```

#### Test 2 - Model fine tuning and optimizations
```
// Store the results from these model runs in a file clearly
// You can change hyperparameters as your wish
// Performance critical hyperparameter - margin - Try margin > 1 and < 1 also. 1 is default

// For training and testing LmBERT model with contrastive loss
python src/train.py --model lmbert_contrastive --hyperparameters "{'batch_size': 32, 'epochs': 10, 'margin': 0.3}"
python src/test.py --model lmbert_contrastive
```

#### Test 3 - Test on actual data for better results
```
// Store the results from these model runs in a file clearly
// THIS WILL TAKE SOME TIME FOR DATA LOADING AND PRE-PROCESSING
// You can change the hyperparameters as in Test 2

// Store the results from these model runs in a file clearly
// For training and testing BERT model
python src/train.py --model bert --use_actual --ratio 0.01
python src/test.py --model bert --use_actual --ratio 0.01

// For training and testing XLM-RoBERTa model
python src/train.py --model roberta --use_actual --ratio 0.01
python src/test.py --model roberta --use_actual --ratio 0.01

// For training and testing LmBERT model
python src/train.py --model lmbert --use_actual --ratio 0.01
python src/test.py --model lmbert --use_actual --ratio 0.01

// For training and testing LmBERT model with contrastive loss
python src/train.py --model lmbert_contrastive --use_actual --ratio 0.01
python src/test.py --model lmbert_contrastive --use_actual --ratio 0.01
```