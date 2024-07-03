# ⭐ GLiClass: Generalist and Lightweight Model for Sequence Classification

This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER/tree/main) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.

It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.

### Instalation:
```
pip install gliclass
```

### How to use:
```python3
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0] #because we have one text

for result in results:
 print(result["label"], "=>", result["score"])
```

### How to train:
Prepare training data in the following format:
[ {"text": "Some text here!", "all_labels": ["sport", "science", "business", ...], "true_labels":   ["other"]}, 
 ...
]

Specify your training parameters in the `train.py` script.
