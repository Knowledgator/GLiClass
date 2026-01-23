# ⭐ GLiClass: Generalist and Lightweight Model for Sequence Classification

**GLiClass** is an efficient, zero-shot sequence classification model inspired by the [GLiNER](https://github.com/urchade/GLiNER/tree/main) framework. It achieves comparable performance to traditional cross-encoder models while being significantly more computationally efficient, offering classification results approximately **10 times faster** by performing classification in a single forward pass.

<p align="center">
    <a href="https://medium.com/@knowledgrator/pushing-zero-shot-classification-to-the-limit-696a2403032f">📄 Blog</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/dkyeAgs9DG">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/knowledgator/GLiClass_SandBox">📺 Demo</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?sort=trending&search=gliclass">🤗 Available models</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://colab.research.google.com/github/Knowledgator/GLiClass/blob/main/finetuning.ipynb">
        <img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />
    </a>
</p>

### 🚀 Quick Start

Install GLiClass easily using pip:

```bash
pip install gliclass
```

#### Install from Source

Clone and install directly from GitHub:

```bash
git clone https://github.com/Knowledgator/GLiClass
cd GLiClass

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install .
```

Verify your installation:

```python
import gliclass
print(gliclass.__version__)
```

### 🧑‍💻 Usage Example

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device='cuda:0'
)

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0]

for result in results:
    print(f"{result['label']} => {result['score']:.3f}")
```

### 🔥 New Features

#### Hierarchical Labels

GLiClass now supports hierarchical label structures using dot notation:

```python
hierarchical_labels = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["product", "service", "shipping"]
}

text = "The product quality is amazing but delivery was slow"
results = pipeline(text, hierarchical_labels, threshold=0.5)[0]

for result in results:
    print(f"{result['label']} => {result['score']:.3f}")
# Output:
# sentiment.positive => 0.892
# topic.product => 0.921
# topic.shipping => 0.763
```

Get hierarchical output matching your input structure:

```python
results = pipeline(text, hierarchical_labels, return_hierarchical=True)[0]
print(results)
# Output:
# {
#     "sentiment": {"positive": 0.892, "negative": 0.051, "neutral": 0.124},
#     "topic": {"product": 0.921, "service": 0.153, "shipping": 0.763}
# }
```

#### Few-Shot Examples

Improve classification accuracy with in-context examples using the `<<EXAMPLE>>` token:

```python
examples = [
    {
        "text": "Love this item, great quality!",
        "labels": ["positive", "product"]
    },
    {
        "text": "Customer support was unhelpful",
        "labels": ["negative", "service"]
    }
]

text = "Fast delivery and the item works perfectly!"
labels = ["positive", "negative", "product", "service", "shipping"]

results = pipeline(text, labels, examples=examples, threshold=0.5)[0]

for result in results:
    print(f"{result['label']} => {result['score']:.3f}")
```

#### Task Description Prompts

Add custom prompts to guide the classification task:

```python
text = "The battery life on this phone is incredible"
labels = ["positive", "negative", "neutral"]

results = pipeline(
    text,
    labels,
    prompt="Classify the sentiment of this product review:",
    threshold=0.5
)[0]
```

Use per-text prompts for batch processing:

```python
texts = ["Review about electronics", "Review about clothing"]
prompts = [
    "Analyze this electronics review:",
    "Analyze this clothing review:"
]

results = pipeline(texts, labels, prompt=prompts)
```

#### Long Document Classification

Process long documents with automatic text chunking:

```python
from gliclass import ZeroShotClassificationWithChunkingPipeline

chunking_pipeline = ZeroShotClassificationWithChunkingPipeline(
    model,
    tokenizer,
    text_chunk_size=8192,
    text_chunk_overlap=256,
    labels_chunk_size=8
)

long_document = "..." # Very long text
labels = ["category1", "category2", "category3"]

results = chunking_pipeline(long_document, labels, threshold=0.5)
```

### 🌟 Retrieval-Augmented Classification (RAC)

With new models trained with retrieval-agumented classification, such as [this model](https://huggingface.co/knowledgator/gliclass-base-v2.0-rac-init) you can specify examples to improve classification accuracy:

```python
example = {
    "text": "A new machine learning platform automates complex data workflows but faces integration issues.",
    "all_labels": ["AI", "automation", "data_analysis", "usability", "integration"],
    "true_labels": ["AI", "integration", "automation"]
}

text = "The new AI-powered tool streamlines data analysis but has limited integration capabilities."
labels = ["AI", "automation", "data_analysis", "usability", "integration"]

results = pipeline(text, labels, threshold=0.1, rac_examples=[example])[0]

for predict in results:
    print(f"{predict['label']} => {predict['score']:.3f}")
```

### 🎯 Key Use Cases

- **Sentiment Analysis:** Rapidly classify texts as positive, negative, or neutral.
- **Document Classification:** Efficiently organize and categorize large document collections.
- **Search Results Re-ranking:** Improve relevance and precision by reranking search outputs.
- **News Categorization:** Automatically tag and organize news articles into predefined categories.
- **Fact Checking:** Quickly validate and categorize statements based on factual accuracy.

### 🛠️ How to Train

Prepare your training data as follows:

```json
[
  {"text": "Sample text.", "all_labels": ["sports", "science", "business"], "true_labels": ["sports"]},
  ...
]
```

Optionally, specify confidence scores explicitly:

```json
[
  {"text": "Sample text.", "all_labels": ["sports", "science"], "true_labels": {"sports": 0.9}},
  ...
]
```

Please, refer to the `train.py` script to set up your training from scratch or fine-tune existing models.

### ⚙️ Advanced Configuration

#### Architecture Types

GLiClass supports multiple architecture types:

- **uni-encoder**: Single encoder for both text and labels (default, most efficient)
- **bi-encoder**: Separate encoders for text and labels
- **bi-encoder-fused**: Bi-encoder with label embeddings fused into text encoding
- **encoder-decoder**: Encoder-decoder architecture for sequence-to-sequence tasks

```python
from gliclass import GLiClassBiEncoder

# Load a bi-encoder model
model = GLiClassBiEncoder.from_pretrained("knowledgator/gliclass-biencoder-v1.0")
```

#### Pooling Strategies

Configure how token embeddings are pooled:

- `first`: First token (CLS token)
- `avg`: Average pooling
- `max`: Max pooling
- `last`: Last token
- `sum`: Sum pooling
- `rms`: Root mean square pooling
- `abs_max`: Max of absolute values
- `abs_avg`: Average of absolute values

```python
from gliclass import GLiClassModelConfig

config = GLiClassModelConfig(
    pooling_strategy='avg',
    class_token_pooling='average'  # or 'first'
)
```

#### Scoring Mechanisms

Choose different scoring mechanisms for classification:

- `simple`: Dot product (fastest)
- `weighted-dot`: Weighted dot product with learned projections
- `mlp`: Multi-layer perceptron scorer
- `hopfield`: Hopfield network-based scorer

```python
config = GLiClassModelConfig(
    scorer_type='mlp'
)
```

#### Flash Attention Backends

Enable Flash Attention for faster inference (requires additional packages):

```python
# Install flash backends
pip install flashdeberta  # For DeBERTa models
pip install turbot5       # For T5 models

# Enable in config
config = GLiClassModelConfig(
    use_flash=True
)
```


## 📚 Citations

If you find GLiClass useful in your research or project, please cite our papers:


```bibtex
@misc{stepanov2025gliclassgeneralistlightweightmodel,
      title={GLiClass: Generalist Lightweight Model for Sequence Classification Tasks}, 
      author={Ihor Stepanov and Mykhailo Shtopko and Dmytro Vodianytskyi and Oleksandr Lukashov and Alexander Yavorskyi and Mykyta Yaroshenko},
      year={2025},
      eprint={2508.07662},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.07662}, 
}
```