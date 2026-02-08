import torch
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any, Tuple
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from .model import GLiClassModel, GLiClassBiEncoder
from .utils import retrieval_augmented_text


def flatten_hierarchical_labels(
    labels: Union[List[str], Dict[str, Any]],
    prefix: str = "",
    separator: str = "."
) -> List[str]:
    """
    Flatten hierarchical labels into dot notation.
    
    Supports arbitrary nesting depth. Examples:
    
    Input: {"sentiment": ["positive", "negative", "neutral"], "topic": ["product", "service", "shipping"]}
    Output: ["sentiment.positive", "sentiment.negative", "sentiment.neutral", 
             "topic.product", "topic.service", "topic.shipping"]
    
    Input: {
        "category": {
            "electronics": ["phone", "laptop"],
            "clothing": ["shirt", "pants"]
        }
    }
    Output: [
        "category.electronics.phone",
        "category.electronics.laptop", 
        "category.clothing.shirt",
        "category.clothing.pants"
    ]
    
    Input: ["label1", "label2"]  # Already flat
    Output: ["label1", "label2"]
    
    Args:
        labels: Either a list of string labels or a hierarchical dict
        prefix: Current prefix for recursion (internal use)
        separator: Separator to use between hierarchy levels (default: ".")
        
    Returns:
        List of flattened label strings with dot notation
    """
    if isinstance(labels, list):
        if prefix:
            return [f"{prefix}{separator}{label}" for label in labels]
        return labels
    
    elif isinstance(labels, dict):
        flattened = []
        for key, value in labels.items():
            new_prefix = f"{prefix}{separator}{key}" if prefix else key
            flattened.extend(flatten_hierarchical_labels(value, new_prefix, separator))
        return flattened
    
    elif isinstance(labels, str):
        if prefix:
            return [f"{prefix}{separator}{labels}"]
        return [labels]
    
    else:
        raise ValueError(f"Unsupported label type: {type(labels)}. Expected list, dict, or str.")


def build_hierarchical_output(
    predictions: List[Dict[str, float]],
    original_labels: Union[List[str], Dict[str, Any]],
    separator: str = ".",
    all_scores: Optional[Dict[str, float]] = None
) -> Union[Dict[str, float], Dict[str, Any]]:
    """
    Build hierarchical output structure matching the input labels structure.
    
    Args:
        predictions: List of prediction dicts with 'label' and 'score'
        original_labels: Original hierarchical labels structure
        separator: Separator used in flattened labels
        all_scores: Optional dict of all label scores (for complete output)
        
    Returns:
        Hierarchical structure with scores matching the input format
        
    Example:
        Input predictions: [
            {'label': 'sentiment.positive', 'score': 0.85},
            {'label': 'topic.product', 'score': 0.72}
        ]
        Input original_labels: {
            "sentiment": ["positive", "negative", "neutral"],
            "topic": ["product", "service", "shipping"]
        }
        Output: {
            "sentiment": {"positive": 0.85, "negative": 0.0, "neutral": 0.0},
            "topic": {"product": 0.72, "service": 0.0, "shipping": 0.0}
        }
    """
    score_lookup = {pred['label']: pred['score'] for pred in predictions}
    
    if all_scores:
        for k, v in all_scores.items():
            if k not in score_lookup:
                score_lookup[k] = v
    
    def _build_recursive(
        structure: Union[List[str], Dict[str, Any]], 
        prefix: str = ""
    ) -> Union[Dict[str, float], Dict[str, Any]]:
        if isinstance(structure, list):
            result = {}
            for label in structure:
                full_label = f"{prefix}{separator}{label}" if prefix else label
                result[label] = score_lookup.get(full_label, 0.0)
            return result
        
        elif isinstance(structure, dict):
            result = {}
            for key, value in structure.items():
                new_prefix = f"{prefix}{separator}{key}" if prefix else key
                result[key] = _build_recursive(value, new_prefix)
            return result
        
        elif isinstance(structure, str):
            full_label = f"{prefix}{separator}{structure}" if prefix else structure
            return {structure: score_lookup.get(full_label, 0.0)}
        
        return {}
    
    if isinstance(original_labels, list):
        return {label: score_lookup.get(label, 0.0) for label in original_labels}
    
    return _build_recursive(original_labels)


def format_examples_prompt(
    examples: List[Dict[str, Any]],
    example_token: str = "<<EXAMPLE>>",
    sep_token: str = "<<SEP>>"
) -> str:
    """
    Format few-shot examples into a prompt string using <<EXAMPLE>> token.

    Format matches training: <<EXAMPLE>>text \nLabels:\n label1, label2
    with a single <<SEP>> after all examples.

    Args:
        examples: List of example dicts with 'text' and 'labels'/'true_labels' keys
        example_token: Token to mark examples (default: "<<EXAMPLE>>")
        sep_token: Separator token after all examples (default: "<<SEP>>")

    Returns:
        Formatted examples string
    """
    if not examples:
        return ""

    formatted_parts = []
    for example in examples:
        text = example.get('text', '')
        labels = example.get('labels', example.get('true_labels', []))

        if isinstance(labels, list):
            labels_str = ', '.join(labels)
        else:
            labels_str = str(labels)

        # Match training format: " \nLabels:\n " instead of "\nLabels: "
        formatted_parts.append(f"{example_token}{text} \nLabels:\n {labels_str}")

    # Add single SEP token after all examples (matching training)
    formatted_parts.append(sep_token)

    return ''.join(formatted_parts)


class BaseZeroShotClassificationPipeline(ABC):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                 classification_type='multi-label', device='cuda:0', progress_bar=True,
                 label_separator: str = "."):
        self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_classes = max_classes
        self.classification_type = classification_type
        self.max_length = max_length
        self.progress_bar = progress_bar
        self.label_separator = label_separator
        
        self.example_token = "<<EXAMPLE>>"
        self.label_token = "<<LABEL>>"
        self.sep_token = "<<SEP>>"

        if not isinstance(device, torch.device):
            if torch.cuda.is_available() and 'cuda' in device:
                self.device = torch.device(device)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        if self.model.device != self.device:
            self.model.to(self.device)

        # Ensure model is in evaluation mode for inference
        self.model.eval()

    def _process_labels(
        self, 
        labels: Union[List[str], Dict[str, Any], List[List[str]], List[Dict[str, Any]]]
    ) -> Union[List[str], List[List[str]]]:
        """Process labels to handle hierarchical structures."""
        if not labels:
            return labels
        
        if isinstance(labels, dict):
            return flatten_hierarchical_labels(labels, separator=self.label_separator)
        
        if isinstance(labels, list):
            if len(labels) == 0:
                return labels
            
            first_elem = labels[0]
            
            if isinstance(first_elem, str):
                return labels
            
            if isinstance(first_elem, dict):
                return [
                    flatten_hierarchical_labels(lbl, separator=self.label_separator) 
                    for lbl in labels
                ]
            
            if isinstance(first_elem, list):
                if first_elem and isinstance(first_elem[0], dict):
                    return [
                        flatten_hierarchical_labels(lbl, separator=self.label_separator) 
                        for lbl in labels
                    ]
                return labels
        
        return labels

    def _format_examples_for_input(
        self, 
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Format few-shot examples using <<EXAMPLE>> and <<SEP>> tokens."""
        if not examples:
            return ""
        return format_examples_prompt(
            examples, 
            example_token=self.example_token,
            sep_token=self.sep_token
        )
    
    def _format_prompt(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        index: int = 0
    ) -> str:
        """Format the task description prompt."""
        if prompt is None:
            return ""
        
        if isinstance(prompt, str):
            return prompt
        
        if isinstance(prompt, list):
            if index < len(prompt):
                return prompt[index]
            return prompt[0] if prompt else ""
        
        return ""

    @abstractmethod
    def prepare_inputs(self, texts, labels, same_labels=False, examples=None, prompt=None):
        pass
    
    def _get_batch_examples(self, examples, start_idx, batch_size):
        """Get examples for current batch."""
        if not examples:
            return None
        if isinstance(examples[0], list):
            return examples[start_idx:start_idx + batch_size]
        return examples
    
    def _get_batch_prompt(self, prompt, start_idx, batch_size):
        """Get prompt for current batch."""
        if not prompt:
            return None
        if isinstance(prompt, list):
            return prompt[start_idx:start_idx + batch_size]
        return prompt
    
    @torch.no_grad()
    def get_embeddings(self, texts, labels, batch_size=8, examples=None, prompt=None):
        if isinstance(texts, str):
            texts = [texts]
        
        labels = self._process_labels(labels)
        
        if isinstance(labels[0], str):
            same_labels = True
        else:
            same_labels = False
        
        results = []

        iterable = range(0, len(texts), batch_size)
        if self.progress_bar:
            iterable = tqdm(iterable)

        for idx in iterable:
            batch_texts = texts[idx:idx+batch_size]
            batch_examples = self._get_batch_examples(examples, idx, len(batch_texts))
            batch_prompt = self._get_batch_prompt(prompt, idx, len(batch_texts))
            
            tokenized_inputs = self.prepare_inputs(
                batch_texts, labels, same_labels, 
                examples=batch_examples, prompt=batch_prompt
            )
            model_output = self.model(
                **tokenized_inputs, 
                output_text_embeddings=True,
                output_class_embeddings=True
            )
            logits = model_output.logits
            text_embeddings = model_output.text_embeddings
            class_embeddings = model_output.class_embeddings
            batch_size_actual = logits.shape[0]
            
            for i in range(batch_size_actual):
                result = {
                    'logits': logits[i].cpu().numpy(),
                    'text_embedding': text_embeddings[i].cpu().numpy(),
                    'class_embeddings': class_embeddings[i].cpu().numpy()
                }
                results.append(result)
        
        return results

    @torch.no_grad()
    def __call__(
        self, 
        texts: Union[str, List[str]], 
        labels: Union[List[str], Dict[str, Any], List[List[str]], List[Dict[str, Any]]],
        threshold: float = 0.5, 
        batch_size: int = 8, 
        rac_examples: Optional[List] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        return_hierarchical: bool = False
    ):
        """
        Perform zero-shot classification.
        
        Args:
            texts: Single text or list of texts to classify
            labels: Labels in various formats (flat list or hierarchical dict)
            threshold: Classification threshold for multi-label (default: 0.5)
            batch_size: Batch size for processing
            rac_examples: Retrieval augmented examples (legacy)
            examples: Few-shot examples with 'text' and 'labels'/'true_labels' keys
            prompt: Task description - string (same for all) or list (per-text)
            return_hierarchical: If True, return hierarchical structure with all scores
                
        Returns:
            List of classification results or hierarchical dict structure.
        """
        original_labels = labels
        
        if isinstance(texts, str):
            if rac_examples:
                texts = retrieval_augmented_text(texts, rac_examples)
            texts = [texts]
        else:
            if rac_examples:
                texts = [
                    retrieval_augmented_text(text, ex) 
                    for text, ex in zip(texts, rac_examples)
                ]
        
        processed_labels = self._process_labels(labels)
        
        if isinstance(processed_labels[0], str):
            same_labels = True
        else:
            same_labels = False
            
        results = []
        all_scores_list = []
        
        iterable = range(0, len(texts), batch_size)
        if self.progress_bar:
            iterable = tqdm(iterable)

        for idx in iterable:
            batch_texts = texts[idx:idx+batch_size]
            if not same_labels:
                batch_labels = processed_labels[idx:idx+batch_size]
            else:
                batch_labels = processed_labels
            
            batch_examples = self._get_batch_examples(examples, idx, len(batch_texts))
            batch_prompt = self._get_batch_prompt(prompt, idx, len(batch_texts))
            
            tokenized_inputs = self.prepare_inputs(
                batch_texts, batch_labels, same_labels, 
                examples=batch_examples, prompt=batch_prompt
            )
            model_output = self.model(**tokenized_inputs)
            logits = model_output.logits
            
            if self.classification_type == 'single-label':
                for i in range(len(batch_texts)):
                    score = torch.softmax(logits[i], dim=-1)
                    if same_labels:
                        curr_labels = batch_labels
                    else:
                        curr_labels = batch_labels[i]
                    
                    if return_hierarchical:
                        all_scores = {curr_labels[j]: score[j].item() for j in range(len(curr_labels))}
                        all_scores_list.append(all_scores)
                    
                    pred_label = curr_labels[torch.argmax(score).item()]
                    results.append([{'label': pred_label, 'score': score.max().item()}])
                    
            elif self.classification_type == 'multi-label':
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits)
                for i in range(len(batch_texts)):
                    text_results = []
                    if same_labels:
                        curr_labels = batch_labels
                    else:
                        curr_labels = batch_labels[i]
                    
                    if return_hierarchical:
                        all_scores = {curr_labels[j]: probs[i][j].item() for j in range(len(curr_labels))}
                        all_scores_list.append(all_scores)
                    
                    for j, prob in enumerate(probs[i][:len(curr_labels)]):
                        score = prob.item()
                        if score >= threshold:
                            text_results.append({'label': curr_labels[j], 'score': score})
                    results.append(text_results)
            else:
                raise ValueError(
                    "Unsupported classification type: choose 'single-label' or 'multi-label'"
                )
        
        if return_hierarchical:
            hierarchical_results = []
            for i, (result, all_scores) in enumerate(zip(results, all_scores_list)):
                if same_labels:
                    orig_lbl = original_labels
                else:
                    orig_lbl = original_labels[i] if i < len(original_labels) else original_labels
                
                hierarchical_results.append(
                    build_hierarchical_output(result, orig_lbl, self.label_separator, all_scores)
                )
            return hierarchical_results
        
        return results


class UniEncoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                 classification_type='multi-label', device='cuda:0', progress_bar=True,
                 label_separator: str = "."):
        super().__init__(
            model, tokenizer, max_classes, max_length, 
            classification_type, device, progress_bar, label_separator
        )

    def prepare_input(self, text, labels, examples=None, prompt=None):
        """
        Prepare input matching training format from data_processing.py:
        Order: Labels → SEP → Prompt → Text → Examples
        """
        input_parts = []

        # 1. Add labels
        for label in labels:
            label_tag = f"{self.label_token}{label}"
            input_parts.append(label_tag)
        input_parts.append(self.sep_token)

        # 2. Add task description prompt
        if prompt:
            input_parts.append(prompt)

        # 3. Format examples to go after text
        examples_str = ""
        if examples:
            examples_str = self._format_examples_for_input(examples)

        if self.model.config.prompt_first:
            return ''.join(input_parts) + text + examples_str
        else:
            return text + ''.join(input_parts) + examples_str

    def prepare_inputs(self, texts, labels, same_labels=False, examples=None, prompt=None):
        inputs = []

        if same_labels:
            for i, text in enumerate(texts):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples

                text_prompt = self._format_prompt(prompt, i)
                inputs.append(self.prepare_input(text, labels, text_examples, text_prompt))
        else:
            for i, (text, labels_) in enumerate(zip(texts, labels)):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples

                text_prompt = self._format_prompt(prompt, i)
                inputs.append(self.prepare_input(text, labels_, text_examples, text_prompt))

        tokenized_inputs = self.tokenizer(
            inputs, truncation=True,
            max_length=self.max_length,
            padding="longest", return_tensors="pt"
        ).to(self.device)

        return tokenized_inputs


class EncoderDecoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024,
                 classification_type='multi-label', device='cuda:0', progress_bar=True,
                 label_separator: str = "."):
        super().__init__(
            model, tokenizer, max_classes, max_length,
            classification_type, device, progress_bar, label_separator
        )

    def prepare_labels_prompt(self, labels, prompt=None):
        """Match training format: Labels → SEP → Prompt"""
        input_parts = []

        for label in labels:
            label_tag = f"{self.label_token}{label}"
            input_parts.append(label_tag)
        input_parts.append(self.sep_token)

        if prompt:
            input_parts.append(prompt)

        return ''.join(input_parts)

    def prepare_inputs(self, texts, labels, same_labels=False, examples=None, prompt=None):
        prompts = []
        processed_texts = []

        if same_labels:
            for i, text in enumerate(texts):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples
                text_prompt = self._format_prompt(prompt, i)
                prompts.append(self.prepare_labels_prompt(labels, text_prompt))
                examples_str = self._format_examples_for_input(text_examples) if text_examples else ""
                processed_texts.append(text + examples_str)
        else:
            for i, labels_ in enumerate(labels):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples
                text_prompt = self._format_prompt(prompt, i)
                prompts.append(self.prepare_labels_prompt(labels_, text_prompt))
                examples_str = self._format_examples_for_input(text_examples) if text_examples else ""
                processed_texts.append(texts[i] + examples_str)
        
        tokenized_inputs = self.tokenizer(
            processed_texts, truncation=True, 
            max_length=self.max_length, 
            padding="longest", return_tensors="pt"
        ).to(self.device)
        
        tokenized_classes = self.tokenizer(
            prompts, max_length=self.max_length, 
            truncation=True, padding="longest", return_tensors='pt'
        ).to(self.device)
        
        tokenized_inputs["class_input_ids"] = tokenized_classes["input_ids"]
        tokenized_inputs["class_attention_mask"] = tokenized_classes["attention_mask"]

        return tokenized_inputs

    
class BiEncoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                 classification_type='multi-label', device='cuda:0', progress_bar=True,
                 label_separator: str = "."):
        super().__init__(
            model, tokenizer, max_classes, max_length, 
            classification_type, device, progress_bar, label_separator
        )
        self.labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)

    def prepare_input(self, text, labels, examples=None, prompt=None):
        input_parts = []

        if prompt:
            input_parts.append(prompt)
            input_parts.append(" ")

        for label in labels:
            input_parts.append(self.label_token)
        input_parts.append(self.sep_token)

        examples_str = ""
        if examples:
            examples_str = self._format_examples_for_input(examples)

        if self.model.config.prompt_first:
            return ''.join(input_parts) + text + examples_str
        else:
            return text + ''.join(input_parts) + examples_str
    
    def prepare_inputs(self, texts, labels, same_labels=False, examples=None, prompt=None):
        if self.model.config.architecture_type == 'bi-encoder-fused':
            inputs = []
            if same_labels:
                for i, text in enumerate(texts):
                    text_examples = None
                    if examples:
                        if isinstance(examples[0], list):
                            text_examples = examples[i] if i < len(examples) else None
                        else:
                            text_examples = examples
                    text_prompt = self._format_prompt(prompt, i)
                    inputs.append(self.prepare_input(text, labels, text_examples, text_prompt))
            else:
                for i, (text, labels_) in enumerate(zip(texts, labels)):
                    text_examples = None
                    if examples:
                        if isinstance(examples[0], list):
                            text_examples = examples[i] if i < len(examples) else None
                        else:
                            text_examples = examples
                    text_prompt = self._format_prompt(prompt, i)
                    inputs.append(self.prepare_input(text, labels_, text_examples, text_prompt))
        else:
            inputs = []
            for i, text in enumerate(texts):
                text_prompt = self._format_prompt(prompt, i)
                if text_prompt:
                    inputs.append(f"{text_prompt} {text}")
                else:
                    inputs.append(text)
            
        if same_labels:
            tokenized_inputs = self.tokenizer(
                inputs, truncation=True,
                max_length=self.max_length,
                padding="longest", return_tensors="pt"
            ).to(self.device)

            tokenized_labels = self.labels_tokenizer(
                labels, truncation=True,
                max_length=self.max_length,
                padding="longest", return_tensors="pt"
            ).to(self.device)
            
            tokenized_inputs['class_input_ids'] = tokenized_labels['input_ids'].expand(
                len(texts), -1, -1
            )
            tokenized_inputs['class_attention_mask'] = tokenized_labels['attention_mask'].expand(
                len(texts), -1, -1
            )
            
            labels_mask = [[1 for _ in range(len(labels))] for _ in range(len(texts))]
            tokenized_inputs["labels_mask"] = torch.tensor(labels_mask).to(self.device)
        else:
            tokenized_inputs = self.tokenizer(
                inputs, truncation=True,
                max_length=self.max_length,
                padding="longest", return_tensors="pt"
            ).to(self.device)
            
            class_input_ids = []
            class_attention_mask = []
            
            for labels_set in labels:
                tokenized_labels = self.labels_tokenizer(
                    labels_set, truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                class_input_ids.append(tokenized_labels["input_ids"])
                class_attention_mask.append(tokenized_labels["attention_mask"])
            
            tokenized_inputs["class_input_ids"] = torch.stack(class_input_ids)
            tokenized_inputs["class_attention_mask"] = torch.stack(class_attention_mask)

            labels_mask = [[1 for _ in range(len(labels[j]))] for j in range(len(texts))]
            tokenized_inputs["labels_mask"] = torch.tensor(labels_mask).to(self.device)
        return tokenized_inputs


class ZeroShotClassificationPipeline:
    """
    Main pipeline class for zero-shot classification with GLiClass models.
    
    Supports:
    - Hierarchical labels with dot notation (e.g., {"sentiment": ["positive", "negative"]})
    - Few-shot examples with <<EXAMPLE>> token
    - Task description prompts
    - Hierarchical output format matching input structure
    
    Example usage:
    
    ```python
    from gliclass import ZeroShotClassificationPipeline
    
    pipeline = ZeroShotClassificationPipeline(model, tokenizer)
    
    # === Hierarchical Labels for Review Classification ===
    hierarchical_labels = {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["product", "service", "shipping"]
    }
    
    # Basic classification
    results = pipeline(
        "The product quality is amazing but delivery was slow",
        hierarchical_labels
    )
    # Results: [
    #     {'label': 'sentiment.positive', 'score': 0.89},
    #     {'label': 'topic.product', 'score': 0.92},
    #     {'label': 'topic.shipping', 'score': 0.76}
    # ]
    
    # === With Task Description Prompt ===
    results = pipeline(
        "The product quality is amazing but delivery was slow",
        hierarchical_labels,
        prompt="Classify this customer review by sentiment and topic:"
    )
    
    # === With Few-Shot Examples (uses <<EXAMPLE>> token) ===
    examples = [
        {
            "text": "Love this item, great quality!",
            "labels": ["sentiment.positive", "topic.product"]
        },
        {
            "text": "Customer support was unhelpful and rude",
            "labels": ["sentiment.negative", "topic.service"]
        },
        {
            "text": "Package arrived damaged after 2 weeks",
            "labels": ["sentiment.negative", "topic.shipping"]
        }
    ]
    
    results = pipeline(
        "Fast delivery and the item works perfectly!",
        hierarchical_labels,
        examples=examples,
        prompt="Classify customer feedback:"
    )
    
    # === Hierarchical Output (matches input structure) ===
    results = pipeline(
        "The product quality is amazing but delivery was slow",
        hierarchical_labels,
        return_hierarchical=True
    )
    # Returns:
    # {
    #     "sentiment": {
    #         "positive": 0.89,
    #         "negative": 0.05,
    #         "neutral": 0.12
    #     },
    #     "topic": {
    #         "product": 0.92,
    #         "service": 0.15,
    #         "shipping": 0.76
    #     }
    # }
    
    # === Per-Text Prompts ===
    results = pipeline(
        ["Electronics review text", "Clothing review text"],
        hierarchical_labels,
        prompt=["Analyze this electronics review:", "Analyze this clothing review:"]
    )
    ```
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        max_classes: int = 25, 
        max_length: int = 1024, 
        classification_type: str = 'multi-label', 
        device: str = 'cuda:0', 
        progress_bar: bool = True,
        label_separator: str = "."
    ):
        """
        Initialize the classification pipeline.
        
        Args:
            model: GLiClass model or path to model
            tokenizer: Tokenizer or path to tokenizer
            max_classes: Maximum number of classes to process
            max_length: Maximum sequence length
            classification_type: 'single-label' or 'multi-label'
            device: Device to run inference on
            progress_bar: Whether to show progress bar
            label_separator: Separator for hierarchical label notation (default: ".")
        """
        if isinstance(model, str):
            model = GLiClassBiEncoder.from_pretrained(model)
            
        self.label_separator = label_separator
            
        if model.config.architecture_type == 'uni-encoder':
            self.pipe = UniEncoderZeroShotClassificationPipeline(
                model, tokenizer, max_classes, max_length, 
                classification_type, device, progress_bar, label_separator
            )
        elif model.config.architecture_type in {'encoder-decoder'}:
            self.pipe = EncoderDecoderZeroShotClassificationPipeline(
                model, tokenizer, max_classes, max_length, 
                classification_type, device, progress_bar, label_separator
            )
        elif model.config.architecture_type in {'bi-encoder', 'bi-encoder-fused'}:
            self.pipe = BiEncoderZeroShotClassificationPipeline(
                model, tokenizer, max_classes, max_length, 
                classification_type, device, progress_bar, label_separator
            )
        else:
            raise NotImplementedError("This architecture is not implemented")
    
    def flatten_labels(
        self, 
        labels: Union[List[str], Dict[str, Any]]
    ) -> List[str]:
        """
        Flatten hierarchical labels to dot notation.
        
        Example:
            >>> pipeline.flatten_labels({
            ...     "sentiment": ["positive", "negative", "neutral"],
            ...     "topic": ["product", "service", "shipping"]
            ... })
            ["sentiment.positive", "sentiment.negative", "sentiment.neutral",
             "topic.product", "topic.service", "topic.shipping"]
        """
        return flatten_hierarchical_labels(labels, separator=self.label_separator)
    
    def get_embeddings(self, *args, **kwargs):
        """Get embeddings for texts and labels."""
        return self.pipe.get_embeddings(*args, **kwargs)
    
    def __call__(
        self, 
        texts: Union[str, List[str]], 
        labels: Union[List[str], Dict[str, Any], List[List[str]], List[Dict[str, Any]]],
        threshold: float = 0.5, 
        batch_size: int = 8, 
        rac_examples: Optional[List] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        return_hierarchical: bool = False
    ):
        """
        Perform zero-shot classification.
        
        Args:
            texts: Single text or list of texts to classify
            labels: Labels - flat list or hierarchical dict
                Examples:
                - ["positive", "negative"] - flat labels
                - {"sentiment": ["positive", "negative"], "topic": ["product", "service"]}
            threshold: Classification threshold for multi-label (default: 0.5)
            batch_size: Batch size for processing
            rac_examples: Retrieval augmented examples (legacy)
            examples: Few-shot examples, each with 'text' and 'labels' keys
            prompt: Task description - string or list of strings (per-text)
            return_hierarchical: If True, return structure matching input labels
            
        Returns:
            List of predictions (flat) or hierarchical dicts with all scores
        """
        return self.pipe(
            texts, labels, 
            threshold=threshold, 
            batch_size=batch_size, 
            rac_examples=rac_examples,
            examples=examples,
            prompt=prompt,
            return_hierarchical=return_hierarchical
        )


class ZeroShotClassificationWithChunkingPipeline(BaseZeroShotClassificationPipeline):
    """Pipeline with long text chunking support."""
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        max_classes: int = 25, 
        max_length: int = 1024, 
        classification_type: str = 'multi-label', 
        device: str = 'cuda:0', 
        progress_bar: bool = True,
        text_chunk_size: int = 8192, 
        text_chunk_overlap: int = 256, 
        labels_chunk_size: int = 8,
        label_separator: str = "."
    ):
        if isinstance(model, str):
            model = GLiClassModel.from_pretrained(model)
        super().__init__(
            model, tokenizer, max_classes, max_length, 
            classification_type, device, progress_bar, label_separator
        )
        
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap
        self.labels_chunk_size = labels_chunk_size

    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = self.text_chunk_size
        if overlap is None:
            overlap = self.text_chunk_overlap
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
            
        return chunks

    def prepare_input(self, text, labels, examples=None, prompt=None):
        """
        Prepare input matching training format from data_processing.py:
        Order: Labels → SEP → Prompt → Text → Examples
        """
        input_parts = []

        # 1. Add labels
        for label in labels:
            label_tag = f"{self.label_token}{label}"
            input_parts.append(label_tag)
        input_parts.append(self.sep_token)

        # 2. Add task description prompt
        if prompt:
            input_parts.append(prompt)

        # 3. Format examples to go after text
        examples_str = ""
        if examples:
            examples_str = self._format_examples_for_input(examples)

        if self.model.config.prompt_first:
            return ''.join(input_parts) + text + examples_str
        else:
            return text + ''.join(input_parts) + examples_str

    def prepare_inputs(self, texts, labels, same_labels=False, examples=None, prompt=None):
        inputs = []

        if same_labels:
            for i, text in enumerate(texts):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples
                text_prompt = self._format_prompt(prompt, i)
                inputs.append(self.prepare_input(text, labels, text_examples, text_prompt))
        else:
            for i, (text, labels_) in enumerate(zip(texts, labels)):
                text_examples = None
                if examples:
                    if isinstance(examples[0], list):
                        text_examples = examples[i] if i < len(examples) else None
                    else:
                        text_examples = examples
                text_prompt = self._format_prompt(prompt, i)
                inputs.append(self.prepare_input(text, labels_, text_examples, text_prompt))

        tokenized_inputs = self.tokenizer(
            inputs, truncation=True,
            max_length=self.max_length,
            padding="longest", return_tensors="pt"
        ).to(self.device)
        return tokenized_inputs

    def aggregate_chunk_scores(
        self, 
        chunk_scores: List[Dict[str, float]], 
        labels: List[str]
    ) -> Dict[str, float]:
        """Aggregate scores across text chunks using max pooling."""
        aggregated = {label: 0.0 for label in labels}
        
        for scores in chunk_scores:
            for label, score in scores.items():
                aggregated[label] = max(aggregated[label], score)
                
        return aggregated

    @torch.no_grad()
    def process_single_text(self, text, labels, threshold=0.5, examples=None, prompt=None):
        """Process a single long text through chunks."""
        text_chunks = self.chunk_text(text)
        
        all_chunk_scores = []
        
        for text_chunk in text_chunks:
            chunk_logits = []
            all_labels = []
            
            for labels_idx in range(0, len(labels), self.labels_chunk_size):
                curr_labels = labels[labels_idx:labels_idx + self.labels_chunk_size]
                if labels_idx == 0:
                    all_labels = []
                all_labels.extend(curr_labels)
                
                tokenized_inputs = self.prepare_inputs(
                    [text_chunk], curr_labels, same_labels=True, 
                    examples=examples, prompt=prompt
                )
                model_output = self.model(**tokenized_inputs)
                logits = model_output.logits
                
                chunk_logits.extend(logits[0][:len(curr_labels)].tolist())
            
            text_logits = torch.tensor(chunk_logits)
            
            if self.classification_type == 'single-label':
                scores = torch.softmax(text_logits, dim=-1)
            else:
                scores = torch.sigmoid(text_logits)
            
            chunk_score_dict = {
                label: scores[i].item() for i, label in enumerate(all_labels)
            }
            all_chunk_scores.append(chunk_score_dict)
        
        aggregated_scores = self.aggregate_chunk_scores(all_chunk_scores, labels)
        
        if self.classification_type == 'single-label':
            total = sum(aggregated_scores.values())
            if total > 0:
                aggregated_scores = {k: v / total for k, v in aggregated_scores.items()}
            
            best_label = max(aggregated_scores, key=aggregated_scores.get)
            return [{'label': best_label, 'score': aggregated_scores[best_label]}], aggregated_scores
        
        else:
            text_results = []
            for label, score in aggregated_scores.items():
                if score >= threshold:
                    text_results.append({'label': label, 'score': score})
            text_results.sort(key=lambda x: x['score'], reverse=True)
            return text_results, aggregated_scores

    @torch.no_grad()
    def __call__(
        self, 
        texts, 
        labels, 
        threshold=0.5, 
        batch_size=8,
        labels_chunk_size=None, 
        text_chunk_size=None, 
        text_chunk_overlap=None,
        rac_examples=None,
        examples=None,
        prompt=None,
        return_hierarchical: bool = False
    ):
        """Classification with chunking for long texts."""
        original_labels = labels
        
        if labels_chunk_size is not None:
            self.labels_chunk_size = labels_chunk_size
        if text_chunk_size is not None:
            self.text_chunk_size = text_chunk_size
        if text_chunk_overlap is not None:
            self.text_chunk_overlap = text_chunk_overlap

        if isinstance(texts, str):
            if rac_examples:
                texts = retrieval_augmented_text(texts, rac_examples)
            texts = [texts]
        else:
            if rac_examples:
                texts = [
                    retrieval_augmented_text(text, ex) 
                    for text, ex in zip(texts, rac_examples)
                ]

        labels = self._process_labels(labels)

        short_texts, short_indices = [], []
        long_texts, long_indices = [], []
        
        for i, text in enumerate(texts):
            if len(text) <= self.text_chunk_size:
                short_texts.append(text)
                short_indices.append(i)
            else:
                long_texts.append(text)
                long_indices.append(i)

        results = [None] * len(texts)
        all_scores_list = [None] * len(texts)

        if short_texts:
            iterable = range(0, len(short_texts), batch_size)
            if self.progress_bar:
                iterable = tqdm(iterable, desc="Processing short texts")

            for idx in iterable:
                batch_texts = short_texts[idx:idx + batch_size]
                batch_indices = short_indices[idx:idx + batch_size]

                all_logits = [[] for _ in range(len(batch_texts))]
                all_labels = []

                for labels_idx in range(0, len(labels), self.labels_chunk_size):
                    curr_labels = labels[labels_idx:labels_idx + self.labels_chunk_size]
                    if labels_idx == 0:
                        all_labels = []
                    all_labels.extend(curr_labels)

                    batch_prompt = self._get_batch_prompt(prompt, idx, len(batch_texts))
                    tokenized_inputs = self.prepare_inputs(
                        batch_texts, curr_labels, same_labels=True, 
                        examples=examples, prompt=batch_prompt
                    )
                    model_output = self.model(**tokenized_inputs)
                    logits = model_output.logits

                    for i in range(len(batch_texts)):
                        all_logits[i].extend(logits[i][:len(curr_labels)].tolist())

                for i, orig_idx in enumerate(batch_indices):
                    text_logits = torch.tensor(all_logits[i])

                    if self.classification_type == 'single-label':
                        score = torch.softmax(text_logits, dim=-1)
                        pred_idx = torch.argmax(score).item()
                        pred_label = all_labels[pred_idx]
                        results[orig_idx] = [{'label': pred_label, 'score': score[pred_idx].item()}]
                        all_scores_list[orig_idx] = {all_labels[j]: score[j].item() for j in range(len(all_labels))}

                    elif self.classification_type == 'multi-label':
                        probs = torch.sigmoid(text_logits)
                        text_results = []
                        all_scores_list[orig_idx] = {all_labels[j]: probs[j].item() for j in range(len(all_labels))}
                        for j, prob in enumerate(probs):
                            score_val = prob.item()
                            if score_val >= threshold:
                                text_results.append({'label': all_labels[j], 'score': score_val})
                        text_results.sort(key=lambda x: x['score'], reverse=True)
                        results[orig_idx] = text_results

        if long_texts:
            iterable = range(len(long_texts))
            if self.progress_bar:
                iterable = tqdm(iterable, desc="Processing long texts")

            for i in iterable:
                text = long_texts[i]
                orig_idx = long_indices[i]
                text_prompt = self._format_prompt(prompt, orig_idx)
                text_results, all_scores = self.process_single_text(
                    text, labels, threshold, examples=examples, prompt=text_prompt
                )
                results[orig_idx] = text_results
                all_scores_list[orig_idx] = all_scores

        if return_hierarchical:
            hierarchical_results = []
            for i, (result, all_scores) in enumerate(zip(results, all_scores_list)):
                hierarchical_results.append(
                    build_hierarchical_output(result, original_labels, self.label_separator, all_scores)
                )
            return hierarchical_results

        return results


# Utility functions

def parse_hierarchical_prediction(prediction: str, separator: str = ".") -> Dict[str, Any]:
    """Parse dot-notation prediction into hierarchy levels."""
    parts = prediction.split(separator)
    result = {'full': prediction}
    for i, part in enumerate(parts):
        result[f'level_{i}'] = part
    return result


def group_predictions_by_hierarchy(
    predictions: List[Dict[str, Any]], 
    separator: str = "."
) -> Dict[str, List[Dict[str, Any]]]:
    """Group predictions by top-level category."""
    grouped = {}
    for pred in predictions:
        label = pred['label']
        parts = label.split(separator)
        top_level = parts[0] if parts else label
        
        if top_level not in grouped:
            grouped[top_level] = []
        grouped[top_level].append(pred)
    
    for key in grouped:
        grouped[key].sort(key=lambda x: x['score'], reverse=True)
    
    return grouped


def get_best_per_category(
    predictions: List[Dict[str, Any]],
    separator: str = "."
) -> Dict[str, Dict[str, Any]]:
    """Get best prediction per top-level category."""
    grouped = group_predictions_by_hierarchy(predictions, separator)
    return {category: preds[0] for category, preds in grouped.items() if preds}