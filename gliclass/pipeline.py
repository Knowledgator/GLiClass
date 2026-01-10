import torch
from tqdm import tqdm
from typing import List, Dict, Union
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from .model import GLiClassModel, GLiClassBiEncoder
from .utils import retrieval_augmented_text

class BaseZeroShotClassificationPipeline(ABC):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0', progress_bar=True):
        self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_classes = max_classes
        self.classification_type = classification_type
        self.max_length = max_length
        self.progress_bar = progress_bar

        if not isinstance(device, torch.device):
            if torch.cuda.is_available() and 'cuda' in device:
                self.device = torch.device(device)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        if self.model.device != self.device:
            self.model.to(self.device)

    @abstractmethod
    def prepare_inputs(self, texts, labels, same_labels = False):
        pass
    
    @torch.no_grad()
    def get_embeddings(self, texts, labels, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
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
            tokenized_inputs = self.prepare_inputs(batch_texts, labels, same_labels)
            model_output = self.model(**tokenized_inputs, output_text_embeddings=True,
                                    output_class_embeddings=True)
            logits = model_output.logits
            text_embeddings = model_output.text_embeddings
            class_embeddings = model_output.class_embeddings
            batch_size = logits.shape[0]
            
            for i in range(batch_size):
                result = {
                    'logits': logits[i].cpu().numpy(),
                    'text_embedding': text_embeddings[i].cpu().numpy(),
                    'class_embeddings': class_embeddings[i].cpu().numpy()
                }
                results.append(result)
        
        return results

    @torch.no_grad()
    def __call__(self, texts, labels, threshold = 0.5, batch_size=8, rac_examples=None):
        if isinstance(texts, str):
            if rac_examples:
                texts = retrieval_augmented_text(texts, rac_examples)
            texts = [texts]
        else:
            if rac_examples:
                texts = [retrieval_augmented_text(text, examples) for text, examples in zip(texts, rac_examples)]
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
            if not same_labels:
                batch_labels = labels[idx:idx+batch_size]
            else:
                batch_labels = labels
            tokenized_inputs = self.prepare_inputs(batch_texts, batch_labels, same_labels)
            model_output = self.model(**tokenized_inputs)
            logits = model_output.logits
            if self.classification_type == 'single-label':
                for i in range(len(batch_texts)):
                    score = torch.softmax(logits[i], dim=-1)
                    if same_labels:
                        curr_labels = batch_labels
                    else:
                        curr_labels = batch_labels[i]
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
                    for j, prob in enumerate(probs[i][:len(curr_labels)]):
                        score = prob.item()
                        if score>=threshold and len(curr_labels):
                            text_results.append({'label': curr_labels[j], 'score': score})
                    results.append(text_results)
            else:
                raise ValueError("Unsupported classification type: choose 'single-label' or 'multi-label'")
        return results
    
class UniEncoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0', progress_bar=True):
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device, progress_bar)

    def prepare_input(self, text, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label.lower()}"
            input_text.append(label_tag)
        input_text.append('<<SEP>>')
        if self.model.config.prompt_first:
            input_text = ''.join(input_text)+text
        else:
            input_text = text+''.join(input_text)
        return input_text

    def prepare_inputs(self, texts, labels, same_labels = False):
        inputs = []
        
        if same_labels:
            for text in texts:
                inputs.append(self.prepare_input(text, labels))
        else:
            for text, labels_ in zip(texts, labels):
                inputs.append(self.prepare_input(text, labels_))
        
        tokenized_inputs = self.tokenizer(inputs, truncation=True, 
                                            max_length=self.max_length, 
                                                    padding="longest", return_tensors="pt").to(self.device)

        return tokenized_inputs

class EncoderDecoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0', progress_bar=True):
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device, progress_bar)

    def prepare_labels_prompt(self, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label.lower()}"
            input_text.append(label_tag)
        input_text.append('<<SEP>>')
        input_text = ''.join(input_text)
        return input_text

    def prepare_inputs(self, texts, labels, same_labels = False):
        prompts = []
        
        if same_labels:
            for _ in texts:
                prompts.append(self.prepare_labels_prompt(labels))
        else:
            for labels_ in labels:
                prompts.append(self.prepare_labels_prompt(labels_))
        
        tokenized_inputs = self.tokenizer(texts, truncation=True, 
                                            max_length=self.max_length, 
                                                    padding="longest", return_tensors="pt").to(self.device)
        
        tokenized_classes = self.tokenizer(prompts, max_length=self.max_length, 
                                        truncation=True, padding="longest", return_tensors='pt').to(self.device)
        tokenized_inputs["class_input_ids"] = tokenized_classes["input_ids"]
        tokenized_inputs["class_attention_mask"] = tokenized_classes["attention_mask"]

        return tokenized_inputs
    
class BiEncoderZeroShotClassificationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0', progress_bar=True):
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device, progress_bar)
        self.labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)

    def prepare_input(self, text, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>"
            input_text.append(label_tag)
        input_text.append('<<SEP>>')
        if self.model.config.prompt_first:
            input_text = ''.join(input_text)+text
        else:
            input_text = text+''.join(input_text)
        return input_text
    
    def prepare_inputs(self, texts, labels, same_labels=False):
        if self.model.config.architecture_type == 'bi-encoder-fused':
            inputs = []
            if same_labels:
                for text in texts:
                    inputs.append(self.prepare_input(text, labels))
            else:
                for text, labels_ in zip(texts, labels):
                    inputs.append(self.prepare_input(text, labels_))
        else:
            inputs = texts
        if same_labels:
            # If all texts use the same labels
            tokenized_inputs = self.tokenizer(inputs, truncation=True,
                                            max_length=self.max_length,
                                            padding="longest", return_tensors="pt").to(self.device)

            tokenized_labels = self.labels_tokenizer(labels, truncation=True,
                                            max_length=self.max_length,
                                            padding="longest", return_tensors="pt").to(self.device)
            tokenized_inputs['class_input_ids'] = tokenized_labels['input_ids'].expand(len(texts), -1, -1)
            tokenized_inputs['class_attention_mask'] = tokenized_labels['attention_mask'].expand(len(texts), -1, -1)
            
            labels_mask = [[1 for i in range(len(labels))] for j in range(len(texts))]
            tokenized_inputs["labels_mask"] = torch.tensor(labels_mask).to(self.device)
        else:
            # If each text has its own set of labels
            tokenized_inputs = self.tokenizer(inputs, truncation=True,
                                            max_length=self.max_length,
                                            padding="longest", return_tensors="pt").to(self.device)
            
            class_input_ids = []
            class_attention_mask = []
            
            for labels_set in labels:
                tokenized_labels = self.labels_tokenizer(labels_set, truncation=True,
                                                max_length=self.max_length,
                                                padding="max_length",
                                                return_tensors="pt").to(self.device)
                class_input_ids.append(tokenized_labels["input_ids"])
                class_attention_mask.append(tokenized_labels["attention_mask"])
            
            tokenized_inputs["class_input_ids"] = torch.stack(class_input_ids)
            tokenized_inputs["class_attention_mask"] = torch.stack(class_attention_mask)

            labels_mask = [[1 for i in range(len(labels[j]))] for j in range(len(texts))]
            tokenized_inputs["labels_mask"] = torch.tensor(labels_mask).to(self.device)
        return tokenized_inputs

class ZeroShotClassificationPipeline:
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0', progress_bar=True):
        if isinstance(model, str):
            model = GLiClassBiEncoder.from_pretrained(model)
        if model.config.architecture_type == 'uni-encoder':
            self.pipe = UniEncoderZeroShotClassificationPipeline(model, tokenizer, max_classes, 
                                                                    max_length, classification_type, device, progress_bar)
        elif model.config.architecture_type in {'encoder-decoder'}:
            self.pipe = EncoderDecoderZeroShotClassificationPipeline(model, tokenizer, max_classes, 
                                                                    max_length, classification_type, device, progress_bar)
        elif model.config.architecture_type in {'bi-encoder', 'bi-encoder-fused'}:
            self.pipe = BiEncoderZeroShotClassificationPipeline(model, tokenizer, max_classes, 
                                                                    max_length, classification_type, device, progress_bar)
        else:
            raise NotImplementedError("This artchitecture is not implemented")
    
    def get_embeddings(self, *args, **kwargs):
        results = self.pipe.get_embeddings(*args, **kwargs)
        return results
    
    def __call__(self, texts, labels, threshold = 0.5, batch_size=8, rac_examples=None):
        results = self.pipe(texts, labels, threshold = threshold, batch_size=batch_size, rac_examples=rac_examples)
        return results
    

class ZeroShotClassificationWithChunkingPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                 classification_type='multi-label', device='cuda:0', progress_bar=True,
                 text_chunk_size=8192, text_chunk_overlap=256, labels_chunk_size=8):
        if isinstance(model, str):
            model = GLiClassModel.from_pretrained(model)
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device, progress_bar)
        
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
                
            # Move start position, accounting for overlap
            start = end - overlap
            
        return chunks

    def prepare_input(self, text, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label.lower()}"
            input_text.append(label_tag)
        input_text.append('<<SEP>>')
        if self.model.config.prompt_first:
            input_text = ''.join(input_text) + text
        else:
            input_text = text + ''.join(input_text)
        return input_text

    def prepare_inputs(self, texts, labels, same_labels=False):
        inputs = []
        
        if same_labels:
            for text in texts:
                inputs.append(self.prepare_input(text, labels))
        else:
            for text, labels_ in zip(texts, labels):
                inputs.append(self.prepare_input(text, labels_))
        
        tokenized_inputs = self.tokenizer(inputs, truncation=True, 
                                          max_length=self.max_length, 
                                          padding="longest", return_tensors="pt").to(self.device)
        return tokenized_inputs

    def aggregate_chunk_scores(self, chunk_scores: List[Dict[str, float]], labels: List[str]) -> Dict[str, float]:
        """
        Aggregate scores across text chunks for each label.
        Uses max pooling - takes the highest score for each label across all chunks.
        """
        aggregated = {label: 0.0 for label in labels}
        
        for scores in chunk_scores:
            for label, score in scores.items():
                aggregated[label] = max(aggregated[label], score)
                
        return aggregated

    @torch.no_grad()
    def process_single_text(self, text, labels, threshold=0.5):
        """Process a single text through all text chunks and label chunks."""
        text_chunks = self.chunk_text(text)
        
        # Store scores for each label across all text chunks
        # List of dicts: [{label: score, ...}, ...]
        all_chunk_scores = []
        
        for text_chunk in text_chunks:
            # Collect logits across all label chunks for this text chunk
            chunk_logits = []
            all_labels = []
            
            for labels_idx in range(0, len(labels), self.labels_chunk_size):
                curr_labels = labels[labels_idx:labels_idx + self.labels_chunk_size]
                if not all_labels or labels_idx == 0:
                    all_labels = []
                if labels_idx == 0:
                    all_labels = []
                all_labels.extend(curr_labels)
                
                tokenized_inputs = self.prepare_inputs([text_chunk], curr_labels, same_labels=True)
                model_output = self.model(**tokenized_inputs)
                logits = model_output.logits
                
                chunk_logits.extend(logits[0][:len(curr_labels)].tolist())
            
            # Convert logits to scores
            text_logits = torch.tensor(chunk_logits)
            
            if self.classification_type == 'single-label':
                scores = torch.softmax(text_logits, dim=-1)
            else:  # multi-label
                scores = torch.sigmoid(text_logits)
            
            # Store scores as dict
            chunk_score_dict = {label: scores[i].item() for i, label in enumerate(all_labels)}
            all_chunk_scores.append(chunk_score_dict)
        
        # Aggregate scores across text chunks
        aggregated_scores = self.aggregate_chunk_scores(all_chunk_scores, labels)
        
        # Format results
        if self.classification_type == 'single-label':
            # Re-normalize after max pooling for single-label
            total = sum(aggregated_scores.values())
            if total > 0:
                aggregated_scores = {k: v / total for k, v in aggregated_scores.items()}
            
            best_label = max(aggregated_scores, key=aggregated_scores.get)
            return [{'label': best_label, 'score': aggregated_scores[best_label]}]
        
        else:  # multi-label
            text_results = []
            for label, score in aggregated_scores.items():
                if score >= threshold:
                    text_results.append({'label': label, 'score': score})
            text_results.sort(key=lambda x: x['score'], reverse=True)
            return text_results

    @torch.no_grad()
    def __call__(self, texts, labels, threshold=0.5, batch_size=8,
                        labels_chunk_size=None, text_chunk_size=None, text_chunk_overlap=None,
                        rac_examples=None):
        """
        Batched version - more efficient when texts are shorter than chunk size.
        Falls back to sequential processing for long texts.
        """
        # Update chunk sizes if provided
        if labels_chunk_size is not None:
            self.labels_chunk_size = labels_chunk_size
        if text_chunk_size is not None:
            self.text_chunk_size = text_chunk_size
        if text_chunk_overlap is not None:
            self.text_chunk_overlap = text_chunk_overlap

        # Convert single text to list
        if isinstance(texts, str):
            if rac_examples:
                texts = retrieval_augmented_text(texts, rac_examples)
            texts = [texts]
        else:
            if rac_examples:
                texts = [retrieval_augmented_text(text, examples) for text, examples in zip(texts, rac_examples)]

        # Separate short and long texts
        short_texts = []
        short_indices = []
        long_texts = []
        long_indices = []
        
        for i, text in enumerate(texts):
            if len(text) <= self.text_chunk_size:
                short_texts.append(text)
                short_indices.append(i)
            else:
                long_texts.append(text)
                long_indices.append(i)

        results = [None] * len(texts)

        # Process short texts in batches (no text chunking needed)
        if short_texts:
            iterable = range(0, len(short_texts), batch_size)
            if self.progress_bar:
                iterable = tqdm(iterable, desc="Processing short texts")

            for idx in iterable:
                batch_texts = short_texts[idx:idx + batch_size]
                batch_indices = short_indices[idx:idx + batch_size]

                # Collect all logits across label chunks
                all_logits = [[] for _ in range(len(batch_texts))]
                all_labels = []

                for labels_idx in range(0, len(labels), self.labels_chunk_size):
                    curr_labels = labels[labels_idx:labels_idx + self.labels_chunk_size]
                    if labels_idx == 0:
                        all_labels = []
                    all_labels.extend(curr_labels)

                    tokenized_inputs = self.prepare_inputs(batch_texts, curr_labels, same_labels=True)
                    model_output = self.model(**tokenized_inputs)
                    logits = model_output.logits

                    for i in range(len(batch_texts)):
                        all_logits[i].extend(logits[i][:len(curr_labels)].tolist())

                # Process collected logits
                for i, orig_idx in enumerate(batch_indices):
                    text_logits = torch.tensor(all_logits[i])

                    if self.classification_type == 'single-label':
                        score = torch.softmax(text_logits, dim=-1)
                        pred_idx = torch.argmax(score).item()
                        pred_label = all_labels[pred_idx]
                        results[orig_idx] = [{'label': pred_label, 'score': score[pred_idx].item()}]

                    elif self.classification_type == 'multi-label':
                        sigmoid = torch.nn.Sigmoid()
                        probs = sigmoid(text_logits)
                        text_results = []
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
                text_results = self.process_single_text(text, labels, threshold)
                results[orig_idx] = text_results

        return results