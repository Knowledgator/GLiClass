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

        if torch.cuda.is_available() and 'cuda' in device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

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
            tokenized_inputs = self.prepare_inputs(batch_texts, labels, same_labels)
            model_output = self.model(**tokenized_inputs)
            logits = model_output.logits
            if self.classification_type == 'single-label':
                for i in range(len(batch_texts)):
                    score = torch.softmax(logits[i], dim=-1)
                    if same_labels:
                        curr_labels = labels
                    else:
                        curr_labels = labels[i]
                    pred_label = curr_labels[torch.argmax(score).item()]
                    results.append([{'label': pred_label, 'score': score.max().item()}])
            elif self.classification_type == 'multi-label':
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits)
                for i in range(len(batch_texts)):
                    text_results = []
                    if same_labels:
                        curr_labels = labels
                    else:
                        curr_labels = labels[i]
                    for j, prob in enumerate(probs[i]):
                        score = prob.item()
                        if score>threshold:
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
    
class ZeroShotClassificationWithLabelsChunkingPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0'):
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device)
        if isinstance(model, str):
            self.model = GLiClassModel.from_pretrained(model)
        else:
            self.model = model

        if self.model.device != self.device:
            self.model.to(self.device)

    def prepare_input(self, text, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label.lower()}"
            input_text.append(label_tag)
        input_text.append('<<SEP>>')
        input_text = ''.join(input_text)+text
        return input_text

    def prepare_inputs(self, texts, labels):
        inputs = []
    
        for text in texts:
            inputs.append(self.prepare_input(text, labels))
        
        tokenized_inputs = self.tokenizer(inputs, truncation=True, 
                                            max_length=self.max_length, 
                                                    padding="longest", return_tensors="pt").to(self.device)
        return tokenized_inputs
    
    @torch.no_grad()
    def __call__(self, texts, labels, threshold = 0.5, batch_size=8, labels_chunk_size=4): #labels - List[str]
        results = []

        iterable = range(0, len(texts), batch_size)
        if self.progress_bar:
            iterable = tqdm(iterable)

        for idx in iterable:
            batch_texts = texts[idx:idx+batch_size]

            batch_results = []
            for labels_batch in range(0, len(labels), labels_chunk_size):
                curr_labels = labels[labels_batch:labels_batch+labels_chunk_size]
                tokenized_inputs = self.prepare_inputs(batch_texts, curr_labels)
                model_output = self.model(**tokenized_inputs)
                logits = model_output.logits
                curr_results = []
                if self.classification_type == 'single-label':
                    for i in range(len(batch_texts)):
                        score = logits[i]
                        pred_label = curr_labels[torch.argmax(score).item()]
                        curr_results.append([{'label': pred_label, 'score': score.max().item()}])
                elif self.classification_type == 'multi-label':
                    sigmoid = torch.nn.Sigmoid()
                    probs = sigmoid(logits)
                    for i in range(len(batch_texts)):
                        text_results = []
                        for j, prob in enumerate(probs[i]):
                            score = prob.item()
                            if score>threshold:
                                text_results.append({'label': curr_labels[j], 'score': score})
                        curr_results.append(text_results)
                else:
                    raise ValueError("Unsupported classification type: choose 'single-label' or 'multi-label'")
                batch_results.append(curr_results)

            # Merge results from different label chunks
            merged_batch_results = []
            for i in range(len(batch_texts)):
                text_results = []
                for chunk_result in batch_results:
                    text_results.extend(chunk_result[i])
                
                if self.classification_type == 'single-label':
                    # Keep only the highest scoring label
                    merged_batch_results.append([max(text_results, key=lambda x: x['score'])])
                else:
                    # Sort multi-label results by score in descending order
                    merged_batch_results.append(sorted(text_results, key=lambda x: x['score'], reverse=True))
            
            results.extend(merged_batch_results)
        
        return results
      