import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .model import GLiClassModel
from .config import GLiClassModelConfig

class ZeroShotClassificationPipeline():
    def __init__(self, model, tokenizer, max_classes=25, max_length=1024, 
                                classification_type='multi-label', device='cuda:0'):
        self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        if isinstance(model, str):
            self.model = GLiClassModel.from_pretrained(model)
        else:
            self.model = model
        self.max_classes = max_classes
        self.classification_type = classification_type
        self.max_length = max_length

        if device == 'cuda:0' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if self.model.device != self.device:
            self.model.to(self.device)

    def prepare_input(self, text, labels):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label}<<SEP>>"
            input_text.append(label_tag)
        input_text = ''.join(input_text)+text
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
                                                    padding="max_length", return_tensors="pt").to(self.device)

        return tokenized_inputs
    
    @torch.no_grad()
    def __call__(self, texts, labels, threshold = 0.5, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels[0], str):
            same_labels = True
        else:
            same_labels = False

        results = []
        for idx in tqdm(range(0, len(texts), batch_size)):
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