import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig

import random
import torch
from torch.utils.data import Dataset


from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding

def prepare_labels(example, label2idx, problem_type):
    if problem_type == 'single_label_classification':
        labels = label2idx[example['true_labels'][0]]
    elif problem_type == 'multi_label_classification':
        labels = [1. if label in example['true_labels'] else 0. for label in example['all_labels']]
    else:
        raise NotImplementedError(f"{problem_type} is not implemented.")
    return torch.tensor(labels)

def tokenize_and_prepare_labels_for_uniencoder(example, tokenizer, max_length, problem_type='multi_label_classification'):
    input_text = []
    random.shuffle(example['all_labels'])
    random.shuffle(example['true_labels'])
    for label in example['all_labels']:
        label_tag = f"<<LABEL>>{label}"
        input_text.append(label_tag)
    input_text.append('<<SEP>>')
    input_text = ''.join(input_text)+example['text']

    label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

    tokenized_inputs = tokenizer(input_text, truncation=True, max_length=max_length, padding="max_length")
    tokenized_inputs['labels'] = prepare_labels(example, label2idx, problem_type)
    return tokenized_inputs

def tokenize_and_prepare_labels_for_encoder_decoder(example, tokenizer, max_length, problem_type='multi_label_classification'):
    class_texts = []
    random.shuffle(example['all_labels'])
    random.shuffle(example['true_labels'])
    for label in example['all_labels']:
        label_tag = f"<<LABEL>>{label}"
        class_texts.append(label_tag)
    class_texts.append('<<SEP>>')
    class_texts = ''.join(class_texts)

    label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

    tokenized_inputs = tokenizer(example['text'], truncation=True, max_length=max_length, padding="max_length")
    tokenized_classes = tokenizer(class_texts, truncation=True, max_length=max_length, padding="longest")
    tokenized_inputs["class_input_ids"] = tokenized_classes["input_ids"]
    tokenized_inputs["class_attention_mask"] = tokenized_classes["attention_mask"]   
    tokenized_inputs['labels'] = prepare_labels(example, label2idx, problem_type)
    return tokenized_inputs

def tokenize_and_prepare_labels_for_biencoder(example, tokenizer, max_length, problem_type='multi_label_classification'):
    input_text = example['text']
    class_texts = example['all_labels']

    tokenized_inputs = tokenizer(input_text, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
    tokenized_classes = tokenizer(class_texts, truncation=True, padding="longest", return_tensors='pt')

    tokenized_inputs["class_input_ids"] = tokenized_classes["input_ids"]
    tokenized_inputs["class_attention_mask"] = tokenized_classes["attention_mask"]

    label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

    tokenized_inputs['labels_mask'] = torch.ones(len(class_texts))
    tokenized_inputs['labels'] = prepare_labels(example, label2idx, problem_type)
    return tokenized_inputs

class GLiClassDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, 
                            problem_type='multi_label_classification', 
                            architecture_type = 'uni-encoder'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._data = examples
        self.problem_type = problem_type
        self.architecture_type = architecture_type
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        example = self._data[idx]

        if self.architecture_type == 'uni-encoder':
            model_inputs = tokenize_and_prepare_labels_for_uniencoder(example, self.tokenizer, self.max_length, self.problem_type)
        elif self.architecture_type == 'encoder-decoder':
            model_inputs = tokenize_and_prepare_labels_for_encoder_decoder(example, self.tokenizer, self.max_length, self.problem_type)
        elif self.architecture_type == 'bi-encoder':
            model_inputs = tokenize_and_prepare_labels_for_biencoder(example, self.tokenizer, self.max_length, self.problem_type)
        else:
            raise NotImplementedError('This architecture type is not implemented.')
        return model_inputs
    
def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    if args.problem_type == 'single_label_classification':
        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    elif args.problem_type == 'multi_label_classification':
        predictions = predictions.reshape(-1)
        preds = (predictions > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    else:
        raise NotImplementedError(f"{args.problem_type} is not implemented.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default= None)
    parser.add_argument('--encoder_model_name', type=str, default = 'google/flan-t5-small')
    parser.add_argument('--save_path', type=str, default = 'models/gliclass/t5_small')
    parser.add_argument('--data_path', type=str, default = '../data/zero-cats.json')
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='first')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--architecture_type', type=str, default='encoder-decoder')
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--others_lr', type=float, default=3e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--focal_loss_alpha', type=float, default=-1)
    parser.add_argument('--focal_loss_gamma', type=float, default=-1)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name, focal_loss_alpha=args.focal_loss_alpha,
                                                                focal_loss_gamma=args.focal_loss_gamma)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            class_token_index=len(tokenizer),
            text_token_index=len(tokenizer)+1,
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            contrastive_loss_coef=args.contrastive_loss_coef,
            normalize_features=args.normalize_features,
            extract_text_features=args.extract_text_features,
            architecture_type=args.architecture_type
        )
        glicalss_config.problem_type = args.problem_type
        model = GLiClassModel(glicalss_config, from_pretrained=True)

        if args.architecture_type in  {'uni-encoder', 'encoder-decoder'}:
            new_words = ["<<LABEL>>", "<<SEP>>"]
            tokenizer.add_tokens(new_words)
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)#, dtype=torch.bfloat16)
        
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    #shuffle
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    train_dataset = GLiClassDataset(train_data, tokenizer, args.max_length, args.problem_type, args.architecture_type)
    test_dataset = GLiClassDataset(test_data, tokenizer, args.max_length, args.problem_type, args.architecture_type)

    data_collator = DataCollatorWithPadding(device=device)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_steps = args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers = args.num_workers,
        logging_steps=100,
        use_cpu = False,
        report_to="none",
        fp16=args.fp16,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()