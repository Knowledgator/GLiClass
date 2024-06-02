import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from typing import Optional
import numpy as np
import argparse
import json

from dataclasses import dataclass, field
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
import transformers
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


from gliclass import GLiClassModelConfig, GLiClassModel

class DataCollatorWithPadding:
    def __init__(self, device = 'cuda:0'):
        self.device=device

    def __call__(self, batch):
        # Assuming batch is a list of dictionaries
        # Extract all keys
        keys = batch[0].keys()
        
        # Create a dictionary to hold padded data
        padded_batch = {key: [] for key in keys}
        
        # Collect data for each key
        for key in keys:
            # Collect the data for the current key
            key_data = [item[key] for item in batch]
            
            # Pad the data for the current key
            if isinstance(key_data[0], torch.Tensor):
                padded_batch[key] = pad_sequence(key_data, batch_first=True)#.to(self.device)
            elif isinstance(key_data[0], list):  # Assuming list of lists for non-tensor data
                max_length = max(len(seq) for seq in key_data)
                padded_batch[key] = torch.tensor([seq + [0] * (max_length - len(seq)) 
                                                    for seq in key_data])#.to(self.device)
            elif type(key_data[0]) in {int, float}:
                padded_batch[key] = torch.tensor(key_data)#.to(self.device)
            else:
                raise TypeError(f"Unsupported data type: {type(key_data[0])}")
        
        return padded_batch
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: Optional[float] = None
    others_weight_decay: Optional[float] = 0.0

class Trainer(transformers.Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.others_lr is not None:
                encoder_parameters = [name for name, _ in opt_model.named_parameters() if "encoder" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.others_weight_decay,
                        "lr": self.args.others_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.others_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
def tokenize_and_prepare_labels(example, tokenizer, max_length, problem_type='multi_label_classification'):
    input_text = []
    random.shuffle(example['all_labels'])
    random.shuffle(example['true_labels'])
    for label in example['all_labels']:
        label_tag = f"<<LABEL>>{label}<<SEP>>"
        input_text.append(label_tag)
    input_text = ''.join(input_text)+example['text']

    label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

    tokenized_inputs = tokenizer(input_text, truncation=True, max_length=max_length, padding="max_length")

    if problem_type == 'single_label_classification':
        labels = label2idx[example['true_labels'][0]]
    elif problem_type == 'multi_label_classification':
        labels = [1. if label in example['true_labels'] else 0. for label in example['all_labels']]
    else:
        raise NotImplementedError(f"{problem_type} is not implemented.")
    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs


class GLiClassDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, problem_type='multi_label_classification'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._data = examples
        self.problem_type = problem_type
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        example = self._data[idx]
        model_inputs = tokenize_and_prepare_labels(example, self.tokenizer, self.max_length, self.problem_type)
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
    parser.add_argument('--encoder_model_name', type=str, default = 'microsoft/deberta-v3-small')
    parser.add_argument('--save_path', type=str, default = 'models/gliclass/deberta_small')
    parser.add_argument('--data_path', type=str, default = '../data/zero-class.json')
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='first')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=3e-5)
    parser.add_argument('--others_lr', type=float, default=5e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.0)
    parser.add_argument('--others_weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            class_token_index=len(tokenizer),
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm
        )
        glicalss_config.problem_type = args.problem_type
        model = GLiClassModel(glicalss_config, from_pretrained=True)

        new_words = ["<<LABEL>>", "<<SEP>>"]
        tokenizer.add_tokens(new_words)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device, dtype=torch.bfloat16)
        
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    #shuffle
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    train_dataset = GLiClassDataset(train_data, tokenizer, args.max_length, args.problem_type)
    test_dataset = GLiClassDataset(test_data, tokenizer, args.max_length, args.problem_type)

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
        use_cpu = False,
        report_to=None,
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