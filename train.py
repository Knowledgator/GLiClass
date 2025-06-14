import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig

import random
import torch

from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset

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
        labels = np.where(labels>0.5, 1, 0)
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

def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name, focal_loss_alpha=args.focal_loss_alpha,
                                                                focal_loss_gamma=args.focal_loss_gamma,
                                                                focal_loss_reduction=args.focal_loss_reduction)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        if args.label_model_name is not None:
            label_model_config = AutoConfig.from_pretrained(args.label_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            label_model_name=args.label_model_name,
            label_model_config=label_model_config,
            class_token_index=len(tokenizer),
            text_token_index=len(tokenizer)+1,
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            focal_loss_reduction=args.focal_loss_reduction,
            contrastive_loss_coef=args.contrastive_loss_coef,
            normalize_features=args.normalize_features,
            extract_text_features=args.extract_text_features,
            architecture_type=args.architecture_type,
            prompt_first=args.prompt_first,
            squeeze_layers=args.squeeze_layers,
            layer_wise=args.layer_wise,
            encoder_layer_id=args.encoder_layer_id,
            shuffle_labels=args.shuffle_labels
        )

        model = GLiClassModel(glicalss_config, from_pretrained=True)

        if args.architecture_type in  {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder'}:
            new_words = ["<<LABEL>>", "<<SEP>>"]
            tokenizer.add_tokens(new_words, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    if model.config.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)
    else:
        labels_tokenizer = None

    model.config.problem_type = args.problem_type

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    train_dataset = GLiClassDataset(train_data, tokenizer, args.max_length, 
                                    args.problem_type, args.architecture_type, 
                                    args.prompt_first, labels_tokenizer=labels_tokenizer)
    test_dataset = GLiClassDataset(test_data, tokenizer, args.max_length, args.problem_type, 
                                        args.architecture_type, args.prompt_first,
                                        labels_tokenizer = labels_tokenizer)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default= None)
    parser.add_argument('--encoder_model_name', type=str, default = 'microsoft/deberta-v3-small')
    parser.add_argument('--label_model_name', type=str, default = "BAAI/bge-small-en-v1.5")
    parser.add_argument('--save_path', type=str, default = 'models/')
    parser.add_argument('--data_path', type=str, default = 'data/zero-cats.json')
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='avg')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--architecture_type', type=str, default='uni-encoder')
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--prompt_first', type=bool, default=True)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--squeeze_layers', type=bool, default=False)
    parser.add_argument('--layer_wise', type=bool, default=False)
    parser.add_argument('--encoder_layer_id', type=int, default=-1)
    parser.add_argument('--shuffle_labels', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--others_lr', type=float, default=3e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--focal_loss_alpha', type=float, default=-1)
    parser.add_argument('--focal_loss_gamma', type=float, default=-1)
    parser.add_argument('--focal_loss_reduction', type=str, default='none', choices=['none', 'mean', 'sum'])
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    args = parser.parse_args()

    main(args)
