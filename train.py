import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from torch.utils.data import WeightedRandomSampler
from packaging import version

import random
import torch

from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset, AugmentationConfig
from gliclass.multimodal import get_tokenizer

class CustomTrainer(Trainer):
    """Trainer with weighted random sampling support."""
    
    def __init__(self, *args, use_weighted_sampling=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weighted_sampling = use_weighted_sampling
    
    def _get_train_sampler(self, train_dataset) -> torch.utils.data.Sampler:
        if not self.use_weighted_sampling:
            return super()._get_train_sampler()
        
        weights = train_dataset.get_diversity()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
def compute_metrics(p, problem_type='multi_label_classification'):
    """Compute evaluation metrics.
    
    Args:
        p: Predictions tuple (predictions, labels)
        problem_type: Type of classification problem
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = p
    labels = labels.reshape(-1)
    
    if problem_type == 'single_label_classification':
        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    elif problem_type == 'multi_label_classification':
        predictions = predictions.reshape(-1)
        preds = (predictions > 0.5).astype(int)
        labels = np.where(labels > 0.5, 1, 0)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    else:
        raise NotImplementedError(f"{problem_type} is not implemented.")


def load_dataset(data_path: str) -> list:
    """Load dataset from JSON file.
    
    Args:
        data_path: Path to JSON data file
        
    Returns:
        List of data samples
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def recurrent_kwargs(args):
    """Recurrent reasoning settings for the decoder-kv scorer; unset flags keep config/checkpoint values."""
    kwargs = dict(
        recurrent_steps=args.recurrent_steps,
        recurrent_min_steps=args.recurrent_min_steps,
        recurrent_inference_max_steps=args.recurrent_inference_max_steps,
        recurrent_halt_threshold=args.recurrent_halt_threshold,
        recurrent_improvement_coef=args.recurrent_improvement_coef,
        recurrent_improvement_margin=args.recurrent_improvement_margin,
        recurrent_bptt_steps=args.recurrent_bptt_steps,
        recurrent_read_text=args.recurrent_read_text,
        recurrent_confidence_coef=args.recurrent_confidence_coef,
        recurrent_confidence_mode=args.recurrent_confidence_mode,
        recurrent_confidence_margin=args.recurrent_confidence_margin,
    )
    return {key: value for key, value in kwargs.items() if value is not None}


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Load or create model
    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(
            args.model_name, 
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            focal_loss_reduction=args.focal_loss_reduction,
            **recurrent_kwargs(args),
        )
        # multi-modal decoder-kv: the processor tokenizes text and prepares images / audio
        tokenizer_class = AutoProcessor if model.config.multimodal else AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(args.model_name)
    else:
        tokenizer_class = AutoProcessor if args.multimodal else AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(args.encoder_model_name)
        text_tokenizer = get_tokenizer(tokenizer)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)

        label_model_config = None
        if args.label_model_name is not None:
            label_model_config = AutoConfig.from_pretrained(args.label_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            label_model_name=args.label_model_name,
            label_model_config=label_model_config,
            class_token_index=len(text_tokenizer),
            text_token_index=len(text_tokenizer)+1,
            example_token_index=len(text_tokenizer)+2,
            pooling_strategy=args.pooler_type,
            class_token_pooling=args.class_token_pooling,
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
            shuffle_labels=args.shuffle_labels,
            dropout=args.dropout,
            use_segment_embeddings=args.use_segment_embeddings,
            multimodal=args.multimodal,
            freeze_media_encoders=not args.train_media_encoders,
            **recurrent_kwargs(args),
        )
        if args.architecture_type == 'decoder-kv':
            # <<SEP>> is added right after <<LABEL>> below
            glicalss_config.sep_token_index = len(text_tokenizer) + 1

        model = GLiClassModel(glicalss_config, from_pretrained=True).to(dtype=torch.float32)

        if args.architecture_type in {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder', 'decoder-kv'}:
            new_words = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]
            text_tokenizer.add_tokens(new_words, special_tokens=True)
            model.resize_token_embeddings(len(text_tokenizer))

    model.to(device)

    # Get labels tokenizer if needed
    if model.config.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)
    else:
        labels_tokenizer = None

    model.config.problem_type = args.problem_type

    # Load current training data
    data = load_dataset(args.data_path)
    print(f'Dataset size: {len(data)}')
    
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    print('Dataset is splitted...')

    # Create augmentation config with all parameters
    augment_config = AugmentationConfig(
        enabled=args.enable_augmentation,
        random_label_removal_prob=args.random_label_removal_prob,
        random_label_addition_prob=args.random_label_addition_prob,
        random_text_addition_prob=args.random_text_addition_prob,
        random_add_description_prob=args.random_add_description_prob,
        random_add_synonyms_prob=args.random_add_synonyms_prob,
        random_add_examples_prob=args.random_add_examples_prob,
        max_num_examples=args.max_num_examples
    )
    
    if args.labels_desc_path is not None:
        labels_descriptions = load_dataset(args.labels_desc_path)
        label_to_description = {item.get("label"): item for item in labels_descriptions}
    else:
        label_to_description = {}

    train_dataset = GLiClassDataset(train_data, tokenizer, augment_config, 
                                    label_to_description, args.max_length, 
                                    args.problem_type, args.architecture_type, 
                                    args.prompt_first, labels_tokenizer=labels_tokenizer)
    
    # Disable augmentation for test dataset
    test_augment_config = AugmentationConfig(enabled=False)
    test_dataset = GLiClassDataset(test_data, tokenizer, test_augment_config, 
                                        label_to_description,
                                        args.max_length, args.problem_type, 
                                        args.architecture_type, args.prompt_first,
                                        labels_tokenizer = labels_tokenizer)

    # Load previous dataset for EWC if provided
    prev_dataset = None
    if args.use_ewc and args.prev_data_path is not None:
        print(f'Loading previous dataset for EWC from: {args.prev_data_path}')
        prev_data = load_dataset(args.prev_data_path)
        print(f'Previous dataset size: {len(prev_data)}')
        
        # Use a subset if specified
        if args.ewc_fisher_samples is not None and args.ewc_fisher_samples < len(prev_data):
            random.shuffle(prev_data)
            prev_data = prev_data[:args.ewc_fisher_samples]
            print(f'Using {len(prev_data)} samples for Fisher estimation')
        
        prev_dataset = GLiClassDataset(prev_data, tokenizer, test_augment_config, 
                                        label_to_description,
                                        args.max_length, args.problem_type, 
                                        args.architecture_type, args.prompt_first,
                                        labels_tokenizer = labels_tokenizer)

    data_collator = DataCollatorWithPadding(device=device)

    # Create training arguments with EWC parameters
    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        recurrent_lr=args.recurrent_lr,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        # transformers v5 dropped warmup_ratio; a float warmup_steps is a ratio there
        **({"warmup_steps": args.warmup_ratio} if version.parse(transformers.__version__) >= version.parse("5.0.0")
           else {"warmup_ratio": args.warmup_ratio}),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        use_cpu=False,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        # EWC parameters
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        ewc_fisher_samples=args.ewc_fisher_samples,
        ewc_normalize_fisher=args.ewc_normalize_fisher,
        ewc_gamma=args.ewc_gamma,
    )

    # Create compute_metrics function with problem_type closure
    def compute_metrics_fn(p):
        return compute_metrics(p, args.problem_type)

    # Create trainer with EWC support
    # Handle version differences between transformers v4 and v5
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_fn,
        "prev_dataset": prev_dataset,  # Pass previous dataset for EWC
    }

    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = CustomTrainer(**trainer_kwargs)
    
    # Print EWC status
    if args.use_ewc:
        if args.prev_data_path is not None:
            print(f'\nEWC enabled with lambda={args.ewc_lambda}')
        else:
            print('\nWarning: EWC is enabled but no previous data path provided. EWC will not be used.')
    
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(args.save_path, 'final_model')
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f'Final model saved to {final_output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GLiClass model with optional EWC for continual learning')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default=None,
                        help='Pretrained model name or path')
    parser.add_argument('--encoder_model_name', type=str, default='microsoft/deberta-v3-small',
                        help='Encoder model name')
    parser.add_argument('--label_model_name', type=str, default="BAAI/bge-small-en-v1.5",
                        help='Label model name')
    
    # Path arguments
    parser.add_argument('--save_path', type=str, default='models/',
                        help='Path to save trained model')
    parser.add_argument('--data_path', type=str, default='data/zero-cats.json',
                        help='Path to training data JSON file')
    parser.add_argument('--prev_data_path', type=str, default=None,
                        help='Path to previous task data for EWC (required if use_ewc=True)')
    parser.add_argument('--labels_desc_path', type=str, default = None)

    # Model architecture arguments
    parser.add_argument('--problem_type', type=str, default='multi_label_classification',
                        choices=['single_label_classification', 'multi_label_classification'])
    parser.add_argument('--pooler_type', type=str, default='avg')
    parser.add_argument('--scorer_type', type=str, default='simple')
    parser.add_argument('--architecture_type', type=str, default='uni-encoder')
    parser.add_argument('--class_token_pooling', type=str, default='first')
    parser.add_argument('--normalize_features', type=bool, default=False)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--prompt_first', type=bool, default=True)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--squeeze_layers', type=bool, default=False)
    parser.add_argument('--layer_wise', type=bool, default=False)
    parser.add_argument('--encoder_layer_id', type=int, default=-1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--shuffle_labels', type=bool, default=True)
    parser.add_argument('--use_segment_embeddings', type=bool, default=False)
    parser.add_argument('--multimodal', action='store_true',
                        help='decoder-kv: keep the vision / audio encoders of Qwen3.5 / Gemma 4; '
                             'data items may then carry "images" and "audio" lists')
    parser.add_argument('--train_media_encoders', action='store_true',
                        help='Also train the vision / audio encoders (frozen by default)')

    # Recurrent hidden reasoning (decoder-kv scorer)
    parser.add_argument('--recurrent_steps', type=int, default=None,
                        help='Max scorer reasoning steps during training (default 1 = disabled)')
    parser.add_argument('--recurrent_min_steps', type=int, default=None,
                        help='Min training steps; depth is sampled from [min, max] per batch (default 1)')
    parser.add_argument('--recurrent_inference_max_steps', type=int, default=None,
                        help='Max reasoning steps at inference (default: recurrent_steps)')
    parser.add_argument('--recurrent_halt_threshold', type=float, default=None,
                        help='Stop inference recurrence once label probabilities change less than this')
    parser.add_argument('--recurrent_improvement_coef', type=float, default=None)
    parser.add_argument('--recurrent_improvement_margin', type=float, default=None)
    parser.add_argument('--recurrent_bptt_steps', type=int, default=None)
    parser.add_argument('--recurrent_read_text', action='store_true', default=None,
                        help='Let every recurrent step cross-attend to the text (default: labels only)')
    parser.add_argument('--recurrent_confidence_coef', type=float, default=None,
                        help='Weight of the confidence (low-entropy) loss on recurrent steps (default 0 = off)')
    parser.add_argument('--recurrent_confidence_mode', type=str, default=None, choices=['relative', 'absolute'])
    parser.add_argument('--recurrent_confidence_margin', type=float, default=None,
                        help='relative mode: each step must cut entropy by this fraction vs the previous step')
    parser.add_argument('--recurrent_lr', type=float, default=None,
                        help='Learning rate for the recurrent reasoning cell (default: others_lr)')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--others_lr', type=float, default=3e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--optim', type=str, default='adamw_torch',
                        help='transformers optimizer name, e.g. adafactor to fit larger backbones in memory')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--logging_steps', type=int, default=100)
    
    # Augmentation parameters
    parser.add_argument('--enable_augmentation', type=bool, default=True)
    parser.add_argument('--random_label_removal_prob', type=float, default=0.05)
    parser.add_argument('--random_label_addition_prob', type=float, default=0.05)
    parser.add_argument('--random_text_addition_prob', type=float, default=0.05)
    parser.add_argument('--random_add_description_prob', type=float, default=0.05)
    parser.add_argument('--random_add_synonyms_prob', type=float, default=0.05)
    parser.add_argument('--random_add_examples_prob', type=float, default=0.1)
    parser.add_argument('--max_num_examples', type=int, default=5)


    # Loss arguments
    parser.add_argument('--focal_loss_alpha', type=float, default=-1)
    parser.add_argument('--focal_loss_gamma', type=float, default=-1)
    parser.add_argument('--focal_loss_reduction', type=str, default='none',
                        choices=['none', 'mean', 'sum'])
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.)
    
    # EWC arguments
    parser.add_argument('--use_ewc', action='store_true',
                        help='Enable Elastic Weight Consolidation for continual learning')
    parser.add_argument('--ewc_lambda', type=float, default=100.0,
                        help='Lambda parameter for EWC penalty (higher = more regularization)')
    parser.add_argument('--ewc_fisher_samples', type=int, default=None,
                        help='Number of samples to use for Fisher information estimation (None = use all)')
    parser.add_argument('--ewc_normalize_fisher', type=bool, default=True,
                        help='Whether to normalize Fisher information values')
    parser.add_argument('--ewc_gamma', type=float, default=0.95,
                        help='Decay factor for Online EWC (0 < gamma < 1)')
    
    args = parser.parse_args()

    # Validate EWC arguments
    if args.use_ewc and args.prev_data_path is None:
        print("Warning: --use_ewc is set but --prev_data_path is not provided.")
        print("EWC requires previous task data to compute Fisher information.")
        print("Training will proceed without EWC.")
    
    main(args)