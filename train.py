import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import argparse
import json
import math

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from torch.utils.data import WeightedRandomSampler
from packaging import version

import random
import torch
from torch import nn

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
            return super()._get_train_sampler(train_dataset)
        
        weights = train_dataset.get_diversity()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
def log_encoder_backend(model):
    """Print which encoder class was loaded, e.g. to confirm the FlashDeBERTa backend is in use."""
    encoder = getattr(model.model, 'encoder_model', None)
    if encoder is None:
        return
    if hasattr(encoder, 'get_base_model'):  # PEFT wrapper
        encoder = encoder.get_base_model()
    name = type(encoder).__name__
    print(f'Encoder backend: {name}')
    if name == 'FlashDebertaV2Model':
        print('FlashDeBERTa is used: attention runs on FlashDeBERTa kernels.')
    elif name == 'DebertaV2Model' and os.environ.get('USE_FLASHDEBERTA'):
        print('WARNING: USE_FLASHDEBERTA is set but the standard DebertaV2Model was loaded '
              '(is the flashdeberta package installed in this Python environment?).')

SPECIAL_TOKEN_INIT_WORDS = {"<<LABEL>>": "label", "<<SEP>>": "text", "<<EXAMPLE>>": "example"}

def init_special_token_embeddings(embeddings, text_tokenizer, new_ids, num_pretrained_rows):
    """Initialize the rows of newly added special tokens from the embedding of a descriptive word.

    Rows of new tokens are either mean-initialized by resize_token_embeddings or, when the backbone
    has spare vocabulary rows (e.g. deberta-v2-xxlarge), left as never-trained pretrained rows that
    look like [PAD]. Either way <<LABEL>>, <<SEP>> and <<EXAMPLE>> start nearly identical to each other
    and with a much smaller norm than real tokens. Seeding each from a distinct word ("label", "text",
    "example") and rescaling to the median norm of the pretrained vocabulary keeps them distinguishable.
    """
    weight = embeddings.weight
    with torch.no_grad():
        target_norm = weight[:num_pretrained_rows].float().norm(dim=-1).median()
        for token, token_id in new_ids.items():
            word = SPECIAL_TOKEN_INIT_WORDS.get(token, token.strip("<>").lower())
            word_ids = [i for i in text_tokenizer(word, add_special_tokens=False)['input_ids']
                        if i < num_pretrained_rows]
            if not word_ids:
                continue
            vector = weight[word_ids].float().mean(dim=0)
            weight[token_id] = (vector * target_norm / vector.norm()).to(weight.dtype)
            print(f'Initialized {token} (id {token_id}) from "{word}" embedding, norm {target_norm:.3f}')

def freeze_decoder(model, train_last_layers=None):
    """decoder-kv: freeze the decoder backbone, train only the scorer.

    Without train_last_layers the rows of the special tokens (<<LABEL>>, <<SEP>>, <<EXAMPLE>>) are new, so they
    stay trainable: their gradients pass through a row mask and the embedding matrix is kept out of weight decay,
    so every pretrained row stays exactly as loaded.

    With train_last_layers (a fraction) the last layers of the decoder and its final norm train too. The input of
    the first trainable layer is detached, so the backward pass stops there instead of running through all frozen
    layers; the special token rows then stay at their initial values.
    """
    decoder = getattr(model.model, 'decoder_model', None)
    if decoder is None:
        raise ValueError('--freeze_decoder / --train_last_layers require --architecture_type decoder-kv')
    decoder.requires_grad_(False)

    if train_last_layers is not None:
        if not 0 < train_last_layers <= 1:
            raise ValueError(f'--train_last_layers must be in (0, 1], got {train_last_layers}')
        text_model = getattr(decoder, 'language_model', decoder)
        layers = text_model.layers
        num_trainable = math.ceil(train_last_layers * len(layers))
        first_trainable = len(layers) - num_trainable
        for layer in layers[first_trainable:]:
            layer.requires_grad_(True)
        if getattr(text_model, 'norm', None) is not None:
            text_model.norm.requires_grad_(True)

        # gradient checkpointing makes the embedding output require grad (enable_input_require_grads),
        # which would otherwise backpropagate through all frozen layers for nothing
        def detach_hidden_states(_module, args, kwargs):
            if args:
                return (args[0].detach(), *args[1:]), kwargs
            return args, {**kwargs, 'hidden_states': kwargs['hidden_states'].detach()}

        layers[first_trainable].register_forward_pre_hook(detach_hidden_states, with_kwargs=True)
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        print(f'Decoder frozen except its last {num_trainable} of {len(layers)} layers '
              f'(layers {first_trainable}-{len(layers) - 1}) and final norm: '
              f'{trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters receive gradients')
        return

    config = model.config
    special_ids = sorted({index for index in (config.class_token_index, config.text_token_index,
                                              config.example_token_index, getattr(config, 'sep_token_index', None))
                          if index is not None and index >= 0})
    embeddings = decoder.get_input_embeddings().weight
    special_ids = [index for index in special_ids if index < embeddings.shape[0]]
    if special_ids:
        mask = torch.zeros(embeddings.shape[0], 1, device=embeddings.device, dtype=embeddings.dtype)
        mask[special_ids] = 1.0
        embeddings.requires_grad_(True)
        embeddings.register_hook(lambda grad: grad * mask)
        model.no_weight_decay_param_names = {name for name, param in model.named_parameters() if param is embeddings}

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad and param is not embeddings)
    trainable += len(special_ids) * embeddings.shape[1]
    total = sum(param.numel() for param in model.parameters())
    print(f'Decoder frozen (special token rows {special_ids} stay trainable): '
          f'{trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters receive gradients')

def cast_frozen_to_bf16(model):
    """Store frozen decoder linear / embedding weights in bf16.

    Under bf16 autocast every fp32 weight is cast to bf16 on each forward (twice with checkpoint recompute);
    frozen weights never need fp32. Norms, convolutions, small fp32 parameters and trainable layers stay fp32.
    """
    decoder = model.model.decoder_model
    saved = 0
    for module in decoder.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and not any(p.requires_grad for p in module.parameters()):
            saved += sum(p.numel() for p in module.parameters() if p.dtype == torch.float32) * 2
            module.to(torch.bfloat16)
    print(f'Frozen decoder weights stored in bf16: {saved / 1e9:.1f} GB saved')

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


def load_dataset(data_path) -> list:
    """Load dataset from one or several JSON / JSONL files.
    
    Args:
        data_path: Path to a JSON (list of samples) or JSONL (one sample per line) file,
            or a list of such paths whose samples are concatenated
        
    Returns:
        List of data samples
    """
    paths = [data_path] if isinstance(data_path, str) else data_path
    data = []
    for path in paths:
        with open(path, 'r') as f:
            if path.endswith('.jsonl'):
                part = [json.loads(line) for line in f if line.strip()]
            else:
                part = json.load(f)
        if len(paths) > 1:
            print(f'Loaded {len(part)} samples from {path}')
        data.extend(part)
    return data


def has_media(example) -> bool:
    return bool(example.get('images') or example.get('audio'))


def filter_text_only(data, name) -> list:
    """Drop samples with images / audio, which a text-only model cannot consume."""
    filtered = [example for example in data if not has_media(example)]
    if len(filtered) < len(data):
        print(f'Removed {len(data) - len(filtered)} multi-modal samples from {name} (model is not multi-modal)')
    return filtered


def recurrent_kwargs(args):
    """Decoder-kv scorer settings (recurrent reasoning, scorer input); unset flags keep config/checkpoint values."""
    kwargs = dict(
        scorer_full_sequence=args.scorer_full_sequence,
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
            use_embedding_projection=args.use_embedding_projection,
            **recurrent_kwargs(args),
        )
        if args.architecture_type == 'decoder-kv':
            # <<SEP>> is added right after <<LABEL>> below
            glicalss_config.sep_token_index = len(text_tokenizer) + 1

        model = GLiClassModel(glicalss_config, from_pretrained=True).to(dtype=torch.float32)

        if args.architecture_type in {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder', 'decoder-kv'}:
            new_words = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]
            num_pretrained_rows = len(text_tokenizer)
            added = [word for word in new_words if word not in text_tokenizer.get_vocab()]
            text_tokenizer.add_tokens(new_words, special_tokens=True)
            embeddings = model.resize_token_embeddings(len(text_tokenizer))
            init_special_token_embeddings(embeddings, text_tokenizer,
                                          {word: text_tokenizer.convert_tokens_to_ids(word) for word in added},
                                          num_pretrained_rows)

    model.to(device)
    log_encoder_backend(model)
    if args.freeze_decoder or args.train_last_layers is not None:
        freeze_decoder(model, args.train_last_layers)
        if args.train_last_layers is not None and args.bf16:
            cast_frozen_to_bf16(model)

    # Get labels tokenizer if needed
    if model.config.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(model.config.label_model_name)
    else:
        labels_tokenizer = None

    model.config.problem_type = args.problem_type

    # Load current training data
    data = load_dataset(args.data_path)
    multimodal = model.config.multimodal
    if not multimodal:
        data = filter_text_only(data, 'training data')
    print(f'Dataset size: {len(data)}')
    
    random.shuffle(data)    
    print('Dataset is shuffled...')

    if args.val_data_path is not None:
        train_data = data
        test_data = load_dataset(args.val_data_path)
        if not multimodal:
            test_data = filter_text_only(test_data, 'validation data')
        random.shuffle(test_data)
        print(f'Validation dataset size: {len(test_data)}')
    else:
        train_data = data[:int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9):]
        print('Dataset is splitted...')
    if args.max_eval_samples is not None:
        test_data = test_data[:args.max_eval_samples]

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
        if not multimodal:
            prev_data = filter_text_only(prev_data, 'previous data')
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
        optim_args=args.optim_args,
        # transformers v4 has no train_sampling_strategy; only pass it when it is not the default
        **({"train_sampling_strategy": args.train_sampling_strategy}
           if args.train_sampling_strategy != 'random' else {}),
        # transformers v5 dropped warmup_ratio; a float warmup_steps is a ratio there
        **({"warmup_steps": args.warmup_ratio} if version.parse(transformers.__version__) >= version.parse("5.0.0")
           else {"warmup_ratio": args.warmup_ratio}),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy="steps" if args.val_data_path is not None else "no",
        eval_steps=args.eval_steps or args.save_steps,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        use_cpu=False,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
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
    parser.add_argument('--data_path', type=str, nargs='+', default=['data/zero-cats.json'],
                        help='Path(s) to training data JSON file(s); multiple files are concatenated')
    parser.add_argument('--val_data_path', type=str, nargs='+', default=None,
                        help='Validation data JSON file(s); replaces the 90/10 split of data_path and enables evaluation')
    parser.add_argument('--prev_data_path', type=str, nargs='+', default=None,
                        help='Path(s) to previous task data for EWC (required if use_ewc=True)')
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
                        help='decoder-kv: keep the vision / audio encoders of Qwen3.5 / Gemma 4 / EmbeddingGemma 2; '
                             'data items may then carry "images" and "audio" lists')
    parser.add_argument('--train_media_encoders', action='store_true',
                        help='Also train the vision / audio encoders (frozen by default)')
    parser.add_argument('--use_embedding_projection', action='store_true',
                        help='EmbeddingGemma 2: feed the scorer the output of embedding_projection instead of the '
                             'final-norm hidden states (the projection scales them ~30x and destabilizes training)')
    parser.add_argument('--freeze_decoder', action='store_true',
                        help='decoder-kv: freeze the decoder backbone and train only the scorer '
                             '(and the embeddings of the added special tokens)')
    parser.add_argument('--train_last_layers', type=float, nargs='?', const=0.3, default=None,
                        help='decoder-kv: freeze the decoder except the given fraction of its last layers '
                             '(and the final norm); 0.3 when passed without a value. The backward pass stops at '
                             'the first trainable layer and, with --bf16, frozen weights are stored in bf16')

    parser.add_argument('--scorer_full_sequence', action='store_true', default=None,
                        help='decoder-kv: run the scorer encoder over the whole sequence (text + labels) '
                             'instead of only the label section')
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
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--others_lr', type=float, default=5e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.01)
    parser.add_argument('--others_weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--optim', type=str, default='adamw_torch',
                        help='transformers optimizer name, e.g. adafactor, adamw_bnb_8bit or adamw_torch_4bit '
                             'to fit larger backbones in memory')
    parser.add_argument('--optim_args', type=str, default=None,
                        help='extra optimizer args as "k1=v1,k2=v2", e.g. block_size=256 for adamw_torch_4bit')
    parser.add_argument('--train_sampling_strategy', type=str, default='random',
                        choices=['random', 'sequential', 'group_by_length', 'batch_rebalance'],
                        help='batch_rebalance: sort each optimizer step\'s samples by (estimated) length and split '
                             'them into variable-size micro-batches of balanced cost, cutting padding')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--save_only_model', action='store_true',
                        help='Save only model weights in checkpoints (no optimizer / scheduler / RNG state; training cannot be resumed from them)')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='Evaluate every N steps when val_data_path is set (default: save_steps)')
    parser.add_argument('--max_eval_samples', type=int, default=5000,
                        help='Evaluate on at most this many validation examples')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--deepspeed', type=str, default=None,
                        help='Path to a DeepSpeed config JSON, e.g. configs/deepspeed/zero2_offload.json')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Recompute activations in the backward pass to save memory')
    parser.add_argument('--logging_steps', type=int, default=100)
    
    # Augmentation parameters
    parser.add_argument('--enable_augmentation', type=bool, default=False)
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