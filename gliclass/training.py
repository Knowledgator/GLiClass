from typing import Optional, Tuple, Dict, List, Union, Any, Callable
from tqdm import tqdm
import numpy as np
import os
from dataclasses import dataclass, field
import torch
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
import transformers

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: Optional[float] = None
    others_weight_decay: Optional[float] = 0.0

class Trainer(transformers.Trainer):
    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        try:
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            del inputs
            torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            #     kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)

            return loss.detach() / self.args.gradient_accumulation_steps
        except Exception as e:
            print(f"Skipping iteration due to error: {e}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return torch.tensor(0.0, requires_grad=True).to(model.device)
        
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on model using inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (nn.Module):
                The model to evaluate.
            inputs (Dict[str, Union[torch.Tensor, Any]]):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument labels. Check your model's documentation for all accepted arguments.
            prediction_loss_only (bool):
                Whether or not to return the loss only.
            ignore_keys (List[str], *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        try:
            with torch.no_grad():
                loss = None
                with self.compute_loss_context_manager():
                    try:
                        outputs = model(**inputs)
                    except Exception as e:
                        raise RuntimeError(f"Error during model forward pass: {str(e)}")

                if not hasattr(outputs, 'loss'):
                    raise AttributeError("Model output does not contain 'loss' attribute")
                loss = outputs.loss

                if not hasattr(outputs, 'logits'):
                    raise AttributeError("Model output does not contain 'logits' attribute")
                logits = outputs.logits

                if 'labels' not in inputs:
                    raise KeyError("'labels' not found in input dictionary")
                labels = inputs['labels']

            if prediction_loss_only:
                return (loss, None, None)
            return (loss, logits, labels)

        except Exception as e:
            print(f"An error occurred during prediction step: {str(e)}")
            return (None, None, None)
        
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

@dataclass
class RLTrainerConfig(TrainingArguments):
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    num_rl_iters: int = field(
        default=3,
        metadata={"help": "Number of RL iterations."},
    )
    gamma: float = field(
        default=-1,
        metadata={"help": "Focal loss gamma."},
    )
    alpha: float = field(
        default=-1,
        metadata={"help": "Focal loss alpha."},
    )

class RLTrainer(Trainer):
    def __init__(
        self,
        reward_components: Optional[List[Tuple[str, Callable]]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if reward_components is None:
            # Default to F1 score as the sole reward component
            reward_components = [('f1', default_f1_reward)]
        self.reward_components = reward_components
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = {
            'total_loss': [],
            'advantages': [],
        }
        # Initialize metrics for each reward component
        for name, _ in self.reward_components.items():
            self.metrics[f'reward_{name}'] = []

    def compute_rewards(
        self,
        probs: torch.Tensor,
        actions: torch.Tensor,
        original_targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        rewards = {}
        total_reward = 0.0
        for name, reward_fn in self.reward_components.items():
            component = reward_fn(probs, actions, original_targets, valid_mask)
            rewards[name] = component
            total_reward += component
        rewards['total_reward'] = total_reward
        return rewards

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        log_prob_prev: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = targets != -100
        original_targets = targets.clone()

        probs = torch.sigmoid(inputs)

        # actions = torch.bernoulli(probs)

        actions = (probs>0.5).float()

        with torch.no_grad():
            metrics = self.compute_rewards(probs, actions, original_targets, valid_mask)
        reward = metrics['total_reward']
        advantages = reward  # Consider normalization
        self.metrics['advantages'].append(advantages.mean().item())

        for name, _ in self.reward_components.items():
            key = f'reward_{name}'
            self.metrics[key].append(metrics[name].mean().item())

        log_prob_current = original_targets * torch.log(probs + 1e-8) + (1 - original_targets) * torch.log(1 - probs + 1e-8)


        if log_prob_prev is None:
            log_prob_prev = log_prob_current.detach()

        log_probs_diff = log_prob_current - log_prob_prev
        ratio = torch.exp(log_probs_diff)

        cliprange = self.args.cliprange
        per_label_loss1 = -ratio * advantages
        per_label_loss2 = -torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
        loss_elements = torch.min(per_label_loss1, per_label_loss2)

        loss_elements = loss_elements * valid_mask
        self.metrics['total_loss'].append(loss_elements.mean().item())

        if self.args.gamma > 0:
            p_t = probs * original_targets + (1 - probs) * (1 - original_targets)
            loss_elements = loss_elements * (p_t ** self.args.gamma)

        if self.args.alpha >= 0:
            alpha_t = self.args.alpha * original_targets + (1 - self.args.alpha) * (1 - original_targets)
            loss_elements = alpha_t * loss_elements

        loss = loss_elements.sum() / valid_mask.sum()
        return loss, log_prob_current

    def _inner_training_loop(self, *args, **kwargs):
        self.create_optimizer()
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        dataloader = self.get_train_dataloader()
        device = accelerator.device

        self._init_metrics()

        for epoch in tqdm(range(args.num_train_epochs), desc="Epochs"):
            self._init_metrics()
            model.train()

            for step, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop('labels').to(device)

                prev_logps = None
                for iter in range(args.num_rl_iters):
                    outputs = model(**inputs)
                    logits = outputs.logits

                    loss, current_logps = self.compute_loss(logits, labels, log_prob_prev=prev_logps)

                    accelerator.backward(loss)
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                    prev_logps = current_logps.detach()

                if step % args.logging_steps == 0:
                    self.log_metrics()

                if args.save_steps is not None and step % args.save_steps == 0:
                    self._save_checkpoint(model, step=step)

            if args.evaluation_strategy == "epoch":
                self.evaluate()

    def log_metrics(self):
        logged_metrics = {
            'loss': np.mean(self.metrics['total_loss']),
            'advantages': np.mean(self.metrics['advantages']),
        }
        # Add user reward components
        for name, _ in self.reward_components.items():
            key = f'reward_{name}'
            logged_metrics[key] = np.mean(self.metrics[key])
        self.log(logged_metrics)
        self._init_metrics()

    def _save_checkpoint(self, model, step=None):
        checkpoint_dir = f"checkpoint-{step}" if step else "final_model"
        output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        print(f"Checkpoint saved to {output_dir}")