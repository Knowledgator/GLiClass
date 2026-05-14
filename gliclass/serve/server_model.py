"""Server-optimized model wrapper for GLiClass with low-level batch processing."""

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from gliclass.model import GLiClassModel
from gliclass.pipeline import format_examples_prompt

logger = logging.getLogger(__name__)


class GLiClassServerModel:
    """Server-optimized wrapper combining pipeline logic and model.

    Splits inference into low-level stages for CPU/GPU parallelism:
    - prepare_batch (CPU): text formatting, label flattening
    - tokenize_batch (CPU): tokenization, input creation
    - collate_batch (CPU): tensor creation, padding
    - run_batch (GPU): forward pass, encoding
    - decode_batch (CPU/GPU): sigmoid, threshold filtering, result formatting

    Avoids DataLoader overhead by keeping tokenizers/collators alive
    and calling model methods directly.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 2048,
        max_classes: int = 25,
        max_labels_alloc: str | int = "dynamic",  # "dynamic" | int | "fixed"
    ):
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.max_classes = max_classes
        self.max_labels_alloc = max_labels_alloc

        logger.info(f"Loading model: {model_name}")
        self.model = GLiClassModel.from_pretrained(model_name)
        self.model.config.max_labels_alloc = max_labels_alloc
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

        self.architecture_type = self.model.config.architecture_type
        self.prompt_first = getattr(self.model.config, "prompt_first", False)

        logger.info(f"Loading tokenizers for {self.architecture_type}")
        # Use model_name, not encoder_model_name, to get tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.architecture_type in {"bi-encoder", "bi-encoder-fused"}:
            # For bi-encoder, load labels tokenizer from model (not label_model_name)
            # to ensure special tokens are available
            self.labels_tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.labels_tokenizer = None

        self.label_token = "<<LABEL>>"
        self.sep_token = "<<SEP>>"
        self.example_token = "<<EXAMPLE>>"

    def _resolve_max_num_classes(self, labels: list[str] | list[list[str]], same_labels: bool) -> int | None:
        if self.max_labels_alloc == "dynamic":
            return len(labels) if same_labels else max(len(lbl) for lbl in labels)
        if isinstance(self.max_labels_alloc, int):
            return self.max_labels_alloc
        return None  # 'fixed': model uses config.max_num_classes

    def prepare_batch(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]],
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
    ) -> dict[str, Any]:
        same_labels = isinstance(labels[0], str) if labels else True

        prepared = {
            "texts": texts,
            "labels": labels,
            "same_labels": same_labels,
            "examples_str": None,
        }

        if examples:
            prepared["examples_str"] = format_examples_prompt(
                examples,
                example_token=self.example_token,
                sep_token=self.sep_token,
            )

        if prompt is not None:
            if isinstance(prompt, str):
                prepared["prompts"] = [prompt] * len(texts)
            else:
                prepared["prompts"] = prompt
        else:
            prepared["prompts"] = [None] * len(texts)

        return prepared

    def _format_input_bi_encoder_fused(
        self,
        text: str,
        labels: list[str],
        examples_str: str | None,
        prompt: str | None,
    ) -> str:
        input_parts = []

        if prompt:
            input_parts.append(prompt)
            input_parts.append(" ")

        for _label in labels:
            input_parts.append(self.label_token)
        input_parts.append(self.sep_token)

        examples_str = examples_str or ""

        if self.prompt_first:
            return "".join(input_parts) + text + examples_str
        else:
            return text + "".join(input_parts) + examples_str

    def _format_input_bi_encoder(
        self,
        text: str,
        prompt: str | None,
    ) -> str:
        if prompt:
            return f"{prompt} {text}"
        return text

    def _format_input_uni_encoder(
        self,
        text: str,
        labels: list[str],
        examples_str: str | None,
        prompt: str | None,
    ) -> str:
        """Match training format from pipeline.prepare_input"""
        input_parts = []

        # Add labels
        for label in labels:
            input_parts.append(f"{self.label_token}{label}")
        input_parts.append(self.sep_token)

        # Add task description prompt
        if prompt:
            input_parts.append(prompt)

        # Format examples
        examples_str = examples_str or ""

        # Use prompt_first to determine order (match pipeline logic)
        if self.prompt_first:
            return "".join(input_parts) + text + examples_str
        else:
            return text + "".join(input_parts) + examples_str

    def tokenize_batch(self, prepared: dict[str, Any]) -> dict[str, Any]:
        texts = prepared["texts"]
        labels = prepared["labels"]
        same_labels = prepared["same_labels"]
        examples_str = prepared["examples_str"]
        prompts = prepared["prompts"]

        if self.architecture_type == "bi-encoder-fused":
            inputs = []
            for i, text in enumerate(texts):
                curr_labels = labels if same_labels else labels[i]
                inputs.append(self._format_input_bi_encoder_fused(text, curr_labels, examples_str, prompts[i]))
        elif self.architecture_type == "bi-encoder":
            inputs = [self._format_input_bi_encoder(text, prompts[i]) for i, text in enumerate(texts)]
        elif self.architecture_type in {"uni-encoder", "encoder-decoder", "encoder-decoder-cls"}:
            inputs = []
            for i, text in enumerate(texts):
                curr_labels = labels if same_labels else labels[i]
                inputs.append(self._format_input_uni_encoder(text, curr_labels, examples_str, prompts[i]))
        else:
            raise NotImplementedError(f"Architecture {self.architecture_type} not implemented")

        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )

        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "same_labels": same_labels,
            "num_samples": len(texts),
        }

        if self.labels_tokenizer is not None:
            if same_labels:
                tokenized_labels = self.labels_tokenizer(
                    labels,
                    truncation=True,
                    max_length=self.max_length,
                    padding="longest",
                    return_tensors="pt",
                )
                result["class_input_ids"] = tokenized_labels["input_ids"].unsqueeze(0).expand(len(texts), -1, -1)
                result["class_attention_mask"] = (
                    tokenized_labels["attention_mask"].unsqueeze(0).expand(len(texts), -1, -1)
                )
                result["labels_mask"] = torch.ones(len(texts), len(labels), dtype=torch.long)
            else:
                class_input_ids = []
                class_attention_mask = []
                for labels_set in labels:
                    tokenized_labels = self.labels_tokenizer(
                        labels_set,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    class_input_ids.append(tokenized_labels["input_ids"])
                    class_attention_mask.append(tokenized_labels["attention_mask"])
                result["class_input_ids"] = torch.stack(class_input_ids)
                result["class_attention_mask"] = torch.stack(class_attention_mask)

        return result

    def collate_batch(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        batch = {}
        for key, value in tokenized.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            else:
                batch[key] = value
        return batch

    @torch.inference_mode()
    def run_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        labels = batch["labels"]
        same_labels = batch["same_labels"]
        max_num_classes = self._resolve_max_num_classes(labels, same_labels)

        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "max_num_classes": max_num_classes,
        }

        if "class_input_ids" in batch:
            model_inputs["class_input_ids"] = batch["class_input_ids"]
            model_inputs["class_attention_mask"] = batch["class_attention_mask"]

        if "labels_mask" in batch:
            model_inputs["labels_mask"] = batch["labels_mask"]

        outputs = self.model(**model_inputs)

        return {
            "logits": outputs.logits,
            "labels": labels,
            "same_labels": same_labels,
            "num_samples": batch["num_samples"],
        }

    def decode_batch(
        self,
        outputs: dict[str, Any],
        threshold: float = 0.5,
        multi_label: bool = True,
    ) -> list[dict[str, Any]]:
        logits = outputs["logits"]
        labels = outputs["labels"]
        same_labels = outputs["same_labels"]
        num_samples = outputs["num_samples"]

        results = []

        if multi_label:
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits)
            probs_cpu = probs.float().cpu().numpy()

            for i in range(num_samples):
                curr_labels = labels if same_labels else labels[i]
                text_results = []

                for j in range(len(curr_labels)):
                    score = float(probs_cpu[i, j])
                    if score >= threshold:
                        text_results.append({"label": curr_labels[j], "score": score})

                results.append(text_results)
        else:
            for i in range(num_samples):
                curr_labels = labels if same_labels else labels[i]
                scores = torch.softmax(logits[i][: len(curr_labels)], dim=-1)
                scores_cpu = scores.float().cpu().numpy()  # Convert to float32 for numpy
                max_idx = int(scores_cpu.argmax())

                results.append(
                    [
                        {
                            "label": curr_labels[max_idx],
                            "score": float(scores_cpu[max_idx]),
                        }
                    ]
                )

        return results

    def predict(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]],
        threshold: float = 0.5,
        multi_label: bool = True,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        prepared = self.prepare_batch(texts, labels, examples, prompt)
        tokenized = self.tokenize_batch(prepared)
        batch = self.collate_batch(tokenized)
        outputs = self.run_batch(batch)
        results = self.decode_batch(outputs, threshold, multi_label)
        return results
