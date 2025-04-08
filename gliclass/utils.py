import torch

def is_module_available(module_name):
    """
    Checks whether the specified Python module is available.
    
    Args:
        module_name (str): The name of the module to check.
        
    Returns:
        bool: True if the module is available, False otherwise.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

class MissedPackageException(Exception):
    """Raised when the requested decoder model is not supported."""
    pass


def retrieval_augmented_text(text: str, examples: list) -> str:
    """
    Constructs a new text by appending relevant retrieved examples to the input text.

    Args:
        text (str): The main input text.
        examples (list): A list of examples in the format
                         {"text": str, "true_labels": List[str], "all_labels": List[str]}.

    Returns:
        str: The modified text with relevant examples appended.
    """
    if not examples:
        return text

    retrieved_examples = []
    all_labels = set(label for example in examples for label in example.get("true_labels", []))
    relevant_examples = [ex for ex in examples if set(ex.get("true_labels", [])) & all_labels]

    for example in relevant_examples:
        example_text = example["text"]
        true_labels = example.get("true_labels", [])
        all_labels = example.get("all_labels", [])

        false_labels = list(set(all_labels) - set(true_labels))

        true_labels_str = " ".join([f"<<TRUE_LABEL>> {label}" for label in true_labels])
        false_labels_str = " ".join([f"<<FALSE_LABEL>> {label}" for label in false_labels])

        retrieved_example_str = f"<<EXAMPLE>> {example_text} {true_labels_str} {false_labels_str} <</EXAMPLE>>"
        retrieved_examples.append(retrieved_example_str)

    augmented_text = f"{text} {' '.join(retrieved_examples)}" if retrieved_examples else text

    return augmented_text

def default_f1_reward(
    probs: torch.Tensor,
    actions: torch.Tensor,
    original_targets: torch.Tensor,
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    A variant that extracts list-of-indices sets and then calculates
    the F1 score in a classical manner. Returns shape (N, 1).
    
    Args:
        probs:              (N, T) Tensor of probabilities (not used here but left for interface consistency).
        actions:            (N, T) Tensor of predicted labels in {0, 1}.
        original_targets:   (N, T) Tensor of ground-truth labels in {0, 1}.
        valid_mask:         (N, T) Tensor indicating which positions are valid (1) vs. invalid (0).

    Returns:
        f1_scores: (N, 1) Tensor containing the F1 score for each row.
    """
    N = actions.shape[0]
    f1_scores = []

    for i in range(N):
        # Filter valid positions
        valid_preds_i = actions[i] * valid_mask[i]
        valid_targets_i = original_targets[i] * valid_mask[i]

        # Get the set of indices where we predicted 1
        predicted_set = set((valid_preds_i == 1).nonzero(as_tuple=True)[0].tolist())
        # Get the set of indices where the ground truth is 1
        target_set = set((valid_targets_i == 1).nonzero(as_tuple=True)[0].tolist())

        # Compute intersection
        intersection = predicted_set.intersection(target_set)

        # Precision
        if len(predicted_set) > 0:
            precision = len(intersection) / len(predicted_set)
        else:
            precision = 0.0

        # Recall
        if len(target_set) > 0:
            recall = len(intersection) / len(target_set)
        else:
            recall = 0.0

        # F1 score
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)
    
    # Convert list to tensor shape (N, 1)
    f1_scores = torch.tensor(f1_scores, dtype=torch.float).unsqueeze(-1)
    return f1_scores.detach().to(probs.device)