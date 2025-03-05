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
