import random
import torch
import copy
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = True
    
    # Probability for each augmentation type
    random_label_removal_prob: float = 0.15
    random_label_addition_prob: float = 0.10
    random_text_addition_prob: float = 0.05
    random_add_description_prob: float = 0.25
    random_add_synonyms_prob: float = 0.1
    random_add_examples_prob: float = 0.25
    max_num_examples: int = 5

class DataAugmenter:
    def __init__(self, config, examples, labels, label2description=None):
        self.config = config
        self.examples = examples
        self.labels = sorted(labels)
        self.max_examples = self.config.max_num_examples
        self.label2description = label2description or {}
    
    def remove_labels(self, true_labels, all_labels):
        if len(all_labels)<=1:
            return true_labels, all_labels
        k = random.randint(1, len(all_labels))
        all_labels = random.sample(all_labels, k=k)
        true_labels = [lbl for lbl in true_labels if lbl in all_labels]
        return true_labels, all_labels

    def add_random_labels(self, all_labels):
        if not self.labels:
            return all_labels
        num_add = len(all_labels) + 1
        k = random.randint(1, min(num_add, len(self.labels)))
        add_labels = random.sample(self.labels, k=k)
        all_labels.extend(add_labels)
        return all_labels
    
    def add_random_text(self, text, all_labels):
        if not self.examples:
            return text
        example = random.sample(self.examples, k=1)[0]
        curr_labels = example['all_labels']
        joint_labels = set(all_labels) & set(curr_labels)
        if len(joint_labels):
            return text
        else:
            if random.randint(0, 1):
                text = example['text'] + ' ' + text
            else:
                text = text + ' ' + example['text']
            return text
    
    def add_random_synonyms(self, all_labels):
        """Replace some labels with their synonyms if available."""
        if not self.label2description:
            return all_labels
        
        augmented_labels = []
        for label in all_labels:
            if label in self.label2description:
                label_info = self.label2description[label]
                synonyms = label_info.get('synonyms', [])
                
                if synonyms and random.random() < 0.5:
                    augmented_labels.append(random.choice(synonyms))
                else:
                    augmented_labels.append(label)
            else:
                augmented_labels.append(label)
        
        return augmented_labels
    
    def add_random_descriptions(self, item):
        """Add descriptions to labels in the text or metadata."""
        if not self.label2description or not item['all_labels']:
            return item
        
        max_labels = min(3, len(item['all_labels']))
        labels_to_describe = random.sample(
            item['all_labels'], 
            k=random.randint(1, max_labels)
        )
        
        descriptions = []
        for label in labels_to_describe:
            if label in self.label2description:
                label_info = self.label2description[label]
                desc_list = label_info.get('descriptions', [])
                if desc_list:
                    descriptions.append(f"{label}: {random.choice(desc_list)}")
        
        if descriptions:
            desc_text = ' '.join(descriptions)
            if random.random() < 0.5:
                item['text'] = desc_text + ' ' + item['text']
            else:
                item['text'] = item['text'] + ' ' + desc_text
        
        return item
    
    def add_random_examples(self, item):
        """Add example texts with similar labels."""
        if not item['all_labels']:
            return item
        
        candidate_examples = item.get("examples", [])

        item_label_set = set(item['all_labels'])

        if not candidate_examples:
            
            for example in self.examples:
                example_label_set = set(example['true_labels'])
                example_text = example['text']

                overlap = item_label_set & example_label_set
                
                # Only consider examples with at least one overlapping label
                if overlap:
                    candidate_examples.append({"text": example_text, "labels": list(example_label_set)})
        
        if not candidate_examples:
            return item
        
        # Sort by overlap and select top examples
        random.shuffle(candidate_examples)
        top_candidates = candidate_examples[:self.max_examples]

        num_examples = random.randint(1, min(2, len(top_candidates)))
        selected_examples = random.sample(top_candidates, k=num_examples)
        
        item['examples'] = selected_examples
        
        return item
    
    def augment(self, item):
        if not self.config.enabled:
            return item

        text = copy.deepcopy(item['text'])
        true_labels = copy.deepcopy(item['true_labels'])
        all_labels = copy.deepcopy(item['all_labels'])

        # Create augmented item
        aug_item = {
            'text': text,
            'true_labels': true_labels,
            'all_labels': all_labels
        }
        
        # Copy any additional fields
        for key in item:
            if key not in aug_item:
                aug_item[key] = copy.deepcopy(item[key])

        if random.random() < self.config.random_label_removal_prob:
            aug_item['true_labels'], aug_item['all_labels'] = self.remove_labels(
                aug_item['true_labels'], aug_item['all_labels']
            )
    
        if random.random() < self.config.random_label_addition_prob:
            aug_item['all_labels'] = self.add_random_labels(aug_item['all_labels'])

        if random.random() < self.config.random_text_addition_prob:
            aug_item['text'] = self.add_random_text(aug_item['text'], aug_item['all_labels'])
        
        if random.random() < self.config.random_add_synonyms_prob:
            aug_item['all_labels'] = self.add_random_synonyms(aug_item['all_labels'])
        
        if random.random() < self.config.random_add_description_prob:
            aug_item = self.add_random_descriptions(aug_item)
        
        if random.random() < self.config.random_add_examples_prob:
            aug_item = self.add_random_examples(aug_item)

        return aug_item
    
class GLiClassDataset(Dataset):
    def __init__(self, examples, tokenizer, augment_config,
                            label2description={},
                            max_length=512, 
                            problem_type='multi_label_classification', 
                            architecture_type = 'uni-encoder',
                            add_description=True,
                            prompt_first=False,
                            get_negatives = False,
                            max_labels = 50,
                            labels_tokenizer=None,
                            shuffle_labels = True):
        self.tokenizer = tokenizer
        self.labels_tokenizer = labels_tokenizer
        self.label2description = label2description
        self.augment_config = augment_config
        self.max_length = max_length
        self._data = examples
        self.add_description = add_description
        self.problem_type = problem_type
        self.architecture_type = architecture_type
        self.prompt_first = prompt_first
        self.dataset_labels = self.collect_dataset_labels()
        self.get_negatives = get_negatives
        self.max_labels = max_labels
        self.shuffle_labels = shuffle_labels
        
        self.sep_token = "<<SEP>>"
        self.label_token = "<<LABEL>>"
        self.example_token = "<<EXAMPLE>>"
        self.augmenter = DataAugmenter(augment_config, examples, self.dataset_labels, label2description)
        print('Total labels: ', len(self.dataset_labels))
    
    def collect_dataset_labels(self):
        dataset_labels = set()
        for example in self._data:
            dataset_labels.update(set(example['all_labels']))
        return dataset_labels
    
    def prepare_labels(self, example, label2idx, problem_type):
        if problem_type == 'single_label_classification':
            labels = label2idx[example['true_labels'][0]]
        elif problem_type == 'multi_label_classification':
            if isinstance(example['true_labels'], dict):
                labels = [example['true_labels'][label] if label in example['true_labels'] else 0. for label in example['all_labels']]
            else:
                labels = [1. if label in example['true_labels'] else 0. for label in example['all_labels']]
        else:
            raise NotImplementedError(f"{problem_type} is not implemented.")
        return torch.tensor(labels)

    def prepare_prompt(self, item):
        prompt_texts = []
        for label in item['all_labels']:
            label_tag = f"{self.label_token}{str(label)}"
            prompt_texts.append(label_tag)
        prompt_texts.append(self.sep_token)
        examples = item.get("examples", [])
        if examples:
            for example in examples:
                prompt_texts.append(self.example_token)
                prompt_texts.append(example.get("text", ""))
                prompt_texts.append(" \nLabels:\n ")
                prompt_texts.append(', '.join(example.get("true_labels", [])))
        prompt_texts.append(self.sep_token)
        return prompt_texts
    
    def tokenize(self, texts):
        tokenized_inputs = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding="longest")
        return tokenized_inputs

    def tokenize_labels(self, labels):
        tokenized_inputs = self.labels_tokenizer(labels, truncation=True, max_length=self.max_length, padding="longest")
        return tokenized_inputs
    
    def tokenize_and_prepare_labels_for_uniencoder(self, example):
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        input_text = self.prepare_prompt(example)
        if self.prompt_first:
            input_text = ''.join(input_text)+str(example['text'])
        else:
            input_text = str(example['text'])+''.join(input_text)
        label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

        tokenized_inputs = self.tokenize(input_text)
        tokenized_inputs['labels'] = self.prepare_labels(example, label2idx, self.problem_type)
        tokenized_inputs['labels_text'] =  example['all_labels']
        tokenized_inputs['input_texts'] = example['text']
        return tokenized_inputs

    def tokenize_and_prepare_labels_for_encoder_decoder(self, example):
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        class_texts = self.prepare_prompt(example)
        class_texts = ''.join(class_texts)

        label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

        tokenized_inputs = self.tokenize(example['text'])
        tokenized_classes = self.tokenize(class_texts)
        tokenized_inputs["class_input_ids"] = tokenized_classes["input_ids"]
        tokenized_inputs["class_attention_mask"] = tokenized_classes["attention_mask"]   
        tokenized_inputs['labels'] = self.prepare_labels(example, label2idx, self.problem_type)
        return tokenized_inputs

    def tokenize_and_prepare_labels_for_biencoder(self, example):
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        def prepare_prompt(labels):
            prompt_texts = []
            for label in labels:
                label_tag = f"<<LABEL>>"
                prompt_texts.append(label_tag)
            prompt_texts.append('<<SEP>>')
            return ''.join(prompt_texts)

        input_text = example['text']
        class_texts = example['all_labels']

        if self.architecture_type == 'bi-encoder-fused':
            prompt = prepare_prompt(class_texts)
            if self.prompt_first:
                input_text = f"{prompt} {input_text}"
            else:
                input_text = f"{input_text} {prompt}"

        tokenized_inputs = self.tokenize(input_text)
        tokenized_classes = self.tokenize_labels(class_texts)

        tokenized_inputs["class_input_ids"] = torch.tensor(tokenized_classes["input_ids"])
        tokenized_inputs["class_attention_mask"] = torch.tensor(tokenized_classes["attention_mask"])

        label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

        tokenized_inputs['labels_mask'] = torch.ones(len(class_texts))
        tokenized_inputs['labels'] = self.prepare_labels(example, label2idx, self.problem_type)
        return tokenized_inputs

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        example = self._data[idx]

        example = self.augmenter.augment(example)

        if self.architecture_type == 'uni-encoder':
            model_inputs = self.tokenize_and_prepare_labels_for_uniencoder(example)
        elif self.architecture_type == 'encoder-decoder':
            model_inputs = self.tokenize_and_prepare_labels_for_encoder_decoder(example)
        elif self.architecture_type in {'bi-encoder', 'bi-encoder-fused'}:
            model_inputs = self.tokenize_and_prepare_labels_for_biencoder(example)
        else:
            raise NotImplementedError('This architecture type is not implemented.')
        return model_inputs
    

def pad_2d_tensor(key_data):
    """
    Pad a list of 2D tensors to have the same size along both dimensions.
    
    :param key_data: List of 2D tensors to pad.
    :return: Tensor of padded tensors stacked along a new batch dimension.
    """
    if not key_data:
        raise ValueError("The input list 'key_data' should not be empty.")

    # Determine the maximum size along both dimensions
    max_rows = max(tensor.shape[0] for tensor in key_data)
    max_cols = max(tensor.shape[1] for tensor in key_data)
    
    tensors = []

    for tensor in key_data:
        rows, cols = tensor.shape
        row_padding = max_rows - rows
        col_padding = max_cols - cols
        # Pad the tensor along both dimensions
        padded_tensor = torch.nn.functional.pad(tensor, (0, col_padding, 0, row_padding),
                                                                 mode='constant', value=0)
        tensors.append(padded_tensor)

    # Stack the tensors into a single tensor along a new batch dimension
    padded_tensors = torch.stack(tensors)

    return padded_tensors

class DataCollatorWithPadding:
    def __init__(self, device = 'cuda:0'):
        self.device = device

    def __call__(self, batch):
        keys = batch[0].keys()
        padded_batch = {key: [] for key in keys}
        
        for key in keys:
            key_data = [item[key] for item in batch]
            if isinstance(key_data[0], torch.Tensor):
                if  key_data[0].dim() == 1:
                    padded_batch[key] = pad_sequence(key_data, batch_first=True)
                elif key_data[0].dim() == 2: 
                    padded_batch[key] = pad_2d_tensor(key_data)
            elif isinstance(key_data[0], list):
                data_el = "string"
                if len(key_data[0]):
                    data_el = key_data[0][0]
                if isinstance(data_el, str):
                    padded_batch[key] = key_data
                else:
                    max_length = max(len(seq) for seq in key_data)
                    padded_batch[key] = torch.tensor([seq + [0] * (max_length - len(seq)) 
                                                        for seq in key_data])
            elif type(key_data[0]) in {int, float}:
                padded_batch[key] = torch.tensor(key_data)
            elif isinstance(key_data[0], str):
                padded_batch[key] = key_data
            else:
                raise TypeError(f"Unsupported data type: {type(key_data[0])}")
        
        return padded_batch