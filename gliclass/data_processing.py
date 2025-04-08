import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class GLiClassDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, 
                            problem_type='multi_label_classification', 
                            architecture_type = 'uni-encoder',
                            prompt_first=False,
                            get_negatives = False,
                            max_labels = 50,
                            labels_tokenizer=None,
                            shuffle_labels = True):
        self.tokenizer = tokenizer
        self.labels_tokenizer = labels_tokenizer
        self.max_length = max_length
        self._data = examples
        self.problem_type = problem_type
        self.architecture_type = architecture_type
        self.prompt_first = prompt_first
        self.dataset_labels = self.collect_dataset_labels()
        self.get_negatives = get_negatives
        self.max_labels = max_labels
        self.shuffle_labels = shuffle_labels
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

    def prepare_prompt(self, example):
        prompt_texts = []
        for label in example['all_labels']:
            label_tag = f"<<LABEL>>{str(label)}"
            prompt_texts.append(label_tag)
        prompt_texts.append('<<SEP>>')
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

        if self.get_negatives and random.randint(0, 1):
            max_negatives = max(self.max_labels-len(example['all_labels']), 1)
            new_negatives = random.sample(self.dataset_labels, k=random.randint(1, max_negatives))
            example['all_labels'].extend(new_negatives)

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