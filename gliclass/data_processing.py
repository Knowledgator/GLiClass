import torch
from torch.nn.utils.rnn import pad_sequence

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
                elif key_data[0].dim() == 2: # span_idx case
                    padded_batch[key] = pad_2d_tensor(key_data)
            elif isinstance(key_data[0], list):
                max_length = max(len(seq) for seq in key_data)
                padded_batch[key] = torch.tensor([seq + [0] * (max_length - len(seq)) 
                                                    for seq in key_data])
            elif type(key_data[0]) in {int, float}:
                padded_batch[key] = torch.tensor(key_data)
            else:
                raise TypeError(f"Unsupported data type: {type(key_data[0])}")
        
        return padded_batch