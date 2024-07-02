import torch
from torch import nn

class ScorerWeightedDot(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        self.proj_text = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 1)  # start, end, score
        )

    def forward(self, text_rep, label_rep):
        batch_size, hidden_size = text_rep.shape
        num_classes = label_rep.shape[1]

        # (batch_size, 1, 3, hidden_size)
        text_rep = self.proj_text(text_rep).view(batch_size, 1, 1, 2, hidden_size)
        label_rep = self.proj_label(label_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        # (2, batch_size, 1, num_classes, hidden_size)
        text_rep = text_rep.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, 1, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # (batch_size, 1, num_classes, hidden_size * 3)
        cat = torch.cat([text_rep[0], label_rep[0], text_rep[1] * label_rep[1]], dim=-1)

        # (batch_size, num_classes)
        scores = self.out_mlp(cat).view(batch_size, num_classes)

        return scores
    
class ScorerDot(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, text_rep, label_rep):
        # dot product with einsum
        scores = torch.einsum('BD,BCD->BC', text_rep, label_rep)
        return scores
    
class MLPScorer(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=256):
        super().__init__()
        
        # Calculate the input size for the MLP
        total_input_size = hidden_size*2
        
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1)
        )

    def forward(self, text_rep, label_rep):
        # Concatenate text and label representations
        batch_size, num_labels, dim = label_rep.shape
        text_rep = text_rep.unsqueeze(1).expand(batch_size, num_labels, dim)
        combined_rep = torch.cat([text_rep, label_rep], dim=-1)
        
        # Pass through MLP
        scores = self.mlp(combined_rep).squeeze(-1)
        
        return scores
    
SCORER2OBJECT = {"weighted-dot": ScorerWeightedDot, 'simple': ScorerDot, 'mlp': MLPScorer}