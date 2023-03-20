import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        distances = torch.sum((embeddings1 - embeddings2) ** 2, dim=1)
        loss = torch.mean((torch.ones(len(labels), device=distances.device) - labels) * torch.max(distances - self.margin, torch.tensor(0.0, device=distances.device)))
        return loss