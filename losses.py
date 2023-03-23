import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        distances = torch.sum((embeddings1 - embeddings2) ** 2, dim=1)
        loss = torch.mean((torch.ones(len(labels), device=distances.device) - labels) * torch.max(distances - self.margin, torch.tensor(0.0, device=distances.device)))
        return loss
    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_contrastive = torch.mean((y) * torch.pow(euclidean_distance, 2) +
                                      (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive