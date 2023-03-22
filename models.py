import json
import random
import numpy as np
import torch
import transformers
import torch.nn as nn
from losses import HingeLoss, ContrastiveLoss
from collections import defaultdict
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Model(nn.Module):
    def __init__(self, name: str, num_choices: int, device: str):
        super().__init__()
        
        self.name = name
        self.num_choices = num_choices
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)

        self.max_length = 512
        
        if "base" in name:
            self.hidden_size = 768
        elif "xx-large" in name:
            self.hidden_size = 1536
        elif "large" in name:
            self.hidden_size = 1024
        elif "tiny" in name:
            self.hidden_size = 128
        elif "small" in name:
            self.hidden_size = 768
        elif "aristo" in name:
            self.hidden_size = 768
            
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.scorer = nn.Linear(self.hidden_size, 2)
        
    def score_input(self, content):
        batch = self.tokenizer(
            content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        out = self.model(
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
            output_hidden_states=True
        )
        return torch.mean(out["hidden_states"][-1], dim=1)

    def forward(self, batch):
        content, labels = batch
        embed = self.score_input(content)
        logits = self.scorer(embed)
        labels = torch.tensor(labels, dtype=torch.long).to(logits.device)
        loss = self.ce_loss_func(logits, labels)
        preds_cls = list(torch.argmax(logits, 1).cpu().numpy())
        positive_logits = logits[:, 1]
        
        if self.num_choices!=-1:
            preds = torch.argmax(positive_logits.reshape(-1, self.num_choices), 1)
            preds = list(preds.cpu().numpy())
        else:
            preds = list(positive_logits.detach().cpu().numpy())
        
        #print(preds)
        #print(preds_cls)
        #print(positive_logits)
        return loss, preds, preds_cls

class MultiModel(nn.Module):
    def __init__(self, name: str, num_choices: int, device: str):
        super().__init__()
        
        self.name = name
        self.num_choices = num_choices
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
        
        self.max_length = 512
        
        if "base" in name:
            self.hidden_size = 768
        elif "xx-large" in name:
            self.hidden_size = 1536
        elif "large" in name:
            self.hidden_size = 1024
        elif "tiny" in name:
            self.hidden_size = 128
        elif "small" in name:
            self.hidden_size = 768
        elif "aristo" in name:
            self.hidden_size = 768
            
        self.img_linear = nn.Linear(in_features=1000, out_features=self.hidden_size, bias=True)
        
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.scorer = nn.Linear(2*self.hidden_size, 2)
        
    def score_input(self, content):
        batch = self.tokenizer(
            content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        out = self.text_model(
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
            output_hidden_states=True
        )
        return torch.mean(out["hidden_states"][-1], dim=1)

    def forward(self, batch):
        content, img_embeds, labels = batch
        img_embeds = torch.Tensor(img_embeds).to(self.device)
        
        content_embed = self.score_input(content)
        
        image_embed = self.img_linear(img_embeds)
                
        embed = torch.cat((content_embed, image_embed), dim=1)
        
        logits = self.scorer(embed)
        
        labels = torch.tensor(labels, dtype=torch.long).to(logits.device)
        loss = self.ce_loss_func(logits, labels)
        preds_cls = list(torch.argmax(logits, 1).cpu().numpy())
        positive_logits = logits[:, 1]
        
        if self.num_choices!=-1:
            preds = torch.argmax(positive_logits.reshape(-1, self.num_choices), 1)
            preds = list(preds.cpu().numpy())
        else:
            preds = list(positive_logits.detach().cpu().numpy())
        
        #print(preds)
        #print(preds_cls)
        #print(positive_logits)
        return loss, preds, preds_cls    
    
    
class ContrastiveModel(nn.Module):
    def __init__(self, name: str, num_choices: int, out_size: int, device: str):
        super().__init__()

        self.name = name
        self.num_choices = num_choices
        self.out_size = out_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)

        if "base" in name:
            self.hidden_size = 768
        elif "xx-large" in name:
            self.hidden_size = 1536
        elif "large" in name:
            self.hidden_size = 1024
        elif "tiny" in name:
            self.hidden_size = 128
        elif "small" in name:
            self.hidden_size = 768
        elif "aristo" in name:
            self.hidden_size = 768
            
        
        self.scorer = nn.Linear(self.hidden_size, out_size)
        
        self.hinge_loss_func = HingeLoss(margin=0.0)
        self.triplet_loss_func = nn.TripletMarginLoss(margin=1.0, p=2)        
        self.ce_loss_func = nn.CosineEmbeddingLoss(margin=-1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.contrastive_loss_func = ContrastiveLoss()

    def get_embeddings(self, text, max_length=512):
        batch = self.tokenizer(
            text, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        out = self.model(
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
            output_hidden_states=True
        )
        return torch.mean(out["hidden_states"][-1], dim=1)

    def forward(self, questions, choices, labels):
        #labels = [-1 if x==0 else 1 for x in labels]
        labels = torch.Tensor(labels).to(self.device)

        question_embed = self.get_embeddings(questions, max_length=200)
        question_embed = self.scorer(question_embed)

        choice_embed = self.get_embeddings(choices, max_length=20)
        choice_embed = self.scorer(choice_embed)

        #loss = self.ce_loss_func(question_embed, choice_embed, labels)
        loss = self.contrastive_loss_func(question_embed, choice_embed, labels)
        
        similarities = self.cos(question_embed, choice_embed)
        
        return loss, similarities
