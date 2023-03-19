import json
import time
import random
import pickle
import gc, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from models import Model
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler

from collections import defaultdict
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

class ScienceQADataset(Dataset):
    def __init__(self, data, shuffle):
        data = pd.DataFrame(data)[["question","choices","answer","hint"]]
        if shuffle:
            data = data.sample(frac=1)

        self.question = list(data["question"].values)
        self.choices = list(data["choices"].values)
        self.hint = list(data["hint"].values)
        self.answer = list(data["answer"].values)
        
        print(len(self.question))
        
    def __len__(self):
        return len(self.question)

    def __getitem__(self, index):
        s1, s2, s3, s4 = self.question[index], self.choices[index], self.hint[index], self.answer[index]
        return s1, s2, s3, s4
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def configure_dataloaders(train_batch_size=16, eval_batch_size=16, shuffle=False):
    "Prepare dataloaders"
    data = load_from_disk("data/scienceqa/scienceQA")

    train_dataset = ScienceQADataset(data["train"], True)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size, collate_fn=train_dataset.collate_fn)

    val_dataset = ScienceQADataset(data["validation"], False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)
    
    test_dataset = ScienceQADataset(data["test"], False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def configure_optimizer(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer

def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler

def expand_batch(batch, input_format):
    content, labels = [], []
    qs, cs, hs, ans = batch
    for i, (q, choices, h, a) in enumerate(zip(*batch)):
        #Plain text
        if input_format == "0":
            for c in choices:
                content.append("{} {}".format(q,c))

        #Add prompts for question and option
        elif input_format == "1":
            for c in choices:
                content.append("Question: {}, Option: {}". format(q, c))

        #Add prompts for question, hint and option
        elif input_format=="2":
            for c in choices:
                content.append("Question: {}, Hint: {}, Option: {}". format(q, h, c))

        y = [0]*len(choices)
        y[a] = 1
        labels += y
        
    return content, labels

def regroup_batch(sizes, l, p):
    it_l, it_p = iter(l), iter(p)
    labels = [np.argmax(list(itertools.islice(it_l, size))) for size in sizes]
    preds = [np.argmax(list(itertools.islice(it_p, size))) for size in sizes]
        
    return labels, preds

def train_or_eval_model(model, dataloader, optimizer=None, input_format="0", split="Train"):
    losses, labels, preds, preds_cls, labels_cls,  = [], [], [], [], []
    if split=="Train":
        model.train()
    else:
        model.eval()
            
    for batch in tqdm(dataloader, leave=False):
        if split=="Train":
            optimizer.zero_grad()
            
        #Store length of choices to re-group them later
        sizes = [len(x) for x in batch[1]]
        
        content, l_cls = expand_batch(batch, input_format)
        loss, p, p_cls = model((content, l_cls))
        
        #Regroup
        ls, ps = regroup_batch(sizes, l_cls, p)
        
        labels.append(ls)
        preds.append(ps)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)
        
        if split=="Train":
            # wandb.log({"Train Step Loss": loss})
            loss.backward()
            optimizer.step()
        # elif split=="Val":
        #    wandb.log({"Val Step Loss": loss})
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    
    if split=="Train":
        wandb.log({"Train Loss": avg_loss})
        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)
        wandb.log({"Train CLS Accuracy": acc})
        
        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = [item for sublist in labels for item in sublist]
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
        wandb.log({"Train Instance Accuracy": instance_acc})
        
        return avg_loss, acc, instance_acc, f1
    
    elif split=="Val":
        wandb.log({"Val Loss": avg_loss})
        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)
        wandb.log({"Val CLS Accuracy": acc})
        
        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = [item for sublist in labels for item in sublist]
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
        wandb.log({"Val Instance Accuracy": instance_acc})
        
        return avg_loss, acc, instance_acc, f1
    
    elif "Test" in split:
        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)
        wandb.log({"Test CLS Accuracy": acc})
        
        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = [item for sublist in labels for item in sublist]
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
        wandb.log({"Test Instance Accuracy": instance_acc})
        
        instance_preds = [str(item) for item in instance_preds]
        print ("Test preds frequency:", dict(pd.Series(instance_preds).value_counts()))

        return instance_preds, avg_loss, acc, instance_acc, f1
    
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--name", default="roberta-large", help="Which model.")
    parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle train data such that positive and negative \
        sequences of the same question are not necessarily in the same batch.")
    parser.add_argument("--device", default="cuda", help="Which accelerator to use.")
    parser.add_argument('--input-format', default="1", help="How to format the input data.")
    
    global args
    args = parser.parse_args()
    print(args)
    
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle
    device = args.device
    input_format = args.input_format
    
    num_choices = -1
    vars(args)["num_choices"] = num_choices
    
    model = Model(
        name=name,
        num_choices=num_choices,
        device = device
    ).to(device)
    
    sep_token = model.tokenizer.sep_token
    
    optimizer = configure_optimizer(model, args)
    
    if "/" in name:
        sp = name[name.index("/")+1:]
    else:
        sp = name
    
    exp_id = str(int(time.time()))
    vars(args)["exp_id"] = exp_id
    rs = "Acc: {}"
    
    path = "saved/scienceqa/" + exp_id + "/" + name.replace("/", "-")
    Path("saved/scienceqa/" + exp_id + "/").mkdir(parents=True, exist_ok=True)
    
    fname = "saved/scienceqa/" + exp_id + "/" + "args.txt"
    
    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()
        
    Path("results/scienceqa/").mkdir(parents=True, exist_ok=True)
    lf_name = "results/scienceqa/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()

    wandb.init(project="ScienceQA-" + sp)
    wandb.watch(model)
    
    train_loader, val_loader, test_loader = configure_dataloaders(
            train_batch_size, eval_batch_size, shuffle
        )
    
    for e in range(epochs):    
        
        train_loss, train_acc, train_ins_acc, train_f1 = train_or_eval_model(model, train_loader, optimizer, input_format=input_format, split="Train")
        val_loss, val_acc, val_ins_acc, val_f1 = train_or_eval_model(model, val_loader, input_format=input_format, split="Val")
        test_preds, test_loss, test_acc, test_ins_acc, test_f1 = train_or_eval_model(model, test_loader, input_format=input_format, split="Test")
        
        with open(path + "-epoch-" + str(e+1) + ".txt", "w") as f:
            f.write("\n".join(list(test_preds)))
        
        x = "Epoch {}: Loss: Train {}; Val {}; Test {}".format(e+1, train_loss, val_loss, test_loss)
        y1 = "Classification Acc: Train {}; Val {}; Test {}".format(train_acc, val_acc, test_acc)
        y2 = "Classification Macro F1: Train {}; Val {}; Test {}".format(train_f1, val_f1, test_f1)
        z = "Instance Acc: Train {}; Val {}; Test {}".format(train_ins_acc, val_ins_acc, test_ins_acc)
            
        print (x)
        print (y1)
        print (y2)
        print (z)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        lf.close()

        f = open(fname, "a")
        f.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        f.close()
        
    lf = open(lf_name, "a")
    lf.write("-"*100 + "\n")
    lf.close()