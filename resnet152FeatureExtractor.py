from resnet152CNNModel import CNNEncoder
import torchvision.transforms as standard_transforms
import torch
from datasets import load_from_disk
import pandas as pd
import numpy as np


def dataloaders(hidden_size, datasetType):

    print("Inside dataloader")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
        ])

    data = load_from_disk("./data/scienceqa/")

    data_wo_images_df = pd.DataFrame()
    for row in data[datasetType]:
        data_wo_images_df = pd.concat([data_wo_images_df, pd.DataFrame([row])], ignore_index=True)

    trainDataJson = {"image": [], "question": [], "choices": [], "answer": [],"hint": []}
    model = CNNEncoder(hidden_size).to(device)
    
    counter = 0
    for i, row in data_wo_images_df.iterrows():
        #print(row[1][0])
        img = row[0]
        question = row[1]
        choices = row[2]
        answer = row[3]
        hint = row[4]
        if img is not None:
            img = img.resize((480, 300))
            img = torch.tensor(np.asarray(input_transform(img))).to(device)
            img = img.unsqueeze(0)
            # print("img.size() = ", img.size())
            imgEmbedding = model(img).detach().numpy()
            # print("imgEmbedding size = ", imgEmbedding.shape)
        else:
            imgEmbedding = torch.zeros(1, 2048).detach().numpy()

        # print("imgEmbedding shape = ", imgEmbedding.shape)

        trainDataJson["image"].append(imgEmbedding)
        trainDataJson["question"].append(question)
        trainDataJson["choices"].append(choices)
        trainDataJson["answer"].append(answer)
        trainDataJson["hint"].append(hint)

        counter += 1
        if(counter % 500 == 0):
            print(counter)

    df = pd.DataFrame(trainDataJson)
    pkl = df.to_pickle(datasetType + "_All152.pkl")



print("train")
dataloaders(768, "train")

print("validation")
dataloaders(768, "validation")

print("test")
dataloaders(768, "test")


# df = pd.read_pickle("train_new.pkl")
# print(df.head())