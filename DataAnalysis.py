
import os
from PIL import Image
from os import listdir
import cv2
import pandas as pd
import csv
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from torch.utils.data import Subset
import torchvision.models as models
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # use headless backend
import matplotlib.pyplot as plt



def build_index(root_dir, output_csv, splits=["train", "val", "test"], labels=None):
    labels = labels or ["CNV", "DME", "DRUSEN", "NORMAL"]
    rows = []

    for split in splits:
        for label in labels:
            dir_path = os.path.join(root_dir, split, label)
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                rows.append([os.path.join(dir_path, fname), split, label])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "split", "label"])
        writer.writerows(rows)





class OCTDataset(Dataset):
    def __init__(self, csv_file, split="train", labels=None,
                 target_size=(256,256), transform=None, normalize="0-1"):

        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        self.labels = labels or ["CNV", "DME", "DRUSEN", "NORMAL"]
        self.label2idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.target_size = target_size
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["file_path"]).convert("L")
        img = img.resize(self.target_size, Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        img = np.array(img, dtype=np.float32)

        if self.normalize == "0-1":
            img = img / 255.0
        elif self.normalize == "-1-1":
            img = (img / 127.5) - 1.0
        elif self.normalize == "standard":
            mean, std = img.mean(), img.std() + 1e-8
            img = (img - mean) / std

        img = torch.tensor(img).unsqueeze(0)
        label = self.label2idx[row["label"]]
        return img, label


def main():
    
    dataset_dir = "C:/Users/VLAD/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/OCT2017"

    build_index(root_dir=dataset_dir,  output_csv="dataset_index.csv")

    # Train set with augmentation
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    image_size = (28, 28)
    train_dataset = OCTDataset("dataset_index.csv", split="train",
                               target_size=image_size,
                               normalize="0-1")

    test_dataset = OCTDataset("dataset_index.csv", split="test", target_size=image_size,
                               normalize="0-1")
    val_dataset = OCTDataset("dataset_index.csv", split="val", target_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    grid_search(train_loader, val_loader, test_loader)
    #mlp_model = MLP()
    #mlp_model.train_model(train_loader, val_loader, 100, 0.01, device="cuda")
    #mlp_model.test(test_loader)





    
    
    
 
    # t-SNE no ready

    """
    indices_per_class = {0: [], 1: [], 2: [], 3: []}

    for idx in range(len(train_dataset)):
        #print(test_dataset[idx])
        _, label = train_dataset[idx]
        if len(indices_per_class[label]) < 1000:
            indices_per_class[label].append(idx)

    subset_indices = sum(indices_per_class.values(), [])
    print(subset_indices)
    subset_dataset = Subset(train_dataset, subset_indices)

    for idx in range(len(subset_dataset)):
        print(subset_dataset[idx])

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=False)

    features, labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.repeat(1, 3, 1, 1)
            vecs = resnet(imgs)
            features.append(vecs)
            labels.append(lbls)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(features)

    label_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
    colors = ["red", "blue", "green", "orange"]

    plt.figure(figsize=(10, 8))

    for i, name in enumerate(label_names):
        mask = labels == i
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            label=name,
            color=colors[i],
            alpha=0.7
        )

    plt.legend(title="Class", fontsize=10)
    plt.title("t-SNE Visualization of OCT Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=300)
    plt.close()
    
    
    """
    









    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)



if __name__ == "__main__":
    main()


