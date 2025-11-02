import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random, os
import torch.nn.functional as F
import itertools
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from DataAnalysis import OCTDataset
from DataAnalysis import build_index
from collections import Counter

class CNNbase(nn.Module):
    def __init__(self, learning_rate=0.01):
        super(CNNbase, self).__init__()
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(1, 32 , kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)

        self.dropout = nn.Dropout(0.25)

        # Compute final output shape after pooling
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        #print(np.shape(x))
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #x = F.relu(self.conv3(x))
        #x = self.pool(x)

        #x = F.relu(self.conv4(x))
        #x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def train_model(self, train_loader, val_loader, num_epochs, device="cuda", weights_tensor=None):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f" Training on: {device}")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        best_f1 = 0
        patience, patience_counter = 10, 0
        epoch_losses = []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                # print(next(self.parameters()).device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_loss, val_f1 = self.validate(val_loader, device=device)
            epoch_losses.append(running_loss / len(train_loader))
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {running_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        plt.figure()
        plt.plot(epoch_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("training_loss.png")
        plt.close()

    def validate(self, val_loader, device="cuda"):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        f1 = f1_score(all_targets, all_preds, average='macro')
        return avg_loss, f1

    def test(self, test_loader, device="cuda"):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average='macro')
        print(f"✅ Test F1-score: {f1:.4f}")

        cm = confusion_matrix(all_targets, all_preds)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm, 2))
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        return f1


def grid_search(train_loader, val_loader, test_loader, weights_tensor):
    parameter_grid = {
        'learning_rate': [0.001],
        'number_of_layers': [1, 2],
        'fc1_size': [128],
        'fc2_size': [256],
        'fc3_size': [512],
    }

    param_combinations = list(itertools.product(*parameter_grid.values()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    for params in param_combinations:
        learning_rate, num_layers, fc1, fc2, fc3 = params
        print(f"\n🧠 Testing config: LR={learning_rate}, Layers={num_layers}, FCs=({fc1}, {fc2}, {fc3})")
        model = CNNbase(learning_rate=learning_rate).to(device)
        model.train_model(train_loader, val_loader, num_epochs=20, device=device, weights_tensor=weights_tensor)
        model.test(test_loader, device=device)


def cross_validation(train_loader, val_loader, test_loader, weights_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1_score_list = []
    for fold in range(5):
        print(f"Starting validation for fold {fold+1}/5")
        model = CNNbase(learning_rate=0.01).to(device)
        model.train_model(train_loader, val_loader, num_epochs=20, device=device, weights_tensor=weights_tensor)
        f1, avg = model.test(test_loader, device=device)
        f1_score_list.append(f1)

    return np.mean(np.array(f1_score_list))

def main():

    device = 'cuda'
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    #dataset_dir = "C:/Users/VLAD/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/OCT2017"

    #build_index(root_dir=dataset_dir, output_csv="dataset_index.csv")

    # Load the CSV
    df = pd.read_csv("dataset_index.csv")

    # Filter for the split you want
    train_df = df[df["split"] == "train"]

    # Count occurrences of each class
    class_counts = train_df["label"].value_counts().to_dict()
    print("Class counts:", class_counts)

    num_samples = len(train_df)
    num_classes = len(train_df["label"].unique())

    class_weights = {cls: num_samples / (num_classes * count)
                     for cls, count in class_counts.items()}

    print("Class weights:", class_weights)

    labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    weights_tensor = torch.tensor([class_weights[label] for label in labels], dtype=torch.float32).to(device)
    weights_tensor = weights_tensor / weights_tensor.sum()
    print(weights_tensor)
    weights_tensor.to(device)

    image_size = (128, 128)
    train_dataset = OCTDataset("dataset_index.csv", split="train",
                               target_size=image_size,
                               normalize="0-1")

    test_dataset = OCTDataset("dataset_index.csv", split="test", target_size=image_size,
                              normalize="0-1")
    val_dataset = OCTDataset("dataset_index.csv", split="val", target_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    grid_search(train_loader, val_loader, test_loader, weights_tensor)
    #cross_validation(train_loader, val_loader, test_loader, weights_tensor)


if __name__ == "__main__":
    main()

