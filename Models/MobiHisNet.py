import os
import random
import csv
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import torch.utils.tensorboard as tb
import seaborn as sns
import matplotlib.pyplot as plt

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")


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
    print(f"Indexed {len(rows)} files from {root_dir}")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "split", "label"])
        writer.writerows(rows)


class DepthWiseSeperableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthWiseSeperableConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batchnorm_depthwise = torch.nn.BatchNorm2d(in_channels)
        self.batchnorm_pointhwise = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batchnorm_depthwise(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pointwise(x)
        x = self.batchnorm_pointhwise(x)
        return x


class MobiHisNet(torch.nn.Module):  # by Kumar et al. 2021x
    # custom architecture, input image = (1, 128, 128)
    def __init__(self, num_classes=4):
        super(MobiHisNet, self).__init__()
        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.feature_extractor = torch.nn.Sequential(
            DepthWiseSeperableConv(32, 64, stride=1),
            DepthWiseSeperableConv(64, 128, stride=2),
            DepthWiseSeperableConv(128, 128, stride=1),
            DepthWiseSeperableConv(128, 256, stride=2),
            DepthWiseSeperableConv(256, 256, stride=1),
            DepthWiseSeperableConv(256, 512, stride=2),
            *[DepthWiseSeperableConv(512, 512, stride=1) for _ in range(5)],
            DepthWiseSeperableConv(512, 1024, stride=2),
            DepthWiseSeperableConv(1024, 1024, stride=1),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):

        x = self.first_conv(x)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class OCTDataset(Dataset):
    def __init__(
        self,
        csv_file=None,
        data=None,
        split="train",
        combined=False,
        labels=None,
        target_size=(256, 256),
        transform=None,
        normalize="0-1",
    ):
        if data is not None:
            self.data = data
            if combined:
                self.data = self.data[
                    self.data["split"].isin(["val", "test"])
                ].reset_index(drop=True)
            else:
                self.data = self.data = self.data[
                    self.data["split"] == split
                ].reset_index(drop=True)
        elif csv_file is not None:
            self.data = pd.read_csv(csv_file)
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        else:
            raise ValueError("Either csv_file or data must be provided.")
        self.labels = labels or ["CNV", "DME", "DRUSEN", "NORMAL"]
        self.label2idx = {label: i for i, label in enumerate(self.labels)}
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

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = self.label2idx[row["label"]]
        return img, label


def train_single_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    f1 = f1_score(all_preds, all_labels, average="macro")
    return avg_loss, f1, all_labels, all_preds


def main():
    # Change this path to the location of the OCT dataset on your machine
    # insert the folder where you dataset is downloaded
    # Note, this can change from device to device - it can differ from Windows 10 or 11 or Mac
    dataset_dir = r"../../home1/s4327276/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/OCT2017 "  # there is a space at the end of the path at times
    build_index(root_dir=dataset_dir, output_csv="dataset_index.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 32
    weight_decay = 0.02
    patience = 6

    df = pd.read_csv("dataset_index.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    combined_test_df = df[df["split"].isin(["val", "test"])].reset_index(drop=True)

    labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    X = train_df["file_path"].values
    y = train_df["label"].values

    # Class weights
    class_counts = train_df["label"].value_counts().to_dict()
    print("Class counts:", class_counts)
    num_samples = len(train_df)
    num_classes = len(labels)
    class_weights = {
        cls: num_samples / (num_classes * count) for cls, count in class_counts.items()
    }
    weights_tensor = torch.tensor(
        [class_weights[label] for label in labels], dtype=torch.float32
    ).to(device)
    weights_tensor.to(device)
    print("Class weights:", class_weights)
    image_size = (128, 128)
    print("Training final model on full training data")
    final_model = MobiHisNet(num_classes=len(labels)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_full_df, val_holdout_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=8
    )
    train_dataset = OCTDataset(
        data=train_full_df, labels=labels, target_size=image_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    validation_dataset = OCTDataset(
        data=val_holdout_df, labels=labels, target_size=image_size
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=3
    )

    test_dataset = OCTDataset(
        data=combined_test_df,
        labels=labels,
        target_size=image_size,
        split=None,
        combined=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_single_epoch(
            final_model, train_loader, loss_fn, optimizer, device
        )
        train_losses.append(train_loss)
        val_loss, f1, _, _ = evaluate(final_model, validation_loader, loss_fn, device)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(final_model.state_dict(), "best_final_model.pth")
            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, F1={f1:.2f}(improved)"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    final_model.load_state_dict(torch.load("best_final_model.pth"))
    test_loss, f1, all_labels, all_preds = evaluate(
        final_model, test_loader, loss_fn, device
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels))))
    labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=labels,
        labels=list(range(len(labels))),
        digits=4,
    )

    print(f"\nFinal Test Loss: {test_loss:.4f}, F1: {f1:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_losses, label="Validation Loss", color="blue")
    plt.plot(train_losses, label="Training Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.savefig("loss_curves_final_model.png")
    plt.close()

    # Save the metrics to text file
    with open("metrics_final_model.txt", "w") as f:
        f.write(f"Final test loss: {test_loss:.4f}\n")
        f.write(f"Final test F1: {f1:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n{np.array2string(cm)}\n")
        f.write(f"Classification Report:\n{report}\n")


if __name__ == "__main__":
    main()
