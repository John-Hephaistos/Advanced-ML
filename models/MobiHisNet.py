import os
from PIL import Image
import csv
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib
import time
import torch.utils.tensorboard as tb

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

    def forward(self, x):
        x = self.relu(
            self.batchnorm_depthwise(self.depthwise(x))
        )  # not sure when batchnorm should be applied!
        x = self.batchnorm_pointhwise(
            self.pointwise(x)
        )  # why different pointiwse/depthwise order?
        return x


class MobiHisNet(torch.nn.Module):  # by Kumar et al. 2021
    # custom architecture, input image = (1, 128, 128)
    def __init__(self, num_classes=4):
        super(MobiHisNet, self).__init__()
        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )
        self.feature_extractor = torch.nn.Sequential(
            DepthWiseSeperableConv(32, 64, stride=1),
            DepthWiseSeperableConv(64, 128, stride=2),
            DepthWiseSeperableConv(128, 256, stride=2),
            DepthWiseSeperableConv(256, 512, stride=2),
            *[
                DepthWiseSeperableConv(256 if i == 0 else 512, 512, stride=1)
                for i in range(5)
            ],
            DepthWiseSeperableConv(512, 1024, stride=2),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
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
        csv_file,
        split="train",
        labels=None,
        target_size=(256, 256),
        transform=None,
        normalize="0-1",
    ):

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


def train_single_Epoch(
    model, tensorboardwriter, dataloader, loss_fn, optimizer, device
):
    # inspired by https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    loss_running = 0.0
    last_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        loss_running += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            last_loss = loss_running / 10  # loss per batch
            print(f" batch {i + 1} loss: {last_loss:.3f}")
            tensorboardwriter.add_scalar("Training Loss", last_loss, i)
            loss_running = 0
    return last_loss


def main():

    dataset_dir = r"C:\Users\Gebruiker\.cache\kagglehub\datasets\paultimothymooney\kermany2018\versions\2\OCT2017"

    build_index(root_dir=dataset_dir, output_csv="dataset_index.csv")

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32

    # Train set with augmentation
    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    )

    train_dataset = OCTDataset(
        "dataset_index.csv", split="train", target_size=(128, 128), normalize="0-1"
    )

    test_dataset = OCTDataset(
        "dataset_index.csv", split="test", target_size=(128, 128), normalize="0-1"
    )
    val_dataset = OCTDataset("dataset_index.csv", split="val", target_size=(128, 128))

    train_Loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_Loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_Loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MobiHisNet(num_classes=4)
    model = model.to(device)
    # calculate tensor for dataset weights
    value_counts = train_dataset.data["label"].value_counts().sort_index()
    label_weights = torch.tensor(
        [
            len(train_dataset) / (len(value_counts) * value_counts.get(lbl, 1))
            for lbl in train_dataset.labels
        ],
        dtype=torch.float,
        device=device,
    )
    # loss - weighted cross entropy
    loss = torch.nn.CrossEntropyLoss(weight=label_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float("inf")
    writer = tb.SummaryWriter()
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        # set model to train mode
        model.train()
        avg_loss = train_single_Epoch(
            model, writer, train_Loader, loss, optimizer, device
        )
        print(f"Avg. Loss: {avg_loss:.3f}")

        # Validation loop
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_Loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = loss(outputs, labels)
                print(f"Validation Loss: {val_loss:.3f}")
                running_val_loss += val_loss.item()
        avg_val_loss = running_val_loss / len(val_Loader)
        print(f"Avg. Validation Loss: {avg_val_loss:.3f}")
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_path = f"best_model_epoch{epoch+1}_{timestamp}.pth"
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
    writer.close()


if __name__ == "__main__":
    main()
