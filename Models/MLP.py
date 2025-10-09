import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class MLP(nn.Module):
    def __init__(self, learning_rate=0.01, number_of_layers=1, fc1=128, fc2=128, fc3=128):
        super(MLP, self).__init__()

        self.learning_rate = learning_rate
        self._out_1 = fc1
        self._out_2 = fc2
        self._out_3 = fc3
        self._out_4 = fc3
        self._number_of_layers = number_of_layers
        self.output_size = 4
        self._input_layer = nn.Linear(in_features=512*512, out_features=self._out_1)
        if self._number_of_layers == 1:
            self._fc_layer_1 = nn.Linear(in_features=self._out_1, out_features=self._out_2)
            self._output_layer = nn.Linear(in_features=self._out_2, out_features=self.output_size)
        elif self._number_of_layers == 2:
            self._fc_layer_1 = nn.Linear(in_features=self._out_1, out_features=self._out_2)
            self._fc_layer_2 = nn.Linear(in_features=self._out_2, out_features=self._out_3)
            self._output_layer = nn.Linear(in_features=self._out_3, out_features=self.output_size)
        elif self._number_of_layers == 3:
            self._fc_layer_1 = nn.Linear(in_features=self._out_1, out_features=self._out_2)
            self._fc_layer_2 = nn.Linear(in_features=self._out_2, out_features=self._out_3)
            self._fc_layer_3 = nn.Linear(in_features=self._out_3, out_features=self._out_4)
            self._output_layer = nn.Linear(in_features=self._out_4, out_features=self.output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        if self._number_of_layers == 1:
            x = self._input_layer(x)
            x =self._fc_layer_1(x)
            x = self._output_layer(x)
        elif self._number_of_layers == 2:
            x = self._input_layer(x)
            x = self._fc_layer_1(x)
            x = self._fc_layer_2(x)
            x = self._output_layer(x)
        elif self._number_of_layers == 3:
            x = self._input_layer(x)
            x = self._fc_layer_1(x)
            x = self._fc_layer_2(x)
            x = self._fc_layer_3(x)
            x = self._output_layer(x)
        return x

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate, device="cuda"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.train()
        early_stop_timer = 15
        epoch_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            if early_stop_timer == 0:
                print(f"Early stopping triggered after 5 epochs without improvement.")
                break
            for i, (inputs, targets) in enumerate(train_loader):
                outputs = self(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                # print(outputs)
                # print(targets)
                # print(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            val_loss, f1 = self.validate(val_loader, device=device)
            self.train()
            epoch_losses.append(epoch_loss / len(train_loader))


            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, F1: {f1:.4f}')
            # Plotting the loss

            plt.figure()
            plt.plot(epoch_losses, label='Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig("plot_loss.png")
            plt.close()
    def validate(self, val_loader, device='cuda', weights_tensor = None):

        self.to(device)
        self.eval()
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        val_loss = 0.0
        count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                count += 1

                # Predictions
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = val_loss / count
        f1 = f1_score(all_targets, all_preds, average='macro')  # or 'weighted' if class imbalance is large
        return avg_loss, f1



    def test(self, val_loader, device='cuda', weights_tensor = None):

        self.to(device)
        self.eval()
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        val_loss = 0.0
        count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                count += 1

                # Predictions
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = val_loss / count
        f1 = f1_score(all_targets, all_preds, average='macro')  # or 'weighted' if class imbalance is large

        classes = [0, 1, 2, 3]

        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=classes)

        # Normalize per row (actual class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Optional: round values for better visual display
        cm_normalized = np.round(cm_normalized, 2)

        # Plot normalized confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # 2 decimal places

        plt.title('Normalized Confusion Matrix', fontsize=15, pad=20)
        plt.xlabel('Prediction', fontsize=11)
        plt.ylabel('Actual', fontsize=11)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.gca().figure.subplots_adjust(bottom=0.2)
        plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)

        plt.show()

        return avg_loss, f1

def grid_search(train_loader, val_loader, test_loader):
    parameter_grid = {
            'learning_rate': [0.01, 0.05, 0.001],
            'number_of_deep_layers': [1, 2, 3],
            'fc1_size': [128, 256],
            'fc2_size': [256, 624],
            'fc3_size': [624]
        }

    param_combinations = list(itertools.product(*parameter_grid.values()))

    for params in param_combinations:
        learning_rate, number_of_deep_layers, fc1_size, fc2_size, fc3_size = params
        mlp_model = MLP(learning_rate=learning_rate,number_of_layers=number_of_deep_layers, fc1=fc1_size, fc2=fc2_size,
                        fc3=fc3_size)

        mlp_model.train_model(train_loader, val_loader, 100, 0.01, device="cuda")
        mlp_model.test(test_loader)
