import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy
from utils import get_device
from models import NN, ConvNN
from utils import EarlyStopping, plot_losses, get_device
from dataset import get_dataloaders
import os

def train_val(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    val_loss = []
    train_loss = []

    DEVICE = get_device()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(DEVICE)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            if isinstance(model, NN):
                X_batch = X_batch.view(X_batch.size(0), -1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_epoch_loss = np.mean(train_losses)
        train_loss.append(train_epoch_loss)


        model.eval()
        acc_val = []
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                if isinstance(model, NN):
                    X_batch = X_batch.view(X_batch.size(0), -1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                proba = torch.softmax(outputs, dim = 1)
                batch_accuracy = accuracy(proba, y_batch)
                acc_val.append(batch_accuracy.item())

                val_losses.append(loss.item())

        val_epoch_loss = np.mean(val_losses)
        val_loss.append(val_epoch_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)
    return train_loss, val_loss, acc_val

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    model = ConvNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, acc_val = train_val(model, criterion, optimizer, train_loader, test_loader, num_epochs=10)

    acc = np.mean(acc_val)
    print(f"Accuracy: {acc:.3f}")

    plot_losses(train_loss, val_loss)

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/cnn_model.pt")
    print("✅ Model saved to models/cnn_model.pt")


