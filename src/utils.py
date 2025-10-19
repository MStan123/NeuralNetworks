import  matplotlib.pyplot as plt
import torch
import os

def plot_losses(train_loss, val_loss, save_path="plots/losses.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved plot to {save_path}")
    plt.close()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def get_device():
    return 'cuda' if torch.cuda.is_available() else "cpu"
