import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from modules.epinet_model import EpiNetModel
from torch.utils.data import DataLoader
from datasets import train_test_split_loader

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute classification accuracy of `model` on data from `loader`.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)               # forward without memory update
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def train_one_task(
    task_id: int,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    model:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    scaler:       GradScaler,
    device:       torch.device,
    epochs:       int
):
    """
    Train and evaluate the model on one task.
    Prints per-epoch training loss and test accuracy.
    """
    print(f"=== Task {task_id} ===")
    for epoch in range(1, epochs + 1):
        # Training pass
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            with autocast():
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset) # type: ignore

        # Evaluation pass
        acc = evaluate(model, test_loader, device)
        print(f"[Task {task_id}] Epoch {epoch}/{epochs}  "
              f"Train Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")
    print()

def train_split_mnist(
    batch_size:      int = 128,
    latent_dim:      int = 128,
    hidden_dim:      int = 256,
    num_classes:     int = 10,
    capacity:        int = 1000,
    decay_rate:      float = 1e-3,
    top_k:           int = 5,
    lambda_coef:     float = 0.5,
    lr:              float = 1e-3,
    epochs_per_task: int = 10,
    num_workers:     int = 4,
    pin_memory:      bool = True,
    test_frac:       float = 0.2
):

    """
    Train EpiNet on Split-MNIST (tasks 0–4, then 5–9).
    Implements joint task and replay loss, with episodic memory updates.
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Obtain train/test loaders for Task1 and Task2
    train1, test1, train2, test2 = train_test_split_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        test_frac=test_frac
    )

    # Instantiate model, optimizer, and mixed‐precision scaler
    input_dim = 28 * 28
    model = EpiNetModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        capacity=capacity,
        decay_rate=decay_rate,
        top_k=top_k,
        lambda_coef=lambda_coef,
        device=device
    ).to(device)

    # Optimizer and mixed precision
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Train & evaluate Task1
    train_one_task(1, train1, test1, model, optimizer, scaler, device, epochs_per_task)

    # Train & evaluate Task2
    train_one_task(2, train2, test2, model, optimizer, scaler, device, epochs_per_task)

    # Final joint evaluation to measure forgetting
    acc1_final = evaluate(model, test1, device)
    acc2_final = evaluate(model, test2, device)
    print(f"=== Final Joint Test Accuracy ===\n"
          f"Task1 (0–4): {acc1_final:.4f}\n"
          f"Task2 (5–9): {acc2_final:.4f}")

    # Save model
    torch.save(model.state_dict(), 'epinet_split_mnist.pth')
    print("Training complete. Model saved to epinet_split_mnist.pth")


if __name__ == '__main__':
    train_split_mnist()
