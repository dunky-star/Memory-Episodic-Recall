import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from epinet_model import EpiNetModel


def train_split_mnist(
    latent_dim=128,
    memory_capacity=1000,
    decay_rate=1e-3,
    top_k=5,
    num_classes=10,
    batch_size=64,
    lr=1e-3,
    beta=0.5,
    epochs_per_task=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load MNIST
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split into two tasks: digits 0-4 and 5-9
    tasks = [list(range(0, 5)), list(range(5, 10))]

    # Initialize model and optimizer
    model = EpiNetModel(
        latent_dim=latent_dim,
        memory_capacity=memory_capacity,
        decay_rate=decay_rate,
        top_k=top_k,
        num_classes=num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track losses
    for task_idx, classes in enumerate(tasks):
        # Subset dataset for current task
        idxs = [i for i, lbl in enumerate(full_train.targets) if int(lbl) in classes]
        task_dataset = Subset(full_train, idxs)
        loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)

        print(f"=== Training Task {task_idx} | Classes {classes} ===")
        for epoch in range(epochs_per_task):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Forward pass
                logits, z = model(x_batch)
                loss_main = F.cross_entropy(logits, y_batch)

                # Replay from memory
                if model.memory.memory:
                    sample_k = min(len(model.memory.memory), batch_size)
                    mem_idxs = np.random.choice(len(model.memory.memory), sample_k, replace=False)
                    x_mem = torch.stack([model.memory.memory[i]['x'] for i in mem_idxs]).to(device)
                    y_mem = torch.tensor([model.memory.memory[i]['y'] for i in mem_idxs], dtype=torch.long, device=device)

                    logits_mem, _ = model(x_mem)
                    loss_replay = F.cross_entropy(logits_mem, y_mem)
                    loss = loss_main + beta * loss_replay
                else:
                    loss = loss_main

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store into episodic memory
                model.memorize(x_batch[0], z[0], y_batch[0], salience=loss_main.item())

                epoch_loss += loss.item() * x_batch.size(0)

            avg_loss = epoch_loss / len(task_dataset)
            print(f"Task {task_idx} | Epoch {epoch} | Loss: {avg_loss:.4f}")

    # Save checkpoint
    # Plain literal
    torch.save(model.state_dict(), 'epinet_split_mnist.pth')
    print("Training complete. Model saved to 'epinet_split_mnist.pth'.")


if __name__ == '__main__':
    train_split_mnist()