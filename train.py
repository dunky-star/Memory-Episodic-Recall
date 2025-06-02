import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from modules.epinet_model import EpiNetModel
from torch.utils.data import DataLoader
from datasets import train_test_split_loader
import torch.nn.functional as F


# 10. Full Training Loop (Split-MNIST) + Visualization

# . Evaluation Function

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, task_id: int | None = None) -> float:
    """
    Compute classification accuracy of `model` on data from `loader`.
    If task_id is given, temporarily sets
    model.current_task_id so the recall mask uses the right memories.
    """
    model.eval()
    prev_tid = model.current_task_id  # Remember state
    if task_id is not None:
        model.current_task_id = task_id

    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    if task_id is not None:  # restore state
        model.current_task_id = prev_tid
    return correct / total if total else 0.0


# λ-schedule  (linear warm-up over first 3 epochs)
def lambda_schedule(epoch: int, warmup_epochs=3, lambda_coef_max=0.5) -> float:
    return min(lambda_coef_max, lambda_coef_max * epoch / warmup_epochs)


# . Train‑One‑Task Helper
def train_one_task(
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        epochs: int,
        hist: dict
):
    # History keys
    k_loss, k_vloss = f"loss_t{task_id}", f"vloss_t{task_id}"
    k_acc, k_vacc = f"acc_t{task_id}", f"vacc_t{task_id}"
    k_mem, k_rec = f"mem_t{task_id}", f"rec_t{task_id}"
    k_rpl = f"rpl_t{task_id}"

    print(f"=== Task {task_id} ===")
    for ep in range(1, epochs + 1):
        #  Training
        model.current_task_id = task_id
        model.lambda_coef = lambda_schedule(ep)
        model.train()
        running_loss = 0.0
        running_rpl = 0.0
        n_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()

            replay_ratio = 0.1  # 33 % old, 75 % new
            n_rpl = int(replay_ratio * x.size(0))  # how many replay items

            if n_rpl and model.memory.z_buffer.size(0):
                # 1-A. pick indices of memories *not* from the current task
                mask = (model.memory.t_buffer != task_id)  # [N] bool
                pool = torch.nonzero(mask).squeeze(1)  # candidate rows
                # guard against pool == 0 (extremely early epochs)
                if pool.numel():
                    idx = pool[torch.randperm(len(pool))[:n_rpl]]  # random sample

                    # 1-B. get stored latents (z) and labels (y)
                    z_rpl = model.memory.z_buffer[idx]  # [n_rpl, d]
                    y_rpl = model.memory.y_buffer[idx]  # [n_rpl]

                    # 1-C. decode them with ZERO recall; no gradient to memory
                    r_rpl = model.recall_engine.recall(z_rpl, model.memory).detach()
                    logits_rpl = model.decoder(z_rpl, r_rpl)

                    # 1-D. cross-entropy loss on replay logits
                    loss_rpl_direct = F.cross_entropy(logits_rpl, y_rpl)
                else:
                    loss_rpl_direct = torch.tensor(0.0, device=device)
            else:
                loss_rpl_direct = torch.tensor(0.0, device=device)

            with autocast():
                loss = model(x.to(device), y.to(device))  # batch-mean loss
            # 3. add the direct replay-batch loss we just computed
            loss = loss + loss_rpl_direct
            running_rpl += model.last_replay_loss + loss_rpl_direct.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer);
            scaler.update()

            running_loss += loss.item()  # accumulate *batch* loss
            n_batches += 1

        avg_loss = running_loss / n_batches  # mean over all batches
        avg_rpl = running_rpl / n_batches

        # Validation
        model.eval()
        val_sum = val_items = correct = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                logits_v = model(xv)
                val_sum += torch.nn.functional.cross_entropy(
                    logits_v, yv, reduction='sum').item()
                val_items += yv.size(0)
                correct += (torch.argmax(logits_v, 1) == yv).sum().item()
        avg_vloss = val_sum / val_items
        val_acc = correct / val_items

        # Test accuracy
        # test_acc = evaluate(model, test_loader, device)
        # Test accuracy
        test_acc = evaluate(model, test_loader, device, task_id=task_id)

        # Memory / recall
        mem_sz = model.memory.z_buffer.size(0)
        recall_score = 0.0
        if mem_sz:
            with torch.no_grad():
                sal = model.memory.mem_ctrl.decay(
                    model.memory.r0_buffer, model.memory.tau_buffer)
                xs, _ = next(iter(train_loader))
                zs = model.encoder(xs[:32].to(device))
                cos = torch.nn.functional.cosine_similarity(
                    zs.unsqueeze(1),
                    model.memory.c_buffer[: zs.size(0)].unsqueeze(0), dim=-1)
                recall_score = (cos * sal[:cos.size(1)].unsqueeze(0)).mean().item()

        # History
        hist[k_loss].append(avg_loss)
        hist[k_vloss].append(avg_vloss)
        hist[k_rpl].append(avg_rpl)
        hist[k_acc].append(test_acc)
        hist[k_vacc].append(val_acc)
        hist[k_mem].append(mem_sz)
        hist[k_rec].append(recall_score)

        # Log line
        print(f"[Task{task_id}] Epoch{ep:02d}/{epochs}  "
              f"Train_loss {avg_loss:.4f}  (replay {avg_rpl:.3f})  "
              f"Val_loss {avg_vloss:.4f}  Val_acc {val_acc:.4f}  "
              f"Test_acc {test_acc:.4f}  Memory {mem_sz:5d}  Recall_score {recall_score:.3f}")
    print()


# . Hyperparameters

batch_size = 128
latent_dim = 128
hidden_dim = 256
num_classes = 10
capacity = 80000
decay_rate = 1e-3
top_k = 64
lambda_coef = 2.0
lr = 1e-3
epochs_per_task = 10
num_workers = 4
pin_memory = True
val_frac = 0.1
test_frac = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# . Load Data

(train1, val1, test1,
 train2, val2, test2,
 train3, val3, test3) = train_test_split_loader(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    val_frac=val_frac,
    test_frac=test_frac
)

# . Instantiate Model, Optimizer & Scaler

model = EpiNetModel(
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    capacity=capacity,
    decay_rate=decay_rate,
    top_k=top_k,
    lambda_coef=lambda_coef,
    device=device
).to(device)

optimizer = Adam(model.parameters(), lr=lr)
scaler = GradScaler()

# history container/Dict
hist = {k: [] for k in
        ("loss_t1 vloss_t1 rpl_t1 acc_t1 vacc_t1 mem_t1 rec_t1 "
         "loss_t2 vloss_t2 rpl_t2 acc_t2 vacc_t2 mem_t2 rec_t2 "
         "loss_t3 vloss_t3 rpl_t3 acc_t3 vacc_t3 mem_t3 rec_t3").split()}




# def plot_history(tid):
#     loss, vloss = hist[f"loss_t{tid}"], hist[f"vloss_t{tid}"]
#     rpl = hist[f"rpl_t{tid}"]
#     acc, vacc = hist[f"acc_t{tid}"], hist[f"vacc_t{tid}"]
#     mem, rec = hist[f"mem_t{tid}"], hist[f"rec_t{tid}"]
#
#     fig, ax = plt.subplots(1, 4, figsize=(18, 4))
#     ax[0].plot(loss, label="Train");
#     ax[0].plot(vloss, label="Val");
#     ax[0].legend()
#     ax[0].set_title(f"Task {tid} Loss")
#
#     ax[1].plot(rpl, label="Replay");
#     ax[1].legend();
#     ax[1].set_title("Replay loss")
#
#     ax[2].plot(acc, label="Test");
#     ax[2].plot(vacc, label="Val");
#     ax[2].legend()
#     ax[2].set_title(f"Task {tid} Accuracy")
#
#     ax[3].plot(mem, label="Memory");
#     ax[3].plot(rec, label="Recall");
#     ax[3].legend()
#     ax[3].set_title("Memory & Recall score")
#
#     for a in ax: a.set_xlabel("Epoch")
#     plt.tight_layout();
#     plt.show()


# plot_history(1);
# plot_history(2);
# plot_history(3)



def main():
    # ## . Train on Task 1
    train_one_task(1, train1, val1, test1, model, optimizer, scaler, device, epochs_per_task, hist)

    # ## . Train on Task 2
    train_one_task(2, train2, val2, test2, model, optimizer, scaler, device, epochs_per_task, hist)

    # ## . Train on Task 3
    train_one_task(3, train3, val3, test3, model, optimizer, scaler, device, epochs_per_task, hist)

    # ## . Final Joint Evaluation

    # --- final joint test loop ---
    print("\n=== Final Joint Test Accuracy ===")
    for tid, te in zip((1, 2, 3), (test1, test2, test3)):
        acc = evaluate(model, te, device, task_id=tid)
        print(f"Task{tid} accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "epinet_split_mnist.pth")
    print("Model saved → epinet_split_mnist.pth")

if __name__ == '__main__':
    main()
