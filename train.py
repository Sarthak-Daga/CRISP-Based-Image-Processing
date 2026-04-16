import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.losses import total_loss

from utils.fivek_datset import FiveKDataset
from models.crisp import CRISP

best_loss = float("inf")
# -----------------------------
# Config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", device)
batch_size = 4
lr = 1e-4
epochs = 50
checkpoint_interval = 10

# -----------------------------
# Dataset
# -----------------------------
train_dataset = FiveKDataset("dataset/OrignalDataset/Usable")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# -----------------------------
# Model
# -----------------------------
model = CRISP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

checkpoint_path = "crisp_checkpoint.pth"
start_epoch = 0

# -----------------------------
# Load checkpoint if exists
# -----------------------------
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# -----------------------------
# Training Loop
# -----------------------------
print("Starting training...")
loss_log_path = "training_losses.txt"

# clear file if exists
with open(loss_log_path, "w") as f:
    f.write("Epoch,Loss\n")

for epoch in range(start_epoch, epochs):

    model.train()
    epoch_loss = 0

    for i, (low, high, expert_idx) in enumerate(tqdm(train_loader)):

        low = low.to(device)
        high = high.to(device)
        expert_idx = expert_idx.to(device)

        optimizer.zero_grad()

        # Forward pass
        output, s = model(low, high)

        loss = total_loss(output, high)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")
    with open(loss_log_path, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f}\n")

    # -----------------------------
    # Save checkpoint every N epochs
    # -----------------------------
    if (epoch + 1) % checkpoint_interval == 0:
        if avg_loss < best_loss:
            best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Checkpoint saved at epoch {epoch+1}")
        torch.cuda.empty_cache()
    
    

# -----------------------------
# Save final model
# -----------------------------
final_path = f"crisp_fivek_{len(train_dataset)}samples_{epochs}ep.pth"
torch.save(model.state_dict(), final_path)

print("Training finished.")
print(f"Final model saved as: {final_path}")
