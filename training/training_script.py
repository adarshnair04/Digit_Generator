import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================
# Hyperparameters
# ===================
BATCH_SIZE = 128
Z_DIM = 100
NUM_CLASSES = 10
HIDDEN_DIM = 256
IMG_DIM = 28 * 28
NUM_EPOCHS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================
# Dataset & Loader
# ===================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================
# Models
# ===================
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, num_classes, hidden_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat([noise, label_embed], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_dim, num_classes, hidden_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(img_dim + num_classes, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat([img, label_embed], dim=1)
        return self.net(x)

# ===================
# Helpers
# ===================
def add_inst_noise(x, epoch, max_epoch=20, sigma=0.05):
    if epoch < max_epoch:
        return x + torch.randn_like(x) * sigma
    return x

def show_generated(generator, n=10):
    generator.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    labels = torch.arange(0, n, device=DEVICE)
    with torch.no_grad():
        samples = generator(z, labels).cpu().view(n, 28, 28)
    generator.train()
    grid = np.concatenate([s.detach().numpy() for s in samples], axis=1)
    plt.imshow(grid, cmap="gray")
    plt.show()

# ===================
# Init
# ===================
gen = Generator(Z_DIM, IMG_DIM, NUM_CLASSES, HIDDEN_DIM).to(DEVICE)
disc = Discriminator(IMG_DIM, NUM_CLASSES, HIDDEN_DIM).to(DEVICE)
criterion = nn.BCELoss()
opt_G = optim.Adam(gen.parameters(), lr=2e-4)
opt_D = optim.Adam(disc.parameters(), lr=2e-4)

# ===================
# Training
# ===================
for epoch in range(NUM_EPOCHS):
    for real_imgs, labels in loader:
        real_imgs, labels = real_imgs.view(-1, 28*28).to(DEVICE), labels.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Labels (with smoothing for real)
        real_targets = torch.full((batch_size, 1), 0.9, device=DEVICE)
        fake_targets = torch.zeros((batch_size, 1), device=DEVICE)

        # ========== Train Discriminator ==========
        noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        gen_imgs = gen(noise, fake_labels)

        real_in = add_inst_noise(real_imgs, epoch)
        fake_in = add_inst_noise(gen_imgs.detach(), epoch)

        real_loss = criterion(disc(real_in, labels), real_targets)
        fake_loss = criterion(disc(fake_in, fake_labels), fake_targets)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ========== Train Generator ==========
        noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        gen_imgs = gen(noise, fake_labels)

        g_loss = criterion(disc(gen_imgs, fake_labels), real_targets)  # want them to be real

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    # LR halve trick after 50 epochs
    if epoch == 50:
        for opt in [opt_G, opt_D]:
            for g in opt.param_groups:
                g['lr'] = 1e-4

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

    if (epoch+1) % 20 == 0:
        show_generated(gen)

# make sure the folder exists
os.makedirs("models", exist_ok=True)

# save only weights# save only weights
torch.save(gen.state_dict(), "models/cgan_generator.pth")
print("✅ Model weights saved successfully.")