import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTNet

device = torch.device("cpu")  # use "cuda" if GPU available

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):  # 3 epochs for demo
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved as mnist_cnn.pth")
