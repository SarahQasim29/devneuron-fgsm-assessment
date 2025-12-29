# backend/evaluate_fgsm.py
import torch
from torchvision import datasets, transforms
from model import MNISTNet
from fgsm import FGSMAttack

device = "cpu"

# Load model
model = MNISTNet().to(device)
model.eval()

# MNIST test loader (no dataset submission required)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=1,
    shuffle=True
)

epsilons = [0.0, 0.05, 0.1, 0.2]
results = []

for eps in epsilons:
    correct = 0
    total = 0
    attacker = FGSMAttack(model, epsilon=eps)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        init_pred = output.argmax(dim=1)

        # Skip if already wrong
        if init_pred.item() != target.item():
            continue

        adv_data = attacker.generate(data, target)
        adv_output = model(adv_data)
        final_pred = adv_output.argmax(dim=1)

        if final_pred.item() == target.item():
            correct += 1

        total += 1

        if total == 200:  # limit for speed
            break

    acc = correct / total
    results.append((eps, acc))
    print(f"Epsilon: {eps} | Accuracy: {acc:.4f}")

# Save results
with open("outputs/evaluation.txt", "w") as f:
    for eps, acc in results:
        f.write(f"Epsilon: {eps}, Accuracy: {acc:.4f}\n")
