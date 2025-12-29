# backend/fgsm.py
import torch

class FGSMAttack:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def generate(self, image, label):
        image.requires_grad = True
        output = self.model(image)
        loss = torch.nn.functional.nll_loss(output, label)
        self.model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        perturbed_image = image + self.epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image
