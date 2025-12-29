# backend/utils.py
import base64
import io
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    tensor = transform(image).unsqueeze(0)
    return tensor

def tensor_to_base64(tensor):
    image = transforms.ToPILImage()(tensor.squeeze())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
