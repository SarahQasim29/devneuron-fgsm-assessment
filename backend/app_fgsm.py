# backend/app_fgsm.py
from fastapi import FastAPI, File, UploadFile, Form
import torch
from fgsm import FGSMAttack
from model import MNISTNet
from utils import preprocess_image, tensor_to_base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Add this **before** defining your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (safe for assessment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = "cpu"
from load_model import load_pretrained_model
model = load_pretrained_model().to(device)

@app.post("/attack")
async def attack(
    file: UploadFile = File(...),
    epsilon: float = Form(0.1)
):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes).to(device)

    output = model(image)
    clean_pred = output.argmax(dim=1)

    attacker = FGSMAttack(model, epsilon)
    adv_image = attacker.generate(image, clean_pred)

    adv_output = model(adv_image)
    adv_pred = adv_output.argmax(dim=1)

    success = clean_pred.item() != adv_pred.item()

    return {
        "clean_prediction": clean_pred.item(),
        "adversarial_prediction": adv_pred.item(),
        "attack_success": success,
        "clean_image": tensor_to_base64(image),
        "adversarial_image": tensor_to_base64(adv_image)
    }
