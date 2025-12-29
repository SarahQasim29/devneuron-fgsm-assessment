# FGSM Adversarial Attack System  
**Software Engineer – AI Assessment (DevNeuron)**


## Objective

This project demonstrates the implementation and evaluation of **adversarial attacks** on a machine learning model using the **Fast Gradient Sign Method (FGSM)**. The system consists of a **FastAPI backend** that performs the attack and a **Next.js frontend** that allows users to interactively upload images, configure attack strength, and visualize results.

The goal of this assessment is to evaluate:
- Understanding of adversarial machine learning concepts
- Ability to build ML-powered APIs
- Frontend–backend integration
- Deployment and technical communication skills

---

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch
- **Frontend:** Next.js (React, TypeScript)
- **Model:** Pretrained MNIST classifier
- **Deployment:** Render Free Tier (AWS Free Tier not used due to account restrictions)

---

## Part 1 — Backend (FastAPI + ML)

### Overview

The backend exposes a REST API that applies the **Fast Gradient Sign Method (FGSM)** to a pretrained MNIST model. The API accepts an image and an epsilon value and returns predictions on both clean and adversarial inputs.

### Key Files
backend/
├── app_fgsm.py # FastAPI application
├── fgsm.py # FGSM attack implementation
├── evaluate_fgsm.py # Model robustness evaluation script
├── train_model.py # Model training script
└── requirements.txt # Python dependencies

---

### FGSM Implementation

The FGSM attack is implemented in `fgsm.py` as a reusable attack class. It perturbs the input image in the direction of the gradient of the loss with respect to the input.

### Model Evaluation

- A pretrained MNIST model was evaluated against FGSM attacks
- Accuracy drops were observed as epsilon increased
- Only **output results and screenshots** are included (no datasets submitted)

---

### FastAPI Endpoint

#### `POST /attack`

**Input:**
- Image file (PNG/JPEG)
- Epsilon (float, default = 0.1)

**Output (JSON):**
- `clean_prediction`
- `adversarial_prediction`
- `adversarial_image` (Base64 encoded)
- `attack_success` (boolean)

---

### Running Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app_fgsm:app --reload

Backend runs at:
http://localhost:8000

##  Part 2 — Frontend (Next.js)
Overview

A single-page Next.js application that allows users to:
-> Upload an image
-> Adjust epsilon via a slider
-> Trigger FGSM attack
-> View clean vs adversarial predictions
-> Display both images side-by-side

Key Files
frontend/
├── app/page.tsx
├── app/globals.css
└── package.json

Running Frontend Locally
cd frontend
npm install
npm run dev


Frontend runs at:
http://localhost:3000

##  Part 3 — Deployment
Deployment Choice

AWS Free Tier deployment was not possible due to payment method requirements.
Therefore, Render Free Tier was used as allowed by the assessment guidelines.

Deployed URLs

Backend API:
https://devneuron-fgsm-assessment-1.onrender.com

Frontend Application:
https://devneuron-fgsm-assessment-final.onrender.com

If deployment was unavailable, localhost execution screenshots are provided.

Part 4 — FGSM Explanation
The Fast Gradient Sign Method (FGSM) is a white-box adversarial attack that generates adversarial examples by applying a small perturbation to the input in the direction of the gradient of the loss function.
Mathematically, FGSM is defined as:
x_adv = x + ε · sign(∇ₓ J(θ, x, y))

Where:
- x is the original input image  
- y is the true label  
- J(θ, x, y) is the loss function  
- ε (epsilon) controls the perturbation strength  

FGSM is computationally efficient and demonstrates how even small, human‑imperceptible changes to the input can significantly affect a model’s predictions.

Observations
-> The model performs well on clean MNIST images
-> Small epsilon values produce minor perturbations with occasional misclassification
-> Increasing epsilon results in stronger attacks and higher misclassification rates
-> Attack success is directly correlated with epsilon magnitude
