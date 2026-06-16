# CRISP Image Enhancement (MIT FiveK) – PyTorch Implementation

This project is a PyTorch implementation of a CRISP-style image enhancement model trained on the MIT FiveK dataset.  
The goal is to learn an ISP-like transformation that converts unedited images into expert-retouched photos.

Contributors : Sarthak Daga (me ofc) , Mayank kumrawat , Utkarsh Tiwari
This repo contains:

- Training pipeline
- Dataset loader for MIT FiveK
- CRISP model implementation
- Perceptual + color + contrast loss
- Checkpoint system
- Inference script
- Loss logging

---

## 📌 Project Goal

Learn an automatic photo enhancement model:
Unedited image → Neural ISP → Expert-style output

The model is trained to reproduce edits made by professional photographers.

Dataset used:
- MIT FiveK
- Expert C style (currently)

---

## 📂 Project Structure

FreshStart/
│
├── models/
│ └── crisp.py
│
├── utils/
│ ├── fivek_dataset.py
│ └── losses.py
│
├── dataset/
│ └── OrignalDataset/
│ └── Usable/
│
├── train.py
├── inference_fivek.py
├── training_losses.txt
├── crisp_checkpoint.pth
├── best_model.pth
│
└── README.md


---

## ⚙️ Requirements

Python 3.10+

Install dependencies:
pip install torch torchvision tqdm pillow

CUDA recommended.

Check GPU:
python -c "import torch; print(torch.cuda.is_available())"

---

## 📦 Dataset

MIT FiveK dataset required.

Folder structure:
dataset/OrignalDataset/Usable/

UneditedsRGB/
ExpertA/
ExpertB/
ExpertC/
ExpertD/
ExpertE/


Dataset loader:
utils/fivek_dataset.py


---

## 🚀 Training

Run:


python train.py


Features:

- checkpoint saving
- best model saving
- loss logging
- GPU training
- random crop augmentation

Loss used:


L1 loss
Perceptual loss
Color loss
Contrast loss


Loss file saved as:


training_losses.txt


---

## 💾 Checkpoints

Saved files:


crisp_checkpoint.pth
best_model.pth
crisp_fivek_XXXsamples_XXep.pth


Best model is saved automatically.

---

## 🎨 Inference

Run:


python inference_fivek.py


Example:


input → Unedited
output → Expert style


Inference uses:


model.decoder(style)
phi scaling
model.isp()


---

## 📊 Training Behavior

Typical loss curve:


0.25 → 0.10 → 0.08 → oscillation


Model learns:

- exposure
- contrast
- color tone
- shadow depth

---

## 🧠 Notes

Current setup:

- Single expert training
- Limited dataset size
- Residual ISP scaling
- Contrast loss added

Future work:

- multi-expert training
- tone curve loss
- histogram loss
- full ISP pipeline

---

## 🖼 Example

Unedited → Expert → Model Output

(Model approximates expert editing style)

---

## 👨‍💻 Author

Sarthak  
IIIT Vadodara  
Project: CRISP Image Enhancement
