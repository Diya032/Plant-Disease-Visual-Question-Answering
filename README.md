
# 🌿 Plant Disease Visual Question Answering (VQA) Demo

This project is a multimodal **Visual Question Answering (VQA)** system designed for **plant disease diagnosis**. It uses the **BLIP encoder** to process image-question pairs and **Flan-T5** to generate answers. The system is deployable via a **FastAPI backend** and a **Streamlit frontend**.

---

## Features

- Vision-language understanding using BLIP + Flan-T5.
- Interactive web UI built with Streamlit.
- FastAPI backend for inference and image/question handling.
  

---

## Demo

https://drive.google.com/file/d/1Aqs0tBaOgg3-HFdvp6q5alWF1NVU4HlV/view?usp=sharing 


https://github.com/user-attachments/assets/5d19036b-8ff2-40aa-aca0-36cd61fb9903


---

## Instructions

1. Clone project using Git
2. Ensure your trained BLIP + Flan-T5 checkpoint (example - best_checkpoint_epoch_0.pth) is placed in the checkpoints/ directory.
3. These checkpoints were downloaded after training in Google Colab Notebook (imported into Kaggle as training will require powerful GPUs)
4. For the purpose of this demo, you will require Python 3.10 version (Stable Python Version i.e Compatible) (due to sentencepiece dependency download requirement)

5. Run in command prompt
   
```bash
  setup.bash
```
Takes care of:
  - Environment setup
  - Dependencies installment
  - Running FastAPI backend
  - Running Streamlit Frontend

---

## Sample Usage

1. Upload an image of a plant or leaf.

2. Type a question (e.g., "What disease is visible?" or "How to treat this?").

3. Click Submit to receive an answer from the model.

---

## Authors / Contributers 

- [@DiyaKhajuria](https://www.github.com/Diya032)

