# 🔍 Adaptation follow human attention: Gaze-assisted medical segment anything model

A PyTorch implementation of **GAM**, novel framework adapts SAM to the medical domain and seamlessly integrates adaptation into clinical workflows which improves the SAM’s performance in medical image tasks. This enables SAM to be effectively applied to medical images and fosters the emergence of foundation segmentation models for medical image tasks. This repository includes training and testing scripts for reproducible experiments.

---

## 📌 Highlights

- 🧠 **GAM**: For the first time, we propose a Gaze-Assisted adaptation for the segment anything Model (GAM). This novel framework adapts SAM to the medical domain and seamlessly integrates adaptation into clinical workflows which improves the SAM’s performance in medical image tasks. This enables SAM to be effectively applied to medical images and fosters the emergence of foundation
- ⚙️ **Gaze Alignment(GA)**: For effective feature-level adaptation, we propose gazealignment learning (GA), which efficiently filters out irrelevant features from images and enables precise adaptation to follow human attention.
- 🚀 **Gaze Balance (GB)**: For effective adaptation in output-level, we propose a gaze-balance (GB) learning for addressing oversegmentation and under-segmentation. Guided by human gaze attention, GB constrains the model to minimize error regions in the output.

---

## 📁 Training
python train_lora.py
## 📁 testing
python test_lora.py

