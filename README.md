# Rare Disease Detection Using Diffusion-Based Synthetic Data Generation and Deep Learning

## 📌 Project Overview
This project presents a modular deep learning framework for rare disease analysis using medical images. The core challenge addressed is the scarcity of labeled medical data for rare diseases, which limits the performance of traditional supervised learning models. To overcome this limitation, the project leverages diffusion-based synthetic image generation using Stable Diffusion fine-tuned with Low-Rank Adaptation (LoRA), followed by disease classification and user-level disease prediction.

The system is designed as a three-module pipeline:
1. Synthetic data generation using diffusion models
2. Disease classification using transfer learning
3. User-level disease presence prediction and risk assessment

The proposed framework is intended as a **decision-support system** and does not replace professional medical diagnosis.

---

## 🧠 Problem Statement
Rare diseases suffer from a severe lack of annotated medical images due to low prevalence, high acquisition costs, and strict privacy regulations. Existing generative approaches such as GANs often fail to produce stable, diverse, and anatomically consistent images. This project addresses these issues by using diffusion models with parameter-efficient fine-tuning to generate high-quality synthetic data and improve downstream disease prediction performance.

---

## 🏗️ System Architecture

### Module 1: Synthetic Dataset Generation
- Fine-tunes Stable Diffusion using LoRA on limited real medical images
- Learns disease-specific visual characteristics
- Generates diverse synthetic medical images
- Saves generated images directly into the unified `data/` directory

### Module 2: Disease Classification
- Uses a **common training pipeline** for all diseases
- Trains classifiers on:
  - Real images only
  - Synthetic images only
  - Real + Synthetic images
- Employs transfer learning for improved accuracy and generalization
- Outputs trained disease-specific models

### Module 3: Disease Prediction & Risk Assessment
- Accepts a single medical image as input
- Uses trained models from Module 2 for inference
- Predicts whether a disease is present or not
- Optionally incorporates symptom information to estimate risk levels
- Acts as a decision-support tool

---

## 📁 Project Folder Structure

```
Final_year_Project/
│
├── diseases_Data/                         # DATA MANAGEMENT
│   ├── raw_data/                          # Original images
│   │   ├── Moyamoya Disease with Intraventricular Hemorrhage/
│   │   ├── Neurofibromatosis Type 1 (NF1)/
│   │   ├── Optic Glioma/
│   │   ├── Tuberous Sclerosis/
│   │   └── normal/
│   │
│   ├── refined_data/                      # Processed/cleaned images
│   │   ├── Moyamoya Disease with Intraventricular Hemorrhage/
│   │   ├── Neurofibromatosis Type 1 (NF1)/
│   │   ├── Optic Glioma/
│   │   └── Tuberous Sclerosis/
│   │
│   └── code_for_refining_data/
│       └── code_for_refine.py
│
├── module_1_lora_SYNTHETIC IMAGE GENERATION/  # SYNTHETIC IMAGE GENERATION
│   ├── code for module/
│   │   ├── LoRa_training.ipynb            # Train LoRA models
│   │   ├── LOAD_LoRA_&_GENERATE_IMAGES.ipynb  # Generate synthetic images
│   │   └── images_to_png.py               # Image format conversion
│   │
│   ├── module1 complete process and workflow.txt
│   │
│   ├── Moyamoya Disease with Intraventricular Hemorrhage/
│   │   └── lora_files/                    # Trained LoRA weights
│   │
│   ├── Neurofibromatosis Type 1 (NF1)/
│   │   └── lora_files/                    # Trained LoRA weights
│   │
│   ├── Optic Glioma/
│   │   └── lora_files/                    # Trained LoRA weights
│   │
│   └── Tuberous Sclerosis/
│       └── lora_files/                    # Trained LoRA weights
│
├── module_2_disease classifier/           # DISEASE CLASSIFICATION
│   ├── Optic Glioma/
│   │   ├── disease_images/                # Disease sample images
│   │   ├── normal_images/                 # Normal sample images
│   │   ├── train/                         # Training dataset
│   │   ├── val/                           # Validation dataset
│   │   └── test/                          # Test dataset
│   │
│   ├── Optic Glioma.ipynb                 # Classification training notebook
│   │
│   └── Module2_Results/
│       ├── accuracy_results.txt           # Performance metrics
│       ├── model_real_Optic Glioma.h5
│       └── model_real_Tuberous_Sclerosis.h5
│
├── module_3_application/                  # INFERENCE + USER INTERFACE
│   ├── backend/
│   │   ├── load_model.py
│   │   ├── predict_image.py
│   │   ├── risk_analysis.py
│   │   └── api.py
│   │
│   └── frontend/
│       ├── src/
│       ├── components/
│       └── pages/
│
├── results/                               # EXPERIMENT OUTPUTS
│   ├── disease_1/
│   │   ├── real_results.txt
│   │   ├── synthetic_results.txt
│   │   └── combined_results.txt
│   │
│   ├── disease_2/
│   └── disease_3/
│
├── docs/                                  # PAPER & DIAGRAMS
│   ├── ieee_paper/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── methodology.tex
│   │   └── results.tex
│   │
│   └── diagrams/
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Technologies Used
- Python
- PyTorch
- Stable Diffusion
- Hugging Face Diffusers
- LoRA (PEFT)
- Transfer Learning (CNN-based classifiers)
- NumPy, OpenCV, PIL
- Frontend technologies (React / HTML / CSS / JavaScript)

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train LoRA for Synthetic Image Generation (Module 1)
```bash
cd module_1_lora/disease_1
python train_lora.py
python generate_images.py
```
Generated images will be saved directly into:
```
data/disease_1/synthetic/
```

### Step 3: Prepare Combined Dataset
Merge real and synthetic images into:
```
data/disease_1/real_plus_synthetic/
```

### Step 4: Train Disease Classifier (Module 2)
```bash
cd module_2_training
python train_classifier.py --config config.yaml
```
Trained models are saved in:
```
module_2_training/saved_models/
```

### Step 5: Run Disease Prediction (Module 3)
```bash
cd module_3_application/backend
python api.py
```
Upload a medical image to receive:
- Disease present / not present
- Confidence score
- Risk level (optional)

---

## 📊 Experimental Evaluation
The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Comparisons are performed across:
- Real images only
- Synthetic images only
- Real + Synthetic images

---

## ⚠️ Ethical Considerations
- The system is intended strictly for **decision support**
- It does **not provide medical diagnosis**
- All patient privacy considerations are respected
- Synthetic data generation helps reduce exposure of sensitive medical data

---

## 📄 Research Paper
This project is structured to support submission to IEEE conferences. The following sections are prepared:
- Abstract
- Literature Review
- Methodology
- Results and Discussion

---

## 📌 License
This project is intended for academic and research purposes only.

