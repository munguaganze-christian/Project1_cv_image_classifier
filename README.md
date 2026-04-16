
---

# CNN models for Image Classification using PyTorch & TensorFlow

This project provides a command‑line interface (CLI) for training and evaluating a convolutional neural network model (**CNN1**) using PyTorch.


## Project Overview
A complete deep learning pipeline for natural scene image classification using **PyTorch** and **TensorFlow**. Trained on the Intel Image Classification dataset (~25k images, 6 classes), optimized for GPU training on Kaggle, and deployed as an interactive web application on **Streamlit Cloud**.

## 📁Project Structure

```Markdown
Project1_cv_img_classifier/
├── data/ 
├──├──seg_pred/ 
├──├──seg_test/ 
├──├──seg_train/ 
├── models/                 # 2 Trained weights (.pth, .keras)
├── src/                    # Model architectures & configuration
├── streamlit_app.py        # Main Streamlit interface & inference logic
├── requirements.txt        # Dependencies for Streamlit deployment
├── requirements_train.txt  # Dependencies for local/Kaggle training
└── README.md
```

## 📦 Dependencies

### 🧠 Training (Local / Kaggle)
Used for data loading, augmentation, model training, and evaluation.

```txt
# requirements_train.txt
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pillow>=9.5.0
```

### 🌐  Deployment (Streamlit Cloud)

Lightweight subset required for inference and UI rendering.
```txt
requirements.txt
streamlit
torch
torchvision
tensorflow
numpy
pillow
pandas
```

**Flexible modes:** supports full training or evaluation‑only mode.  
- **GPU support:** automatically uses CUDA when available for faster computation.  
- **Dynamic configuration:** adjust learning rate, weight decay, and number of epochs directly from the CLI.
---

## Usage
Download the Intel image dataset at https://www.kaggle.com/datasets/puneet6060/intel-image-classification.


### Local Training (CPU/GPU)

```bash

python -m venv cv_venv && source venv_cv/bin/activate  # Linux/Mac
pip install -r requirements_train.txt

#Train PyTorch model
python src/train.py --model pytorch --epochs 15

#Train TensorFlow model
python src/train.py --model tensorflow --epochs 15
```

### Streamlit Deployment
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

#### Using the evaluation script

Once your trained model is saved as `your_name_model.pth` and `your_name_model.keras`, in the models folder; 

run:
```bash
python treamlit_app.py
treamlit run
```

Then the interface will open in your default web navigator
---