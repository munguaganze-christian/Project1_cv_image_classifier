# My Project configuration settings

#src/config.py
#central configuration : paths, hyperparameters, augmentation & deployment

from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Trained models for deployment
PYTORCH_MODEL_PATH = MODELS_DIR / "Christian_model.pth"
TENSORFLOW_MODEL_PATH = MODELS_DIR / "Christian_model.keras"

# DATASET
IMG_SIZE = 150
NUM_CLASSES = 6
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Training
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# AUGMENTATION (used only in Training)
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# NORMALISATION (used both in train + inference)
# Statistiques ImageNet (standard pour les CNN)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]