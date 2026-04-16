# Training script with CLI argument for model selection"""

import argparse
import sys
import os
import random
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
tf.random.set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Seed fixed {SEED} | Reproductibility")

# Path setup
def setup_python_path():
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

PROJECT_ROOT = setup_python_path()

# Imports projet
from src.config import *
from src.dataset import load_pytorch_data, get_tensorflow_datagen
from src.model_pytorch import create_model_pytorch
from src.model_tensorflow import create_model_tensorflow

def train_pytorch(model, train_loader, test_loader, epochs, device):
    """Train PyTorch model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        scheduler.step(val_loss)
    
    return model

def train_tensorflow(model, train_gen, val_gen, epochs):
    """Train TensorFlow model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['pytorch', 'tensorflow'],
                       help='Model framework to train')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR),
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    print(f"🚀 Training {args.model.upper()} model...")
    
    if args.model == 'pytorch':
        # Load data
        train_loader, test_loader, classes = load_pytorch_data(
            args.data_dir, batch_size=BATCH_SIZE)
        
        # Create & train model
        model = create_model_pytorch(num_classes=NUM_CLASSES)
        trained_model = train_pytorch(
            model, train_loader, test_loader, args.epochs, DEVICE)
        
        # Save model
        torch.save(trained_model.state_dict(), PYTORCH_MODEL_PATH)
        print(f"✅ PyTorch model saved to {PYTORCH_MODEL_PATH}")
        
    else:  # tensorflow
        # Load data
        train_gen, val_gen, test_gen = get_tensorflow_datagen(
            args.data_dir, batch_size=BATCH_SIZE)
        
        # Create & train model
        model = create_model_tensorflow(num_classes=NUM_CLASSES)
        trained_model, history = train_tensorflow(
            model, train_gen, val_gen, args.epochs)
        
        # Save model
        trained_model.save(TENSORFLOW_MODEL_PATH)
        print(f"✅ TensorFlow model saved to {TENSORFLOW_MODEL_PATH}")

if __name__ == '__main__':
    main()