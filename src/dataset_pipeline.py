# Data loading, preprocessing, and augmentation

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import tensorflow as tf
import os
from src.config import AUGMENTATION_CONFIG

def get_pytorch_transforms(train=True):
    """Get PyTorch transforms for training or inference"""
    if train:
        return transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

def load_pytorch_data(data_dir, batch_size=32):
    """Load data using PyTorch DataLoader"""
    train_transform = get_pytorch_transforms(train=True)
    test_transform = get_pytorch_transforms(train=False)
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'seg_train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'seg_test'), transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader, train_dataset.classes

def get_tensorflow_datagen(data_dir, batch_size=32):
    """Get TensorFlow ImageDataGenerators"""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'seg_train'),
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'seg_train'),
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'seg_test'),
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator