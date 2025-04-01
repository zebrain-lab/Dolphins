"""
Training Command Module

This module provides command-line interface for training whistle detection models.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..detection.model import create_model
from ..utils.helpers import ensure_directory_exists

def setup_train_parser(subparsers):
    """
    Set up the command-line parser for the train command.
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add the train parser to
        
    Returns
    -------
    argparse.ArgumentParser
        The train command parser
    """
    parser = subparsers.add_parser('train', help='Train a whistle detection model')
    
    parser.add_argument('--train_dir', required=True, help='Directory containing training data')
    parser.add_argument('--val_dir', required=True, help='Directory containing validation data')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and training results')
    parser.add_argument('--pretrained_model', help='Path to pretrained model for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--trainable_layers', type=int, default=5, help='Number of trainable layers in VGG16')
    
    return parser

def train_command(args):
    """
    Execute the train command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    
    # Check if data directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found at {args.train_dir}")
        return 1
    
    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory not found at {args.val_dir}")
        return 1
    
    # Set up data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x/255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x/255.0
    )
    
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Create model
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes, args.pretrained_model, args.trainable_layers)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'model.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    print(f"Training complete. Model saved to {args.output_dir}")
    return 0 