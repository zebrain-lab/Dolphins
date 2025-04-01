"""
Model Module

This module handles loading and creating neural network models for whistle detection.
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten

def load_detection_model(model_path):
    """
    Load a pre-trained whistle detection model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file
        
    Returns
    -------
    tensorflow.keras.Model
        Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

def create_model(num_classes=2, trainable_layers=5, pretrained_model_path=None):
    """
    Create a fine-tunable VGG16 model for whistle detection.
    
    Parameters
    ----------
    num_classes : int, optional
        Number of output classes
    trainable_layers : int, optional
        Number of last layers to make trainable
    pretrained_model_path : str, optional
        Path to pretrained model weights. If None, uses ImageNet weights
        
    Returns
    -------
    tensorflow.keras.Model
        Model for whistle detection
    """
    if pretrained_model_path:
        # Load the pretrained model
        loaded_model = load_model(pretrained_model_path)
        
        # Extract the VGG16 part
        vgg16_model = loaded_model.get_layer('vgg16')
        
        # Create new model
        input_layer = tf.keras.Input(shape=(224, 224, 3))
        
        # Rebuild VGG16 layer by layer with proper trainable settings
        x = input_layer
        for layer in vgg16_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                x = tf.keras.layers.Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    weights=layer.get_weights(),
                    trainable=False  # Initially freeze all conv layers
                )(x)
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.MaxPooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding
                )(x)
        
        # Unfreeze the last n convolutional layers
        conv_layers = [layer for layer in vgg16_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        for layer in conv_layers[-trainable_layers:]:
            layer.trainable = True
        
        # Add custom top layers
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        
        # Load weights from the pretrained model for the custom top layers
        model.get_layer('dense').set_weights(loaded_model.get_layer('dense').get_weights())
        model.get_layer('dense_1').set_weights(loaded_model.get_layer('dense_1').get_weights())
        model.get_layer('dense_2').set_weights(loaded_model.get_layer('dense_2').get_weights())
        
    else:
        # Create new model with ImageNet weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze all base layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze the last n layers
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        # Add custom top layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    return model 