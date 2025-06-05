import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def create_model(num_classes=2, trainable_layers=5, pretrained_model_path=None):
    """
    Create a fine-tunable VGG16 model.
    
    Args:
        num_classes (int): Number of output classes
        trainable_layers (int): Number of last layers to make trainable
        pretrained_model_path (str): Path to pretrained model weights. If None, uses ImageNet weights
    """
    if pretrained_model_path:
        # Load the pretrained model
        # loaded_model = load_model(pretrained_model_path)
        model = load_model(pretrained_model_path) # for testing; to be removed

        # print("\nLoaded model architecture:")
        # loaded_model.summary()
        
        # # Extract the VGG16 part
        # vgg16_model = loaded_model.get_layer('vgg16')
        
        # # Create new model
        # input_layer = tf.keras.Input(shape=(224, 224, 3))
        
        # # Rebuild VGG16 layer by layer with proper trainable settings
        # x = input_layer
        # for layer in vgg16_model.layers:
        #     if isinstance(layer, tf.keras.layers.Conv2D):
        #         x = tf.keras.layers.Conv2D(
        #             filters=layer.filters,
        #             kernel_size=layer.kernel_size,
        #             strides=layer.strides,
        #             padding=layer.padding,
        #             activation=layer.activation,
        #             weights=layer.get_weights(),
        #             trainable=False  # Initially freeze all conv layers
        #         )(x)
        #     elif isinstance(layer, tf.keras.layers.MaxPooling2D):
        #         x = tf.keras.layers.MaxPooling2D(
        #             pool_size=layer.pool_size,
        #             strides=layer.strides,
        #             padding=layer.padding
        #         )(x)
        
        # # Unfreeze the last n convolutional layers
        # conv_layers = [layer for layer in vgg16_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        # for layer in conv_layers[-trainable_layers:]:
        #     layer.trainable = True
        
        # # Add custom top layers
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(50, activation='relu')(x)
        # x = tf.keras.layers.Dense(20, activation='relu')(x)
        # outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        # model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        
        # # Load weights from the pretrained model for the custom top layers
        # model.get_layer('dense').set_weights(loaded_model.get_layer('dense').get_weights())
        # model.get_layer('dense_1').set_weights(loaded_model.get_layer('dense_1').get_weights())
        # model.get_layer('dense_2').set_weights(loaded_model.get_layer('dense_2').get_weights())
        
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
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        x = tf.keras.layers.Dense(20, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # Print model summary showing trainable status of each layer
    print("\nFinal model architecture:")
    model.summary()
    print("\nTrainable layers:")
    for layer in model.layers:
        if layer.trainable:
            print(f"- {layer.name}")
    
    return model

def setup_data_generators(train_dir, val_dir, test_dir=None, batch_size=32):
    """
    Set up data generators for training, validation, and testing.
    """
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal"),
    ])

    # Setup datasets
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        label_mode='categorical'
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        label_mode='categorical'
    )
    
    test_ds = None
    if test_dir:
        test_ds = image_dataset_from_directory(
            test_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            label_mode='categorical'
        )

    # Configure datasets for performance
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    ).prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if test_ds:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Apply preprocessing
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    if test_ds:
        test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    return train_ds, val_ds, test_ds

def evaluate_model(model, test_dataset, output_dir):
    """
    Evaluate the model on test set and generate performance metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: Test data as tf.data.Dataset
        output_dir: Directory to save evaluation results
    """
    # Get predictions and true labels
    y_pred_list = []
    y_true_list = []
    
    # Iterate through the dataset to get predictions and true labels
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred_list.extend(np.argmax(predictions, axis=1))
        y_true_list.extend(np.argmax(labels, axis=1))
    
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    
    # Get class names from the dataset
    class_names = ['noise', 'whistle']  # Replace with your actual class names
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Calculate and return test accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_accuracy

def train_model(model, train_generator, validation_generator, 
                epochs=50, initial_learning_rate=1e-4, 
                output_dir='models', model_name='model_finetuned_vgg.h5'):
    """
    Train the model with the specified parameters.
    
    Args:
        model: The model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of epochs to train
        initial_learning_rate: Initial learning rate
        output_dir: Directory to save models and plots
        model_name: Name of the fine-tuned model file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup callbacks with new model name
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, model_name),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VGG16 model on spectrogram images')
    parser.add_argument('--train_dir', required=True, help='Directory containing training data')
    parser.add_argument('--val_dir', required=True, help='Directory containing validation data')
    parser.add_argument('--test_dir', required=True, help='Directory containing test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--trainable_layers', type=int, default=5, 
                        help='Number of last layers to make trainable')
    parser.add_argument('--output_dir', default='models', 
                        help='Directory to save the trained model')
    parser.add_argument('--pretrained_model', default=None,
                        help='Path to pretrained model to use instead of ImageNet weights')
    parser.add_argument('--model_name', default='model_finetuned_vgg.h5',
                        help='Name for the fine-tuned model file')
    
    args = parser.parse_args()
    
    # Create and setup model
    model = create_model(
        trainable_layers=args.trainable_layers,
        pretrained_model_path=args.pretrained_model
    )
    
    # Setup data generators
    train_ds, val_ds, test_ds = setup_data_generators(
        args.train_dir,
        args.val_dir,
        args.test_dir,
        args.batch_size
    )
    
    # Train model
    # history = train_model(
    #     model,
    #     train_ds,
    #     val_ds,
    #     epochs=args.epochs,
    #     initial_learning_rate=args.learning_rate,
    #     output_dir=args.output_dir,
    #     model_name=args.model_name
    # )
    
    # Load best model and evaluate on test set
    # best_model = load_model(os.path.join(args.output_dir, args.model_name))
    best_model = model # for testing; to be removed
    test_accuracy = evaluate_model(best_model, test_ds, args.output_dir)
    
    # Print final results
    print("\nTraining completed!")
    # print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"\nEvaluation results saved in {args.output_dir}")

if __name__ == "__main__":
    main() 