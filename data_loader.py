import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import get_data_generators, IMG_HEIGHT, IMG_WIDTH

# (Add build_scratch_cnn and build_transfer_model functions from above)

def plot_history(history, model_name):
    """Plots training & validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.savefig(f'{model_name}_history.png')
    plt.show()

if __name__ == '__main__':
    TRAIN_DATA_PATH = 'data/train'
    VAL_DATA_PATH = 'data/val'
    EPOCHS = 20
    print("Class indices:", train_gen.class_indices)
    
    train_gen, val_gen = get_data_generators(TRAIN_DATA_PATH, VAL_DATA_PATH)
    num_classes = len(train_gen.class_indices)
    
    # --- Model Selection ---
    # Example: Training MobileNetV2
    base_mobilenet = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                       include_top=False,
                                                       weights='imagenet')
    
    model = build_transfer_model(base_mobilenet, num_classes)
    model_name = "MobileNetV2"
    
    print(f"--- Training {model_name} ---")
    model.summary()

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    # Save the trained model
    model.save(f'{model_name}_fish_classifier.h5')
    print(f"Model saved to {model_name}_fish_classifier.h5")

    # Plot and save training history
    plot_history(history, model_name)

    # Repeat the process for other models (VGG16, ResNet50, etc.)