import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_generators

def evaluate_model(model_path, val_data_path):
    """Loads a model and evaluates it on the validation set."""
    model = tf.keras.models.load_model(model_path)
    _, val_gen = get_data_generators(val_data_path, val_data_path) # Use val path for both
    
    # Get true labels and predictions
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    class_names = list(val_gen.class_indices.keys())
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n--- Classification Report for {model_path} ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Confusion Matrix for {model_path}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{model_path}_confusion_matrix.png')
    plt.show()

    return report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']

if __name__ == '__main__':
    VAL_DATA_PATH = 'data/val'
    models_to_evaluate = [
        'CNN_from_Scratch_fish_classifier.h5',
        'MobileNetV2_fish_classifier.h5',
        'ResNet50_fish_classifier.h5',
        # Add paths to your other trained models
    ]
    
    results = []
    for model_file in models_to_evaluate:
        try:
            acc, pre, rec, f1 = evaluate_model(f'models/{model_file}', VAL_DATA_PATH)
            results.append({
                'Model': model_file.split('_')[0],
                'Accuracy': acc,
                'Precision': pre,
                'Recall': rec,
                'F1-Score': f1
            })
        except Exception as e:
            print(f"Could not evaluate {model_file}. Error: {e}")

    # Create and save comparison report
    report_df = pd.DataFrame(results)
    print("\n--- Model Comparison Report ---")
    print(report_df)
    report_df.to_csv('model_comparison_report.csv', index=False)