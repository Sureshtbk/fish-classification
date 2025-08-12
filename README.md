# Fish Species Classification Project 🐟

## Overview
This project uses deep learning to classify fish images into multiple species. It compares a custom-built Convolutional Neural Network (CNN) against several pre-trained models (VGG16, ResNet50, MobileNetV2, etc.) to identify the most accurate architecture. The best-performing model is deployed as a user-friendly web application using Streamlit.

## Business Use Cases
- **Enhanced Accuracy:** Systematically determine the best model architecture for fish image classification.
- **Deployment Ready:** Create a user-friendly web application for real-time predictions.
- **Model Comparison:** Evaluate and compare metrics across models to select the most suitable approach.

## Tech Stack
- **Frameworks & Libraries:** TensorFlow, Keras, Streamlit, Scikit-learn, Pandas, NumPy
- **Language:** Python 3.x
- **Platform:** GitHub

## Project Workflow & Execution


1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fish-classification.git](https://github.com/your-username/fish-classification.git)
    cd fish-classification
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    Download the dataset from [link_to_your_dataset] and unzip it into a `data/` directory.

4.  **Train the Models:**
    Run the training script. You can modify the script to select different models.
    ```bash
    python train.py
    ```

5.  **Evaluate Models:**
    Generate the comparison report and confusion matrices.
    ```bash
    python evaluate.py
    ```

6.  **Run the Streamlit Application:**
    Launch the web app to perform real-time predictions.
    ```bash
    streamlit run app.py
    ```

## Model Performance
The models were evaluated based on Accuracy, Precision, Recall, and F1-Score. MobileNetV2 was selected as the best model for deployment.

| Model           | Accuracy | Precision | Recall | F1-Score |
| --------------- | -------- | --------- | ------ | -------- |
| CNN from Scratch| 0.75     | 0.74      | 0.75   | 0.74     |
| **MobileNetV2** | **0.96** | **0.95** | **0.96**| **0.95** |
| ResNet50        | 0.94     | 0.93      | 0.94   | 0.93     |
*(Note: Replace with your actual results from `model_comparison_report.csv`)*

## Demo
A live demo of the Streamlit application in action.

[Link to your LinkedIn video post]