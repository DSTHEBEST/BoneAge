# Bone Age Prediction System

## Overview

This project implements an automated system for predicting bone age from hand X-ray images using deep learning. The system utilizes a ResNet50 model, enhanced with techniques like CLAHE preprocessing and data augmentation, to achieve a Mean Absolute Error (MAE) of approximately 6.2 months. This streamlined workflow offers an efficient solution for automated bone age assessment and diagnostics.

## Key Features

-   **Automated Bone Age Prediction:** Predicts bone age from hand X-ray images using a deep learning model.
-   **ResNet50 Architecture:** Employs transfer learning with a ResNet50 model for feature extraction.
-   **CLAHE Preprocessing:** Enhances image contrast using Contrast-Limited Adaptive Histogram Equalization.
-   **Data Augmentation:** Improves model robustness with image rotations, zooms, and horizontal flips.
-   **Performance Evaluation:** Quantifies model performance with MAE, MSE, and R² metrics.
-   **TensorFlow/Keras Implementation:** Built using industry-standard deep learning libraries.

## System Architecture

1.  **Data Input:**
    -   Hand X-ray images in `.png` format.
    -   CSV file containing image IDs and corresponding bone ages.
2.  **Data Preprocessing:**
    -   Verifies image integrity to filter out corrupted files.
    -   Applies CLAHE to enhance image contrast.
    -   Scales pixel values between 0 and 1.
3.  **Data Augmentation:**
    -   Randomly rotates images within a range of 10 degrees.
    -   Zooms in/out of images by up to 10%.
    -   Flips images horizontally.
4.  **Model Building:**
    -   Loads pre-trained ResNet50 model (excluding top layer).
    -   Adds custom layers on top:
        -   Global Average Pooling.
        -   Dense layers with ReLU activation and Dropout.
        -   Linear output layer for regression.
5.  **Model Training:**
    -   Uses Adam optimizer with a learning rate of 1e-4.
    -   Trains the model for 50 epochs.
    -   Implements early stopping to prevent overfitting.
    -   Reduces the learning rate on plateau for finer adjustments.
    -   Saves the best model based on validation loss.
6.  **Model Evaluation:**
    -   Calculates MAE, MSE, and R² metrics on the validation set.
    -   Generates a scatter plot of prediction errors to visualize performance.
7.  **Output:**
    -   Saves predictions to a CSV file for submission or further analysis.

## Getting Started

### Prerequisites

-   Python 3.6+
-   TensorFlow 2.x
-   Keras
-   NumPy
-   Pandas
-   Pillow (PIL)
-   Matplotlib
-   Scikit-learn

### Installation

1.  Clone the repository:

    ```
    git clone [repository URL]
    cd [repository directory]
    ```

2.  Install the required packages:

    ```
    pip install tensorflow pandas scikit-learn pillow matplotlib
    ```

### Data Preparation

1.  Download the Bone Age dataset.
2.  Organize the data as follows:

    ```
    BoneAge_Dataset/
    ├── BoneAge_train.csv
    └── Training_dataset_BoneAge/
        ├── 1.png
        ├── 2.png
        └── ...
    ```

### Usage

1.  Update the data paths in the script:

    ```
    df = pd.read_csv(r"path/to/BoneAge_train.csv")
    image_folder = r"path/to/Training_dataset_BoneAge"
    ```

2.  Run the script:

    ```
    python bone_age_prediction.py
    ```

### Expected Output

The script will:

-   Train a deep learning model for bone age prediction.
-   Evaluate the model on the validation set.
-   Print MAE, MSE, and R² metrics.
-   Display a scatter plot of prediction errors.
-   Save predictions to `submission.csv`.

### Code Snippets

