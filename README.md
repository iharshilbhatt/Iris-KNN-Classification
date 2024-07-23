# Iris k-NN Classification

This project performs k-Nearest Neighbors (k-NN) classification on the Iris dataset using the `scikit-learn` library. The goal is to classify Iris flower species based on sepal and petal measurements.

## Project Overview

The Iris dataset is a classic dataset in machine learning that contains measurements of iris flowers. The dataset consists of 150 samples from three species of iris flowers: Iris-setosa, Iris-versicolor, and Iris-virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width.

This project demonstrates how to apply the k-NN classification algorithm to the Iris dataset. The script performs the following steps:

1. **Load and Preprocess Data**: Load the Iris dataset and preprocess it by converting categorical labels to numerical labels.
2. **Data Splitting**: Split the dataset into training and testing sets.
3. **Model Training**: Train a k-NN classifier on the training set.
4. **Prediction and Evaluation**: Evaluate the model's performance on the testing set using accuracy and confusion matrix.

## Files

- `knn_classification.py`: Python script implementing the k-NN classification algorithm.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.
- `README.md`: This file, providing details about the project.

## Requirements

To run this project, you need to install the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (optional, for visualization)

You can install these libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib

```

Script Details
knn_classification.py
This script performs the following steps:

Load Dataset:

Downloads the Iris dataset from the UCI Machine Learning Repository.
Assigns column names to the dataset.
Preprocess Data:

Converts string labels to numerical values.
Splits the dataset into training and testing sets using train_test_split.
Train Model:

Initializes and trains a k-NN classifier with k=3.
Evaluate Model:

Makes predictions on the test set.
Calculates and prints the confusion matrix and accuracy score.
