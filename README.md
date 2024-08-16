# ECG Anomaly Detection Project

## Table of Contents
1. [Introduction](#introduction)
2. [Preprocessing](#preprocessing)
    - [Min-Max Normalization](#min-max-normalization)
    - [Train-Test-Validation Split](#train-test-validation-split)
3. [Models](#models)
    - [Isolation Forest](#isolation-forest)
    - [Autoencoder (Dense Layers)](#autoencoder-dense-layers)
    - [LSTM Autoencoder](#lstm-autoencoder)
    - [LSTM Autoencoder + CNN as Embedding Layer](#lstm-autoencoder-cnn-embedding-layer)
    - [LSTM Autoencoder + CNN Trainable as Feature Extractor](#lstm-autoencoder-cnn-trainable-feature-extractor)
4. [Evaluation](#evaluation)
    - [Confusion Matrix](#confusion-matrix)
    - [Precision-Recall Plot](#precision-recall-plot)
5. [Conclusion](#conclusion)

## Introduction

This project focuses on detecting anomalies in Electrocardiogram (ECG) signals. Several machine learning models were employed, including Isolation Forest, Autoencoders, and LSTM-based models. The goal is to identify and compare the performance of these models in detecting ECG anomalies.

## Preprocessing

### Min-Max Normalization

To ensure the features are on a similar scale, Min-Max normalization was applied. This technique scales the data to a range of [0, 1] by subtracting the minimum value and dividing by the range of the data.

### Train-Test-Validation Split

The dataset was split into three parts: training, testing, and validation. This split is crucial for unbiased model evaluation:
- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune the model's hyperparameters.
- **Test Set**: Used to evaluate the model's performance.

## Models

### Isolation Forest

Isolation Forest is an unsupervised learning algorithm for anomaly detection. In this project, it was used with a contamination level of 0.5. The model isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

- **Parameters**:
  - Contamination: 0.5

#### Model Plot
![Isolation Forest Model Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/tree/isolation_forest_tree.png)

#### Confusion Matrix
![Isolation Forest Confusion Matrix](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/IsolationForestcm.png)

#### Precision-Recall Plot
![Isolation Forest Precision-Recall Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/IsolationForest.png)

### Autoencoder (Dense Layers)

An autoencoder with dense layers was used for anomaly detection. The model learns to compress and then reconstruct the data, and anomalies are detected based on the reconstruction error.

- **Threshold**: 40 (based on reconstruction error)

#### Model Plot
![Autoencoder Model Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/architecture/dense_autoencoder.png)

#### Confusion Matrix
![Autoencoder Confusion Matrix](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/DenseAutoencodercm.png)

#### Precision-Recall Plot
![Autoencoder Precision-Recall Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/DenseAutoencoder.png)

### LSTM Autoencoder

A Long Short-Term Memory (LSTM) autoencoder was implemented to capture temporal dependencies in the ECG signals. It is particularly effective for sequential data.

- **Threshold**: 40 (based on reconstruction error)

#### Model Plot
![LSTM Autoencoder Model Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/architecture/lstm_autoencoder.png)

#### Confusion Matrix
![LSTM Autoencoder Confusion Matrix](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/LSTMAutoencodercm.png)

#### Precision-Recall Plot
![LSTM Autoencoder Precision-Recall Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/LSTMAutoencoder.png)

### LSTM Autoencoder + CNN as Embedding Layer

In this model, a Convolutional Neural Network (CNN) is used as an embedding layer before feeding the data into the LSTM autoencoder. This helps in extracting more meaningful features from the ECG signals.

- **Threshold**: 40 (based on reconstruction error)

#### Model Plot
##### CNN Embedding
![CNN Embedding Model Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/architecture/cnn_embed.png)

##### LSTM Autoencoder
![LSTM Autoencoder](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/architecture/cnn_embed_lstm_autoencoder.png)

#### Confusion Matrix
![LSTM Autoencoder + CNN Embedding Confusion Matrix](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/CNNEmbedINGLSTMAutoencodercm.png)

#### Precision-Recall Plot
![LSTM Autoencoder + CNN Embedding Precision-Recall Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/CNNEmbedINGLSTMAutoencoder.png)

### LSTM Autoencoder + CNN Trainable as Feature Extractor

This model uses a CNN as a trainable feature extractor combined with an LSTM autoencoder. The CNN layers are trained along with the LSTM autoencoder to enhance the feature extraction process.

- **Threshold**: 40 (based on reconstruction error)

#### Model Plot
![LSTM Autoencoder + CNN Trainable Model Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/architecture/cnn_lstm_autoencoder.png)

#### Confusion Matrix
![LSTM Autoencoder + CNN Trainable Confusion Matrix](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/CNNLSTMAutoencodercm.png)

#### Precision-Recall Plot
![LSTM Autoencoder + CNN Trainable Precision-Recall Plot](https://github.com/ramy-ibrahim-ahmed/ECG-Anomaly-detection/blob/main/images/evaluation/CNNLSTMAutoencoder.png)

## Evaluation

### Confusion Matrix

The confusion matrix was used to evaluate the performance of each model. It provides insight into the number of true positives, true negatives, false positives, and false negatives.

### Precision-Recall Plot

The precision-recall plot helps in understanding the trade-off between precision and recall for different threshold values. It is particularly useful for imbalanced datasets.

## Conclusion

In this project, we explored various models for ECG anomaly detection. Each model was evaluated using confusion matrices and precision-recall plots. The results highlight the strengths and weaknesses of each approach, providing insights into which models perform best under different conditions. Further work can involve fine-tuning the models and exploring additional architectures to improve detection accuracy.
