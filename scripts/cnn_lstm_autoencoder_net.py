"""
{'filters': 128, 'kernel_size': 3, 'lstm_units_1': 64, 'lstm_units_2': 32, 'batch_size': 16, 'loss': 'mse'}
"""

import numpy as np
import matplotlib.pyplot as plt

from data_preparation import load_ECG
from keras import layers, Sequential, utils, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# DATA LOAD
xtrain, xtest, ytrain, ytest = load_ECG()

# DATA SCALING
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# PREPARE DATA FOR SEQUENTIAL NET
"""
    LSTM takes (sambles, timesteps, num_features)
"""
timesteps = 20
num_features = xtrain.shape[1] // timesteps
xtrain = xtrain[:, : num_features * timesteps]
xtrain = xtrain.reshape((xtrain.shape[0], timesteps, num_features))
xtest = xtest[:, : num_features * timesteps]
xtest = xtest.reshape((xtest.shape[0], timesteps, num_features))

# NET MAIN
utils.set_random_seed(123)
cnn_lstm_autoencoder = Sequential(
    [
        # Feature extractor
        Input(shape=(timesteps, num_features)),
        layers.Conv1D(filters=64, kernel_size=2, activation="gelu"),
        layers.MaxPool1D(pool_size=2),
        layers.Flatten(),
        layers.Reshape((1, -1)),
        # Encoder
        layers.LSTM(units=128, activation="gelu", return_sequences=True),
        layers.LSTM(units=64, activation="gelu", return_sequences=False),
        # Bottlneck
        layers.RepeatVector(1),
        # Decoder
        layers.LSTM(units=64, activation="gelu", return_sequences=True),
        layers.LSTM(units=128, activation="gelu", return_sequences=True),
        layers.TimeDistributed(layers.Dense(num_features)),
    ]
)
cnn_lstm_autoencoder.compile(
    optimizer="adam",
    loss="mae",
)
cnn_lstm_autoencoder.summary()

# NET TRAINING
history = cnn_lstm_autoencoder.fit(
    xtrain,
    xtrain,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# EVALIUATION ON UNSEEN DATA
reconstructions = cnn_lstm_autoencoder.predict(xtest)
reconstruction_error = np.mean(np.abs(reconstructions - xtest), axis=2).mean(axis=1)

# PERCISION & RECALL & AUPRC FOR UNSEEN DATA
precision, recall, thresholds = precision_recall_curve(ytest, reconstruction_error)
auprc_test = auc(recall, precision)
print(f"AUPRC on the test set: {auprc_test:.3f}")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker=".", label=f"Test Set (AUPRC = {auprc_test:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\evaluation\CNNLSTMAutoencoder.png"
)
plt.show()

# GET ANOMALIES BASED ON BEST THRESHOLD ON RECALL-PERSISION
# best_threshold = np.percentile(reconstruction_error, 95)
best_threshold = np.percentile(reconstruction_error, 40)
ypred = (reconstruction_error > best_threshold).astype(int)

# CONFUSION MATRIX
cm = confusion_matrix(ytest, ypred, normalize="true")
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"],
)
display.plot()
plt.title("Confusion Matrix")
plt.savefig(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\evaluation\CNNLSTMAutoencodercm.png"
)
plt.show()

# SAVE NET
cnn_lstm_autoencoder.save(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\Models\CNNLSTMAutoencoder.keras"
)

utils.plot_model(
    cnn_lstm_autoencoder,
    to_file=r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\architecture\cnn_lstm_autoencoder.png",
    show_shapes=True,
    show_layer_names=True,
)
