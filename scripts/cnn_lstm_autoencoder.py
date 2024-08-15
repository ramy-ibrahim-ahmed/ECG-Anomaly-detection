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

# CNN FEATURE EXTRACTOR
cnn_extractor = Sequential(
    [
        Input(shape=(timesteps, num_features)),
        layers.Conv1D(
            filters=64,
            kernel_size=2,
            activation="relu",
        ),
        layers.MaxPool1D(pool_size=2),
        layers.Flatten(),
    ]
)
cnn_extractor.compile(
    optimizer="adam",
    loss="mae"
)

# EMBEDDING WITH CNN
cnn_features_train = cnn_extractor.predict(xtrain)
cnn_features_test = cnn_extractor.predict(xtest)

# (samples, 1, num_features) 
cnn_features_train = cnn_features_train.reshape((cnn_features_train.shape[0], 1, -1))
cnn_features_test = cnn_features_test.reshape((cnn_features_test.shape[0], 1, -1))


# NET LSTM AUTOENCODER
utils.set_random_seed(123)
lstm_autoencoder = Sequential(
    [
        # Encoder
        Input(shape=(1, cnn_features_train.shape[2])),
        layers.LSTM(
            units=128,
            activation="relu",
            return_sequences=True,
        ),
        layers.LSTM(
            units=64,
            activation="relu",
            return_sequences=False,
        ),
        # Bottlneck
        layers.RepeatVector(1),
        # Decoder
        layers.LSTM(
            units=64,
            activation="relu",
            return_sequences=True,
        ),
        layers.LSTM(
            units=128,
            activation="relu",
            return_sequences=True,
        ),
        layers.TimeDistributed(layers.Dense(cnn_features_train.shape[2])),
    ]
)
lstm_autoencoder.compile(
    optimizer="adam",
    loss="mae",
)
lstm_autoencoder.summary()

# NET TRAINING
history = lstm_autoencoder.fit(
    cnn_features_train, 
    cnn_features_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# EVALIUATION ON UNSEEN DATA
reconstructions = lstm_autoencoder.predict(cnn_features_test)
reconstruction_error = np.mean(np.abs(reconstructions - cnn_features_test), axis=2).mean(axis=1)

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
plt.show()

# GET ANOMALIES BASED ON BEST THRESHOLD ON RECALL-PERSISION
best_threshold = np.percentile(reconstruction_error, 95)
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
plt.show()

# SAVE NET
lstm_autoencoder.save(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\Models\CNNEMBEDDINGLSTMAutoencoder.keras"
)
