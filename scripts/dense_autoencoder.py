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

# NET AUTOENCODER
utils.set_random_seed(123)
autoencoder = Sequential(
    [
        # Encoder
        Input(shape=(xtrain.shape[1],)),
        layers.Dense(units=64, activation="relu"),
        # Bottlnick
        layers.Dense(units=32, activation="relu"),
        # Decoder
        layers.Dense(units=64, activation="relu"),
        layers.Dense(units=xtrain.shape[1], activation="relu"),
    ]
)
autoencoder.compile(
    optimizer="adam",
    loss="mse",
)
autoencoder.summary()

# NET TRAINING
autoencoder.fit(
    xtrain,
    xtrain,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# NET EVALUATION
reconstructions = autoencoder.predict(xtest)
reconstruction_error = np.mean(np.abs(reconstructions - xtest), axis=1)

# PERCISION & RECALL & AUPRC FOR UNSEEN DATA
percision, recall, thresholds = precision_recall_curve(ytest, reconstruction_error)
auprc_test = auc(recall, percision)
print(f"AUPRC on the test set: {auprc_test:.3f}")

# PERCISION-RECALL CURVE
plt.figure(figsize=(8, 6))
plt.plot(
    recall,
    percision,
    marker=".",
    label=f"Test set AUPRC = {auprc_test:.3f}",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Percision-Recall Curve")
plt.legend()
plt.grid(True, lw=0.5, linestyle="--")
plt.tight_layout()
plt.show()

# GET ANOMALIES BASED ON BEST THRESHOLD ON RECALL-PERSISION
# best_threshold = thresholds[np.argmax(precision * recall)]
"""
    sets the threshold at the 40th percentile of the reconstruction error. 
    Data points with error greater than this threshold are considered anomalies.
"""
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
autoencoder.save(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\Models\DenseAutoencoder.keras"
)
