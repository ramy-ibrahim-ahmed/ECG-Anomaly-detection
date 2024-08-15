import numpy as np
import matplotlib.pyplot as plt
import joblib
import graphviz

from sklearn.tree import export_graphviz
from data_preparation import load_ECG
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
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

# MODEL TRAINING
isolation_forest = IsolationForest(
    contamination=0.5,
    random_state=123,
)
isolation_forest.fit(xtrain)

# ANOMALY DETECTION
ptrain = isolation_forest.predict(xtrain)
ptest = isolation_forest.predict(xtest)
ptrain = np.where(ptrain == -1, 1, 0)
ptest = np.where(ptest == -1, 1, 0)

# PERCISION & RECALL & AUPRC FOR UNSEEN DATA
percision, recall, thresholds = precision_recall_curve(ytest, ptest)
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
plt.savefig(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\evaluation\IsolationForest.png"
)
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(ytest, ptest, normalize="true")
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"],
)
display.plot()
plt.title("Confusion Matrix")
plt.savefig(
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\evaluation\IsolationForestcm.png"
)
plt.show()

joblib.dump(
    isolation_forest,
    r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\Models\IsolationForest.joblib",
)


# def plot_tree(tree):
#     """Plot a single decision tree."""
#     dot_data = export_graphviz(tree, filled=True, rounded=True, special_characters=True)
#     graph = graphviz.Source(dot_data)
#     return graph


# def plot_isolation_forest_trees(iso_forest):
#     """Plot all trees in the Isolation Forest"""
#     for i, tree in enumerate(iso_forest.estimators_):
#         graph = plot_tree(tree)
#         graph.render(
#             f"isolation_forest_tree_{i}",
#             format="png",
#             directory=r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\images\tree",
#         )
#         print(f"Plotted Isolation Forest Tree {i}")


# plot_isolation_forest_trees(isolation_forest)
