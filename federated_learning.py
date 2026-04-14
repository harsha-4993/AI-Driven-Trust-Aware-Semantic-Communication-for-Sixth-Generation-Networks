import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv("dataset.csv")

# Split into 3 clients (simulate devices)
clients = np.array_split(data, 3)

models = []

for client in clients:
    X = client.drop("TrustRank", axis=1)
    y = client["TrustRank"]

    model = SVC()
    model.fit(X, y)
    models.append(model)

print("Local models trained")

# Simple aggregation (majority voting idea)
def federated_predict(models, sample):
    preds = [m.predict([sample])[0] for m in models]
    return max(set(preds), key=preds.count)

# Example
sample = data.drop("TrustRank", axis=1).iloc[0].values
print("Federated Prediction:", federated_predict(models, sample))