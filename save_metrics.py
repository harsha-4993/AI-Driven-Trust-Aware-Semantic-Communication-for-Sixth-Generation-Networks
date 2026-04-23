import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os

model = joblib.load("best_model.pkl")
X_test, y_test = joblib.load("test_data.pkl")
y_pred = model.predict(X_test)

os.makedirs("results", exist_ok=True)

# Save report
report = classification_report(y_test, y_pred, output_dict=True)
with open("results/classification_report.json", "w") as f:
    json.dump(report, f)

# CM
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig("results/conf_matrix.png")
plt.close()

# ROC
y_test_bin = label_binarize(y_test, classes=[1,2,3])
if hasattr(model, 'predict_proba'):
    y_score = model.predict_proba(X_test)
else:
    # decision function fallback
    y_score = model.decision_function(X_test)

plt.figure(figsize=(6,5))
n_classes = y_test_bin.shape[1]
if len(y_score.shape) == 1:
    y_score = np.vstack([-y_score, y_score]).T # rough fallback
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i] if len(y_score.shape)>1 else y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i+1} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("results/roc_curve.png")
plt.close()
print("Metrics saved successfully.")
