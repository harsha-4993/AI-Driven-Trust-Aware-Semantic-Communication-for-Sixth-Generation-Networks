import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ================= LOAD =================

model = joblib.load("best_model.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# ================= PREDICT =================

y_pred = model.predict(X_test)

print("\n📊 Classification Report")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ================= ROC =================

y_test_bin = label_binarize(y_test, classes=[1,2,3])
y_score = model.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()