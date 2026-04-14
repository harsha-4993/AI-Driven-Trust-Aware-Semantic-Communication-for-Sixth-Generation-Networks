import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# ================= LOAD DATA =================

data = pd.read_csv("dataset.csv")

X = data.drop("TrustRank", axis=1)
y = data["TrustRank"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= BASELINE MODELS =================

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM_basic": SVC()
}

results = {}

print("\n🔹 Baseline Model Accuracy")

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = (acc, model)
    print(f"{name}: {acc:.4f}")

# ================= DEEP LEARNING MODEL =================

print("\n🔹 Deep Learning Model")

dl_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300))
])

dl_model.fit(X_train, y_train)
dl_acc = dl_model.score(X_test, y_test)

results["DeepLearning"] = (dl_acc, dl_model)

print(f"Deep Learning (MLP): {dl_acc:.4f}")

# ================= TUNED SVM =================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

param_grid = {
    'svm__kernel': ['rbf', 'linear', 'poly'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

svm_model = grid.best_estimator_
svm_acc = svm_model.score(X_test, y_test)

results["Tuned_SVM"] = (svm_acc, svm_model)

print("\n🔹 Tuned SVM")
print("Best Params:", grid.best_params_)
print(f"Tuned SVM Accuracy: {svm_acc:.4f}")

# ================= SELECT BEST MODEL =================

best_model_name = max(results, key=lambda x: results[x][0])
best_accuracy, best_model = results[best_model_name]

print("\n🏆 BEST MODEL SELECTED")
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

# ================= SAVE =================

joblib.dump(best_model, "best_model.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")

print("✅ Best model saved successfully!")