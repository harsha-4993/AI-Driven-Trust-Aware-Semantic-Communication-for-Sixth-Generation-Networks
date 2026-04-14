import joblib
import numpy as np

model = joblib.load("best_model.pkl")

def update_trust(user_feature_stream):
    history = []

    for features in user_feature_stream:
        pred = model.predict([features])[0]
        history.append(pred)

    # moving average trust
    return round(sum(history) / len(history))


# ===== Example test =====
if __name__ == "__main__":
    dummy_stream = [
        [100,1,2,2,90,20,1],
        [120,1,2,3,92,21,1],
        [150,2,2,3,95,19,2],
    ]

    print("Updated Trust:", update_trust(dummy_stream))