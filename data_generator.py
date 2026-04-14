import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

# ================= NETWORK PHYSICS =================

distance = np.random.randint(10, 500, n)

L0 = 32.4
alpha = 3

# Path loss (paper Eq. style)
path_loss = L0 + 10 * alpha * np.log10(distance)

# SNR derived from path loss
snr = 30 - (path_loss / 10) + np.random.normal(0, 1, n)

data = pd.DataFrame({
    'Distance': distance,
    'ServingStation': np.random.randint(1, 5, n),
    'ApplicationType': np.random.randint(1, 4, n),
    'Mobility': np.random.randint(0, 10, n),
    'PathLoss': path_loss,
    'SNR': snr,
    'Layer': np.random.randint(1, 5, n),
})

# ================= TRUST LOGIC =================

def assign_trust(row):
    score = 0

    # SNR impact (reduced dominance)
    if row['SNR'] > 20:
        score += 2
    elif row['SNR'] > 10:
        score += 1

    # Distance (reduced weight)
    if row['Distance'] < 200:
        score += 1

    # Mobility (NEW strong effect)
    if row['Mobility'] < 3:
        score += 2
    elif row['Mobility'] < 6:
        score += 1

    # Application Type (NEW strong effect)
    if row['ApplicationType'] == 1:  # high priority
        score += 2
    elif row['ApplicationType'] == 2:
        score += 1

    # Serving Station (NEW)
    if row['ServingStation'] <= 2:
        score += 1

    # Layer
    if row['Layer'] == 1:
        score += 1

    # Final decision
    if score >= 6:
        return 3
    elif score >= 3:
        return 2
    else:
        return 1
data['TrustRank'] = data.apply(assign_trust, axis=1)

data.to_csv("dataset.csv", index=False)
print("✅ Physics-based dataset created!")