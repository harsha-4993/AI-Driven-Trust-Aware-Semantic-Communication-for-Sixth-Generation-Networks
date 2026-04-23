# AI-Driven Trust-Aware Semantic Communication for 6G Networks

This project implements a web-based integration of an Artificial Intelligence gateway that fuses Machine Learning-based Quality of Service (QoS) prediction and state-of-the-art Semantic Image Transmission to optimize network resources in highly dynamic, bandwidth-constrained 6G network environments.

## Features

- **Live Context-Aware Dashboard:** Simulated 6G scenarios (distance, mobility, and variable application layers) processed through an SVM classification engine predicting user and environment network trust.
- **Trust Smoothing:** An Exponential Moving Average (EMA) prevents noisy fluctuations, stabilizing resource allocation logic in the application. 
- **Semantic Encoder:** Connects to powerful pre-trained vision models (`openai/clip-vit-base-patch32`) natively to translate images to latent semantic representations when channel degradations are severe.
- **Adaptive Transmission Simulator:** Dynamically assigns transmission architectures. Compare standard JPEG degradations visually, byte-for-byte, against compressed semantic payload recovery on customized low-SNR channel streams.
- **Deep-Dark Glassmorphic Aesthetics:** An intensely styled neon and deep dark-mode (`#060714`) interface powering dual throughput speedometers, linear-gradient history charts, and unified backend data visualizations.
- **Model Intelligence Evaluation:** Run API-level assessment of the internal SVM algorithms showing explicit Classification Reports directly via the frontend.
- **Federated Learning Scaling:** Integrated `federated_learning.py` simulating decentralized model aggregations across dynamic base station nodes, improving the global trust evaluation parameters offline.
- **True Offline Independence:** All analytical frontend engines (Chart.js and Plotly.js) scale completely locally. After the first model synchronization, the system functions 100% autonomously without external internet reliance.
## Requirements

1. **Python 3.8+**
2. See [requirements.txt](requirements.txt) for backend packaging.

## Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd unified_6g_app

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

*(Note for the Semantic Encoder)*: 
The semantic encoder attempts to pull standard transformers from HuggingFace (`openai/clip-vit-base-patch32`). If PyTorch and Transformers run into issues linking with GPU memory arrays locally, the server seamlessly falls back to a simulated latent array logic.

## Usage

Start the unified server logic:
```bash
python app.py
```
Open your browser and navigate to the link rendered in your terminal: `http://localhost:5000`

## Structure

- **`app.py`:** Core unified Flask application and gateway controller linking UI and Models.
- **`templates/index.html`:** The Glassmorphic, Javascript-heavy custom Frontend bridging Semantic and Classic charts. 
- **`encoder.py`:** PyTorch framework applying CLIP to process semantic feature extraction natively.
- **`channel.py`:** Simulates Additive White Gaussian Noise (AWGN) and calculates fading based on variable distance logic.
- **`train_models.py`:** Pipeline algorithm used to train initial network logs with Scikit-learn resulting in a scalable Support Vector Machine.
- **`results/`:** Offline model performance tests.
