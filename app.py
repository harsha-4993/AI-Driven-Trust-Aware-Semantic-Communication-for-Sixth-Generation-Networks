"""
Unified 6G Semantic Trust Gateway
- SVM Trust Prediction + Behavior Model
- Semantic Image Transmission
- Model Evaluation API (Classification Report, Confusion Matrix, ROC)
- Adaptive Bandwidth Allocation
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import base64
import joblib
import json

from encoder import SemanticEncoder
from channel import ChannelSimulator

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ===== LOAD MODELS & RESOURCES =====
print("Loading Semantic Encoder...")
encoder = SemanticEncoder()
print("Loading Trust SVM Model...")
trust_model = joblib.load("best_model.pkl")
print("Loading Test Data...")
X_test, y_test = joblib.load("test_data.pkl")
print("✅ All models ready!")

# Global state for Behavior Model (EMA Smoothing)
global_state = {
    "smoothed_trust": 2.5,
    "alpha": 0.3
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def apply_jpeg_degradation(image, snr_db):
    img_array = np.array(image).astype(np.float32) / 255.0
    if snr_db <= 0:
        noise_level = 0.8; blur_radius = 3
    elif snr_db < 8:
        noise_level = 0.6 - (snr_db / 20); blur_radius = 2
    elif snr_db < 16:
        noise_level = 0.3 - ((snr_db - 8) / 30); blur_radius = 1
    elif snr_db < 25:
        noise_level = 0.1 - ((snr_db - 16) / 90); blur_radius = 0.5
    else:
        noise_level = 0.02; blur_radius = 0
    noise_level = max(0.02, min(0.8, noise_level))
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    if blur_radius > 0:
        pil_img = Image.fromarray((noisy * 255).astype('uint8'))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        noisy = np.array(pil_img) / 255.0
    if 8 <= snr_db < 20:
        noisy = np.round(noisy * 64) / 64
    return Image.fromarray((np.clip(noisy, 0, 1) * 255).astype('uint8'))

def apply_semantic_reconstruction(image, reconstruction_quality, snr_db, trust_level):
    result = image.copy()
    adaptive_quality = reconstruction_quality * (trust_level / 3.0)
    if adaptive_quality < 0.4:
        result = result.filter(ImageFilter.GaussianBlur(radius=6))
        result = result.point(lambda p: p * 0.8 + 40)
    elif adaptive_quality < 0.6:
        result = result.filter(ImageFilter.GaussianBlur(radius=3))
    elif adaptive_quality < 0.8:
        result = result.filter(ImageFilter.GaussianBlur(radius=1))
        result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
    elif adaptive_quality < 0.95:
        result = result.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=0))
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), f"SNR: {snr_db:.1f}dB | Trust: {trust_level:.1f}", fill=(0, 255, 128))
    return result

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory('results', filename)

# ==================== LIVE NETWORK STATE ====================
@app.route('/api/evaluate', methods=['POST'])
def evaluate_network():
    try:
        data = request.json or {}
        distance = float(data.get('distance', 100))
        ss = int(data.get('ss', 1))
        at = int(data.get('at', 1))
        mobility = float(data.get('mobility', 2))
        layer = int(data.get('layer', 1))

        L0 = 32.4; alpha = 3
        path_loss = L0 + 10 * alpha * np.log10(max(distance, 1))
        snr = 30 - (path_loss / 10)

        X = np.array([[distance, ss, at, mobility, path_loss, snr, layer]])
        raw_trust = int(trust_model.predict(X)[0])

        global_state['smoothed_trust'] = (global_state['alpha'] * raw_trust) + ((1 - global_state['alpha']) * global_state['smoothed_trust'])
        smoothed = global_state['smoothed_trust']

        qos = "High QoS" if smoothed > 2.5 else "Medium QoS" if smoothed > 1.5 else "Low QoS"
        base_speed = 40
        optimized_speed = 100 if smoothed > 2.5 else 60 if smoothed > 1.5 else 20

        return jsonify({
            'success': True, 'path_loss': round(path_loss, 2), 'snr_db': round(snr, 2),
            'raw_trust': raw_trust, 'smoothed_trust': round(smoothed, 3),
            'qos': qos, 'base_speed': base_speed, 'optimized_speed': optimized_speed,
            'distance': distance, 'mobility': mobility
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MODEL EVALUATION ====================
@app.route('/api/model_eval', methods=['GET'])
def model_evaluation():
    """Return classification report, confusion matrix, ROC data"""
    try:
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
        from sklearn.preprocessing import label_binarize

        y_pred = trust_model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred).tolist()

        # ROC Curve (micro-average)
        y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
        y_score = trust_model.predict_proba(X_test)

        roc_data = {}
        for i, cls in enumerate([1, 2, 3]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[f'class_{cls}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': round(float(roc_auc), 4)
            }

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        roc_data['micro_avg'] = {
            'fpr': fpr_micro.tolist(),
            'tpr': tpr_micro.tolist(),
            'auc': round(float(roc_auc_micro), 4)
        }

        # Per-class accuracy for bar chart
        per_class = {}
        for cls_key in ['1', '2', '3']:
            if cls_key in report:
                per_class[cls_key] = {
                    'precision': round(report[cls_key]['precision'], 3),
                    'recall': round(report[cls_key]['recall'], 3),
                    'f1': round(report[cls_key]['f1-score'], 3),
                    'support': int(report[cls_key]['support'])
                }

        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 4),
            'confusion_matrix': cm,
            'roc': roc_data,
            'per_class': per_class
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== SEMANTIC TRANSMISSION ====================
@app.route('/api/transmit', methods=['POST'])
def transmit():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        distance = float(request.form.get('distance', 100))
        ss = int(request.form.get('ss', 1))
        at = int(request.form.get('at', 1))
        mobility = float(request.form.get('mobility', 2))
        layer = int(request.form.get('layer', 1))

        L0 = 32.4; alpha_loss = 3
        path_loss = L0 + 10 * alpha_loss * np.log10(max(distance, 1))
        snr_db = 30 - (path_loss / 10)

        X = np.array([[distance, ss, at, mobility, path_loss, snr_db, layer]])
        raw_trust = trust_model.predict(X)[0]
        global_state['smoothed_trust'] = (global_state['alpha'] * raw_trust) + ((1 - global_state['alpha']) * global_state['smoothed_trust'])
        current_trust = global_state['smoothed_trust']

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        original_img = Image.open(filepath).convert('RGB')
        original_base64 = image_to_base64(original_img)

        # Traditional
        degraded_jpeg = apply_jpeg_degradation(original_img, snr_db)
        jpeg_base64 = image_to_base64(degraded_jpeg)
        trad_psnr = min(38, max(10, 12 + (snr_db / 5) * 8))
        trad_ssim = min(0.95, max(0.2, 0.25 + (snr_db / 5) * 0.25))
        trad_size = max(20, 150 - (snr_db * 2))
        trad_compression = round((1 - trad_size/150) * 100, 1)

        # Semantic
        encoding_result = encoder.encode(filepath)
        original_vector = encoding_result["latent_vector"]
        received_vector = ChannelSimulator.transmit(original_vector, snr_db)
        semantic_similarity = float(np.dot(original_vector, received_vector) /
                                    (np.linalg.norm(original_vector) * np.linalg.norm(received_vector)))
        distortion = float(np.mean((original_vector - received_vector) ** 2))
        reconstruction_quality = max(0, min(1, 1 - distortion * 8))
        sem_psnr = 20 + (reconstruction_quality * 20)
        sem_ssim = 0.55 + (reconstruction_quality * 0.4)

        semantic_img = apply_semantic_reconstruction(original_img, reconstruction_quality, snr_db, current_trust)
        semantic_base64 = image_to_base64(semantic_img)

        semantic_size = 12 * (current_trust / 3.0)
        bandwidth_savings = round((1 - semantic_size/150) * 100, 1)
        overall_trust_score = min(100, max(0, (current_trust / 3.0 * 100)))

        semantic_desc = encoding_result.get("description", {"scene_type": "unknown", "objects": ["unknown"], "confidence": 0.5})

        response = {
            'success': True,
            'images': {
                'original': f"data:image/png;base64,{original_base64}",
                'traditional': f"data:image/png;base64,{jpeg_base64}",
                'semantic': f"data:image/png;base64,{semantic_base64}"
            },
            'traditional': {
                'psnr': round(trad_psnr, 2), 'ssim': round(trad_ssim, 3),
                'size_kb': round(trad_size, 1), 'compression_ratio': trad_compression,
                'quality_desc': "Good" if snr_db > 20 else "Moderate" if snr_db > 10 else "Poor"
            },
            'semantic': {
                'psnr': round(sem_psnr, 2), 'ssim': round(sem_ssim, 3),
                'semantic_similarity': round(semantic_similarity, 4),
                'reconstruction_quality': round(reconstruction_quality, 4),
                'size_kb': round(semantic_size, 1), 'compression_ratio': bandwidth_savings
            },
            'semantic_description': {
                'scene': semantic_desc.get('scene_type', 'unknown'),
                'objects': semantic_desc.get('objects', ['unknown']),
                'confidence': semantic_desc.get('confidence', 0.5)
            },
            'trust_score': round(overall_trust_score, 1),
            'snr_db': round(snr_db, 2),
            'bandwidth_savings': bandwidth_savings,
            'smoothed_trust': round(current_trust, 3)
        }

        os.remove(filepath)
        return jsonify(response)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Unified 6G Trust Gateway at http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)