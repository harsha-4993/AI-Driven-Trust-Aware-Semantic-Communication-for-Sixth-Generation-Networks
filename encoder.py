"""
Step 2: Basic Semantic Encoder for 6G Image Transmission
Reference: LRISC Paper (IEEE 2025) - Latent Feature-Guided Conditional Diffusion
CORRECTED VERSION - Fixed CLIP compatibility issue
"""

import torch
from PIL import Image
import numpy as np
import os

# Try to load CLIP model
try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
    print("[OK] CLIP library found")
except ImportError:
    CLIP_AVAILABLE = False
    print("[!] CLIP not found - will use simulation mode")
    print("    Run: pip install transformers")

class SemanticEncoder:
    """
    Extracts semantic meaning from images
    This is the first component of the LRISC paper architecture
    """
    
    def __init__(self):
        print("\n[*] Initializing Semantic Encoder...")
        
        if CLIP_AVAILABLE:
            try:
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.model.eval()
                self.mode = "actual"
                print("[OK] Using actual CLIP model (real semantic extraction)")
            except Exception as e:
                print(f"[!] Could not load CLIP: {e}")
                self.mode = "simulation"
                print("[OK] Using simulation mode")
        else:
            self.mode = "simulation"
            print("[OK] Using simulation mode")
    
    def encode(self, image_path):
        """
        Convert image to semantic latent vector
        This produces the "latent features" mentioned in LRISC paper
        """
        # Validate image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Path is not a file: {image_path}")
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            raise ValueError(f"Image file is empty: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to open image {image_path}: {str(e)}")
        
        print(f"   [IMG] Processing: {os.path.basename(image_path)} ({file_size} bytes)")
        
        if self.mode == "simulation":
            return self._simulate_encode(image, image_path)
        
        # Real encoding with CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # FIXED: Get the image features correctly
            outputs = self.model.get_image_features(**inputs)
            
            # Check if we got a tensor or an object
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = outputs  # Direct tensor
        
        # Convert to numpy and normalize
        if torch.is_tensor(features):
            features_np = features.cpu().numpy()[0]
        else:
            features_np = features[0] if len(features.shape) > 1 else features
        
        # Normalize (as done in LRISC paper for channel robustness)
        norm = np.linalg.norm(features_np)
        if norm > 0:
            features_np = features_np / norm
        
        return {
            "latent_vector": features_np,
            "shape": features_np.shape,
            "mode": "actual",
            "image_name": os.path.basename(image_path)
        }
    
    def _simulate_encode(self, image, image_path):
        """
        Simulation mode - generates realistic features when CLIP not available
        Based on image properties (brightness, colors)
        """
        # Resize and convert to array
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        # Create 512-dim feature vector
        features = np.zeros(512)
        
        # Fill with meaningful values (not just random)
        features[0] = img_array.mean()                    # Overall brightness
        features[1] = img_array[:,:,0].mean()            # Red channel
        features[2] = img_array[:,:,1].mean()            # Green channel
        features[3] = img_array[:,:,2].mean()            # Blue channel
        features[4] = img_array.std()                     # Contrast
        features[5] = np.percentile(img_array, 90)        # Brightness percentile
        
        # Fill remaining with small random values (simulating other features)
        features[6:] = np.random.randn(506) * 0.1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return {
            "latent_vector": features,
            "shape": features.shape,
            "mode": "simulated",
            "image_name": os.path.basename(image_path)
        }


# ============= TEST THE ENCODER =============
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: Testing Semantic Encoder")
    print("=" * 60)
    
    # Create encoder
    encoder = SemanticEncoder()
    
    # Check if test_images folder has images
    test_folder = "test_images"
    if not os.path.exists(test_folder):
        print(f"\n[ERROR] '{test_folder}' folder not found!")
        print("   Please create 'test_images' folder and add some images.")
    else:
        images = [f for f in os.listdir(test_folder) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            print(f"\n[ERROR] No images found in '{test_folder}' folder!")
            print("   Please add at least one .jpg or .png image.")
        else:
            print(f"\n[DIR] Found {len(images)} image(s) in test_images/")
            
            # Test each image
            for img in images:
                print("\n" + "-" * 40)
                image_path = os.path.join(test_folder, img)
                result = encoder.encode(image_path)
                
                print(f"   [OK]  Encoding complete!")
                print(f"   [STATS] Mode: {result['mode']}")
                print(f"   [SHAPE] Latent vector shape: {result['shape']}")
                print(f"   [NUM] First 5 values: {[round(float(x), 3) for x in result['latent_vector'][:5]]}")
                print(f"   [NORM]  Vector norm: {np.linalg.norm(result['latent_vector']):.3f}")
            
            print("\n" + "=" * 60)
            print("[OK] STEP 2 COMPLETE! Semantic encoder is working.")
            print("=" * 60)
            print("\n[NEXT] Ready for Step 3: Channel Simulation")