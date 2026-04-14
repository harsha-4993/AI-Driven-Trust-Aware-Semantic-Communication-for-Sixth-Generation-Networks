"""
Step 3: Channel Simulator for 6G Semantic Transmission
Reference: LRISC Paper (IEEE 2025) - AWGN Channel Model
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ChannelSimulator:
    """
    Simulates 6G wireless channel conditions
    Uses Additive White Gaussian Noise (AWGN) as per LRISC paper
    """
    
    @staticmethod
    def awgn_noise(signal, snr_db):
        """
        Add AWGN noise to signal
        SNR in dB: Higher value = cleaner channel
        
        Formula from LRISC paper Section III-B
        """
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear if snr_linear > 0 else signal_power
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        
        return signal + noise
    
    @staticmethod
    def transmit(semantic_vector, snr_db):
        """
        Transmit semantic vector over noisy channel
        Returns received (noisy) vector
        """
        return ChannelSimulator.awgn_noise(semantic_vector, snr_db)
    
    @staticmethod
    def calculate_snr_quality(snr_db):
        """
        Returns quality description based on SNR value
        """
        if snr_db >= 25:
            return "Excellent", "#00ff00"
        elif snr_db >= 15:
            return "Good", "#88ff88"
        elif snr_db >= 8:
            return "Moderate", "#ffff00"
        elif snr_db >= 3:
            return "Poor", "#ff8800"
        else:
            return "Very Poor", "#ff0000"


class TransmissionExperiment:
    """
    Runs transmission experiments and measures quality
    """
    
    def __init__(self):
        self.results = []
    
    def run_experiment(self, original_vector, snr_db):
        """
        Run one transmission experiment at given SNR
        """
        # Transmit over channel
        received_vector = ChannelSimulator.transmit(original_vector, snr_db)
        
        # Calculate distortion (MSE)
        mse = np.mean((original_vector - received_vector) ** 2)
        
        # Calculate reconstruction quality (1 - normalized MSE)
        quality = max(0, min(1, 1 - mse * 10))
        
        return {
            "snr_db": snr_db,
            "received_vector": received_vector,
            "mse": mse,
            "quality": quality,
            "distortion": mse * 100
        }
    
    def run_snr_sweep(self, original_vector, snr_values=[0, 5, 10, 15, 20, 25, 30]):
        """
        Run experiments across multiple SNR values
        """
        print("\n" + "=" * 60)
        print("RUNNING CHANNEL SIMULATION EXPERIMENTS")
        print("=" * 60)
        
        for snr in snr_values:
            result = self.run_experiment(original_vector, snr)
            self.results.append(result)
            
            quality_desc, color = ChannelSimulator.calculate_snr_quality(snr)
            bar_length = int(result['quality'] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            print(f"   SNR {snr:2d} dB | {quality_desc:10s} | {bar} | Quality: {result['quality']:.2f}")
        
        return self.results
    
    def plot_results(self):
        """
        Create quality vs SNR graph
        """
        if not self.results:
            print("No results to plot")
            return
        
        snr_values = [r['snr_db'] for r in self.results]
        quality_values = [r['quality'] for r in self.results]
        mse_values = [r['mse'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Quality vs SNR
        ax1.plot(snr_values, quality_values, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('SNR (dB) → Better Channel →', fontsize=12)
        ax1.set_ylabel('Reconstruction Quality', fontsize=12)
        ax1.set_title('Semantic Transmission Quality vs Channel Condition', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add color zones
        ax1.axvspan(0, 5, alpha=0.2, color='red', label='Very Poor')
        ax1.axvspan(5, 12, alpha=0.2, color='orange', label='Poor')
        ax1.axvspan(12, 20, alpha=0.2, color='yellow', label='Moderate')
        ax1.axvspan(20, 30, alpha=0.2, color='green', label='Good')
        
        # Plot 2: MSE vs SNR
        ax2.plot(snr_values, mse_values, 'r-s', linewidth=2, markersize=8)
        ax2.set_xlabel('SNR (dB) → Better Channel →', fontsize=12)
        ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
        ax2.set_title('Distortion vs Channel Condition', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/channel_quality_plot.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: results/channel_quality_plot.png")
        
        plt.show()
        return fig


# ============= TEST THE CHANNEL SIMULATOR =============
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: Testing Channel Simulator")
    print("=" * 60)
    
    # Create a test semantic vector (simulating output from encoder)
    print("\n📡 Creating test semantic vector...")
    test_vector = np.random.randn(512)
    test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize
    print(f"   Vector shape: {test_vector.shape}")
    print(f"   Vector norm: {np.linalg.norm(test_vector):.3f}")
    
    # Run experiments
    experiment = TransmissionExperiment()
    results = experiment.run_snr_sweep(test_vector)
    
    # Print summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    
    for r in results:
        quality_desc, _ = ChannelSimulator.calculate_snr_quality(r['snr_db'])
        print(f"   SNR {r['snr_db']:2d} dB: {quality_desc} (Quality: {r['quality']:.2f}, MSE: {r['mse']:.4f})")
    
    # Plot results
    print("\n📈 Generating quality plot...")
    experiment.plot_results()
    
    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE! Channel simulator is working.")
    print("=" * 60)
    print("\n➡️ Ready for Step 4: Full Integration (Encoder + Channel)")