import numpy as np
import torch
import sys
import os

# Add repo to path
sys.path.append(os.getcwd())

import model

def verify_amag():
    print("Initializing ModelWrapper...")
    wrapper = model.load()
    
    # Test Case 1: Monkey 'affi' (239 channels)
    print("\n--- Test Case 1: Monkey 'affi' (239 channels) ---")
    B = 2
    T = 20
    N = 239
    F = 9
    
    # Create dummy input: (Batch, 20, 239, 9)
    # Fill with recognizable pattern to check first 10 steps preservation
    dummy_input = np.random.randn(B, T, N, F).astype(np.float32)
    # Set first 10 steps of feature 0 to a specific value to verify identity preservation
    dummy_input[:, :10, :, 0] = 7.0 
    
    output = wrapper(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    if output.shape == (B, 20, N):
        print("PASS: Output shape is correct.")
    else:
        print(f"FAIL: Output shape incorrect. Expected ({B}, 20, {N}).")
        
    # Verify preservation of observed data
    preserved_chunk = output[:, :10, :]
    expected_chunk = dummy_input[:, :10, :, 0]
    
    if np.allclose(preserved_chunk, expected_chunk, atol=1e-5):
        print("PASS: First 10 steps preserved correctly.")
    else:
        print("FAIL: First 10 steps NOT preserved.")
        diff = np.abs(preserved_chunk - expected_chunk).max()
        print(f"Max difference: {diff}")

    # Test Case 2: Monkey 'beignet' (87 channels)
    print("\n--- Test Case 2: Monkey 'beignet' (87 channels) ---")
    N_b = 87
    dummy_input_b = np.random.randn(B, T, N_b, F).astype(np.float32)
    
    output_b = wrapper(dummy_input_b)
    
    print(f"Input Shape: {dummy_input_b.shape}")
    print(f"Output Shape: {output_b.shape}")
    
    if output_b.shape == (B, 20, N_b):
        print("PASS: Output shape for 'beignet' is correct.")
    else:
        print(f"FAIL: Output shape for 'beignet' incorrect.")

if __name__ == "__main__":
    verify_amag()
