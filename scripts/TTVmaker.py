#!/usr/bin/env python
"""Script to reduce the original dataset and create training, validation and test set."""
import h5py
import numpy as np
import argparse

def create_subset():
    parser = argparse.ArgumentParser(description="Create sub-sampled H5 file.")
    parser.add_argument("input_file", help="Path to input H5 file (L.h5)")
    parser.add_argument("output_file", help="Path to output H5 file")
    # Define row counts as arguments
    parser.add_argument("-Q", type=int, required=True, help="Rows for Dataset A (from noise)")
    parser.add_argument("-W", type=int, required=True, help="Rows for Dataset B (from noise)")
    parser.add_argument("-E", type=int, required=True, help="Rows for Dataset C (noise part)")
    parser.add_argument("-R", type=int, required=True, help="Rows for Dataset C (injection part)")
    
    args = parser.parse_args()

    with h5py.File(args.input_file, 'r') as source, h5py.File(args.output_file, 'w') as dest:
        print(f"Opening {args.input_file}...")
        
        # Load source datasets (using read-only to save memory if needed)
        noise = source['noise']
        injection = source['injection']
        
        total_noise_rows = noise.shape[0]
        total_inj_rows = injection.shape[0]

        # --- 1. Generate Non-Continuous Indices ---
        # We need Q + W + E unique rows from 'noise' to avoid overlap (optional, but good practice)
        total_noise_needed = args.Q + args.W + args.E
        
        if total_noise_needed > total_noise_rows:
            raise ValueError(f"Not enough rows in 'noise'! Needed {total_noise_needed}, have {total_noise_rows}.")

        # Select random indices from noise (not continuous)
        noise_indices = np.random.choice(total_noise_rows, total_noise_needed, replace=False)
        
        # Split these indices for A, B, and C
        idx_A = np.sort(noise_indices[:args.Q])
        idx_B = np.sort(noise_indices[args.Q : args.Q + args.W])
        idx_C_noise = np.sort(noise_indices[args.Q + args.W :])
        
        # Select random indices from injection for C
        if args.R > total_inj_rows:
            raise ValueError(f"Not enough rows in 'injection'! Needed {args.R}, have {total_inj_rows}.")
        
        idx_C_inj = np.sort(np.random.choice(total_inj_rows, args.R, replace=False))

        print("Indices selected. Writing new datasets...")

        # --- 2. Write Dataset A ---
        # fancy indexing: noise[idx_A] reads only those specific rows
        data_A = noise[idx_A] 
        dest.create_dataset('A', data=data_A, chunks=True) 
        print(f"Created Dataset A: {data_A.shape}")

        # --- 3. Write Dataset B ---
        data_B = noise[idx_B]
        dest.create_dataset('B', data=data_B, chunks=True)
        print(f"Created Dataset B: {data_B.shape}")

        # --- 4. Write Dataset C (Merged) ---
        # Get the two parts
        part_noise = noise[idx_C_noise]
        part_inj = injection[idx_C_inj]
        
        # Concatenate them
        data_C = np.concatenate((part_noise, part_inj), axis=0)
        
        # Optional: Shuffle C so noise/injection aren't just stacked top/bottom
        #np.random.shuffle(data_C)
        
        dest.create_dataset('C', data=data_C, chunks=True)
        print(f"Created Dataset C: {data_C.shape} (Merged)")

if __name__ == "__main__":
    create_subset()