import os
import numpy as np
import pandas as pd

combined_data = []
combined_df = pd.DataFrame(columns=["FileNumber"])

for i in range(10000):
    csv_path = f"./DATA/data_{i}/data.csv"
    npz_path = f"./DATA/data_{i}/simple_mcmc_data_7.npz"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["FileNumber"] = i  # Add file number column
        combined_df = combined_df.append(df)
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            data2 = data['delta']
            combined_data.append(data2)

if combined_data:
    combined_data = np.vstack(combined_data)
    np.savez("combined_data.npz", data=combined_data)

if not combined_df.empty:
    combined_df.to_csv("combined_data.csv", index=False)