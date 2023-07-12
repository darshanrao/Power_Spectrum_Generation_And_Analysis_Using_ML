import pandas as pd

# Assuming you have the original data in a DataFrame called 'data'

data = pd.read_csv("combined_data.csv")

# Define the bin ranges for each variable
hii_bin_range = [5, 200]
ion_bin_range = [4, 6]
rbubble_bin_range = [5, 20]

# Define the number of bins
num_bins = 25

# Compute the bin widths for each variable
hii_bin_width = (hii_bin_range[1] - hii_bin_range[0]) / num_bins
ion_bin_width = (ion_bin_range[1] - ion_bin_range[0]) / num_bins
rbubble_bin_width = (rbubble_bin_range[1] - rbubble_bin_range[0]) / num_bins

# Create the bin labels for each variable
hii_bin_labels = [i+1 for i in range(num_bins)]
ion_bin_labels = [i+1 for i in range(num_bins)]
rbubble_bin_labels = [i+1 for i in range(num_bins)]

# Apply binning to each variable
data['HII_EFF_FACTOR_bin'] = pd.cut(data['HII_EFF_FACTOR'], bins=num_bins, labels=hii_bin_labels, include_lowest=True)
data['ION_Tvir_MIN_bin'] = pd.cut(data['ION_Tvir_MIN'], bins=num_bins, labels=ion_bin_labels, include_lowest=True)
data['R_BUBBLE_MAX_bin'] = pd.cut(data['R_BUBBLE_MAX'], bins=num_bins, labels=rbubble_bin_labels, include_lowest=True)

# Save the binning results to a CSV file
data.to_csv('binned_data.csv', index=False)