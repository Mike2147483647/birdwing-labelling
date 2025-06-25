import pandas as pd
import numpy as np


# loading the numpy array and the associating dataframe
full_bilateral_markers = pd.read_csv("2025-03-24-bilateral_frame_info_df.csv")
bilateral_markers = np.load("2025-06-23-bilateral_markers.npy")

# checking the head of the loaded dataframes and arrays
print(full_bilateral_markers.head())
print(bilateral_markers[:5])