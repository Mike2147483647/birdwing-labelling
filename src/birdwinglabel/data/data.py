import pandas as pd
import numpy as np
import os


# get the current working directory
current_dir = os.path.dirname(__file__)

# find path from current directory to data
path_fullBilateral = os.path.join(current_dir, "2024-03-24-FullBilateralMarkers.csv")
path_bilateral_frame = os.path.join(current_dir, "2025-03-24-bilateral_frame_info_df.csv")
path_bilateral_markers = os.path.join(current_dir, "2025-06-23-bilateral_markers.npy")
path_fullNoLabels = os.path.join(current_dir, "2025-06-23-FullNoLabels.csv")


# loading the data (csv, npy) with the path found above
full_bilateral_markers = pd.read_csv(path_fullBilateral)
bilateral_markers = np.load(path_bilateral_markers)


if __name__ == "__main__":

    # checking the head of the loaded dataframes and arrays
    print(full_bilateral_markers.head())
    print(bilateral_markers[:5])
