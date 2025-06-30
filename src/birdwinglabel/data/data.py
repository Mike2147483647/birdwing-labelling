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
full_no_labels = pd.read_csv(path_fullNoLabels)
bilateral_frame = pd.read_csv(path_bilateral_frame)


if __name__ == "__main__":

    # get list of unique seqID of full_bilateral_markers
    full_bilateral_seqID = full_bilateral_markers['seqID'].to_list()
    full_bilateral_seqID = list(set(full_bilateral_seqID))
    # print(full_bilateral_seqID)    # has length 1635

    # find subset of first seqID
    flight_id = full_bilateral_markers['seqID'][0]
    print(flight_id)
    flight_seq = full_bilateral_markers[
        full_bilateral_markers['seqID'] == flight_id
    ]
    print(flight_seq)
    flight_index = flight_seq.index.tolist()
    print(flight_index)

    # subset it into just coordinates
    flight_seq = flight_seq.iloc[:,12:36]
    flight_seq = flight_seq.apply(lambda row: row.values.reshape(8,3), axis = 1)
    print(flight_seq.head())
    flight_seq = np.stack(flight_seq.to_numpy())
    print(flight_seq[:5])
    print(np.array_equal(flight_seq, bilateral_markers[flight_index]))