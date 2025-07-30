import numpy as np
import pandas as pd
from pathlib import Path

import birdwinglabel.dataprocessing.data as data
from birdwinglabel.dataprocessing.data import full_no_labels

from birdwinglabel.visualisation.plot3d import transformer_labelled_df


# labelled by transformer
print(transformer_labelled_df.info())
labels = np.concatenate(transformer_labelled_df['labels'].to_numpy())
print(labels[1])
print(labels.shape)

num_zeros = np.sum(labels == 0)
total_entries = len(labels)
print(f"Transformer labelled: \nNumber of zeros: {num_zeros}")
print(f"Total entries: {total_entries}")


# half manual labelled
manual_seqID = data.get_list_of_seqID(data.full_bilateral_markers)      # gold data seqID
manual_df = data.subset_by_seqID(data.full_bilateral_markers, [manual_seqID[300]])
print(manual_df.info())

