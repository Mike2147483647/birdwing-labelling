import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
from pathlib import Path

import birdwinglabel.dataprocessing.data as data

from plot3d import plot_sequence


# prep the dataset for plot
data_dir = Path(__file__).parent.parent / 'dataprocessing' / 'labelled_df.pkl'
labelled_df = pd.read_pickle(data_dir)

seqID_list = data.get_list_of_seqID(data.full_bilateral_markers)
sample_seqID = seqID_list[3]
sample_data_without_label = data.subset_by_seqID(data.full_bilateral_markers, [sample_seqID])
sample_frameID = data.get_list_of_frameID(sample_data_without_label)
sample_df = data.subset_by_frameID(labelled_df, sample_frameID)

# plot
plot_sequence(sample_df)


