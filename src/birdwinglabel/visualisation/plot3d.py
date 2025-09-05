import pandas as pd

import birdwinglabel.dataprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3D
from pathlib import Path




# plot a whole sequence
def plot_sequence(labelled_df, interval_len:int = 500, save_file_path = None):
    '''
    :param labelled_df:
    :param interval_len: int, the larger the slower
    :param save_file_path: string must be end with .mp4
    :return:
    '''
    # Define edges between labels 1-8 (example: [(1,2), (2,3), ...])
    edges = [(1,3),(1,5),(3,5),
             (2,4),(2,6),(4,6),
             (7,8)]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Add origin point
    origin_scatter = ax.scatter([0], [0], [0], color='red', s=50, label='Origin')

    # Prepare lines for each edge
    lines = [Line3D([], [], [], lw=2) for _ in edges]
    for line in lines:
        ax.add_line(line)

    # Add scatter for markers 1-8
    marker_scatter = ax.scatter([], [], [], color='blue', s=30, label='Markers 1-8', alpha=1.0)

    # Add scatter for non markers
    zero_scatter = ax.scatter([], [], [], color='green', s=30, label='Label 0', alpha=1.0)

    paused = [False]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        marker_scatter._offsets3d = ([], [], [])
        return lines + [marker_scatter]

    last_marker_coords = [[], [], []]

    def update(frame):
        if not paused[0]:
            coords = np.asarray(labelled_df.iloc[frame, 1])
            labels = np.asarray(labelled_df.iloc[frame, 2])
            mask = np.isin(labels, np.arange(1, 9))
            marker_coords = coords[mask]
            if marker_coords.shape[0] > 0:
                marker_scatter._offsets3d = (marker_coords[:, 0], marker_coords[:, 1], marker_coords[:, 2])
                last_marker_coords[0] = marker_coords[:, 0]
                last_marker_coords[1] = marker_coords[:, 1]
                last_marker_coords[2] = marker_coords[:, 2]
            else:
                marker_scatter._offsets3d = ([], [], [])
                last_marker_coords[0] = []
                last_marker_coords[1] = []
                last_marker_coords[2] = []
            coords = np.asarray(labelled_df.iloc[frame, 1])
            labels = np.asarray(labelled_df.iloc[frame, 2])
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            for i, (start, end) in enumerate(edges):
                if start in label_to_idx and end in label_to_idx:
                    s_idx, e_idx = label_to_idx[start], label_to_idx[end]
                    x = [coords[s_idx, 0], coords[e_idx, 0]]
                    y = [coords[s_idx, 1], coords[e_idx, 1]]
                    z = [coords[s_idx, 2], coords[e_idx, 2]]
                    lines[i].set_data(x, y)
                    lines[i].set_3d_properties(z)
                else:
                    lines[i].set_data([], [])
                    lines[i].set_3d_properties([])
            mask = np.isin(labels, np.arange(1, 9))
            marker_coords = coords[mask]
            if marker_coords.shape[0] > 0:
                marker_scatter._offsets3d = (marker_coords[:, 0], marker_coords[:, 1], marker_coords[:, 2])
            else:
                marker_scatter._offsets3d = ([], [], [])
            zero_mask = labels == 0
            zero_coords = coords[zero_mask]
            if zero_coords.shape[0] > 0:
                zero_scatter._offsets3d = (zero_coords[:, 0], zero_coords[:, 1], zero_coords[:, 2])
            else:
                zero_scatter._offsets3d = ([], [], [])
            return lines + [marker_scatter, zero_scatter]
        else:
            marker_scatter._offsets3d = (last_marker_coords[0], last_marker_coords[1], last_marker_coords[2])
        return lines + [marker_scatter, zero_scatter]

    def toggle_pause(event):
        paused[0] = not paused[0]

    ani = FuncAnimation(fig, update, frames=len(labelled_df), init_func=init,
                        blit=False, interval= interval_len)
    if save_file_path is not None:
        ani.save(save_file_path, writer='ffmpeg')

    # Add pause button
    axpause = plt.axes([0.8, 0.01, 0.1, 0.05])
    bpause = Button(axpause, 'Pause/Resume')
    bpause.on_clicked(toggle_pause)

    plt.show()


data_dir =  Path(__file__).parent.parent / 'EncoderOnlyTransformers' / 'Transformer_labelled_df.pkl'
transformer_labelled_df = pd.read_pickle(data_dir)

if __name__ == '__main__':
    # load dataset


    plot_sequence(transformer_labelled_df)

