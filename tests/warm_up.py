import birdwinglabel.dataprocessing.data as birdData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D



# get list of unique seqID of full_bilateral_markers
full_bilateral_seqID = birdData.full_bilateral_markers['seqID'].to_list()
full_bilateral_seqID = list(set(full_bilateral_seqID))
# print(full_bilateral_seqID)    # has length 1635

# find subset of first seqID
flight_id = birdData.full_bilateral_markers['seqID'][0]
print(flight_id)
flight_seq = birdData.full_bilateral_markers[
    birdData.full_bilateral_markers['seqID'] == flight_id
]
print(flight_seq)
flight_index = flight_seq.index.tolist()
print(flight_index)

# check the subset of this flight sequence matches with the np array supplied
flight_seq = flight_seq.iloc[:,12:36]
flight_seq = flight_seq.apply(lambda row: row.values.reshape(8,3), axis = 1)
print(flight_seq.head())
flight_seq = np.stack(flight_seq.to_numpy())
print(flight_seq[:5])
print(np.array_equal(flight_seq, birdData.bilateral_markers[flight_index]))

# subset the np array for the first sequence
flight1 = birdData.bilateral_markers[flight_index]

# plot a marker over time in x
def plot_single_marker_in_x():
    plt.plot(flight_index, flight1[:,0,0])
    plt.grid(True)
    plt.show()
    plt.close()


# plot a marker over time in 3d
def plot_single_marker_3d():

    # set coordinates and time
    t = flight_index
    x = flight_seq[:, 0, 0]
    y = flight_seq[:, 0, 1]
    z = flight_seq[:, 0, 2]

    # set up figure
    single_marker_fig = plt.figure()
    ax = single_marker_fig.add_subplot(projection='3d')
    line, = ax.plot([], [], [], lw=2)

    # Init function
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    # Animation function
    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    # Create animation
    ani = FuncAnimation(single_marker_fig, update, frames=len(t), init_func=init, blit=True, interval=50)

    plt.show()

# plot_single_marker_3d()

# plot a whole sequence
def plot_sequence():

    # add the origin for reference
    new_rows = np.zeros((len(flight_index),1,3))
    global flight_seq
    flight_seq = np.concatenate([flight_seq, new_rows], axis = 1)
    print(flight_seq[:5])

    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # set up edges in figure
    edges = [(6,7), (7,8), (6,8), (0,2), (2,4), (0,4),
             (1,3), (3,5), (1,5), (2,8), (4,8), (3,8), (5,8)]

    # Create one Line3D per edge
    lines = []
    for _ in edges:
        line = Line3D([], [], [], lw=2)
        ax.add_line(line)
        lines.append(line)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        for i, (start, end) in enumerate(edges):
            x = [flight_seq[frame, start, 0], flight_seq[frame, end, 0]]
            y = [flight_seq[frame, start, 1], flight_seq[frame, end, 1]]
            z = [flight_seq[frame, start, 2], flight_seq[frame, end, 2]]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
        return lines

    ani = FuncAnimation(fig, update, frames=flight_index, init_func=init,
                        blit=True, interval=50)

    plt.show()

plot_sequence()