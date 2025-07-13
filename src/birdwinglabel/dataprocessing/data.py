import pandas as pd
import numpy as np
from pathlib import Path


# get the current working directory
current_dir = Path(__file__).parent

# find path from current directory to data
data_dir = current_dir.parent / "data"

path_fullBilateral = data_dir / "2024-03-24-FullBilateralMarkers.csv"
path_bilateral_frame = data_dir / "2025-03-24-bilateral_frame_info_df.csv"
path_bilateral_markers = data_dir / "2025-06-23-bilateral_markers.npy"
path_fullNoLabels = data_dir / "2025-06-23-FullNoLabels.csv"



# loading the data (csv, npy) with the path found above
full_bilateral_markers = pd.read_csv(path_fullBilateral)
bilateral_markers = np.load(path_bilateral_markers)
full_no_labels = pd.read_csv(path_fullNoLabels)
bilateral_frame = pd.read_csv(path_bilateral_frame)

# remove duplicate frameID
full_bilateral_markers = full_bilateral_markers.drop_duplicates(subset='frameID')


# get list of seqID names
def get_list_of_seqID(bird_data):
    return bird_data['seqID'].unique().tolist()

# get list of frameID names
def get_list_of_frameID(bird_data):
    return bird_data['frameID'].unique().tolist()

# subset data by seqID
def subset_by_seqID(bird_data, seq_id_list):
    return bird_data[bird_data['seqID'].isin(seq_id_list)]

# subset data by frameID
def subset_by_frameID(bird_data, frame_id_list):
    return bird_data[bird_data['frameID'].isin(frame_id_list)]

# get list of seqID of unlabelled data
def unlabelled_seqID():
    labelled_seqID = get_list_of_seqID(full_bilateral_markers)
    all_seqID = get_list_of_seqID(full_no_labels)
    unlabelled_seqID = np.setdiff1d(all_seqID, labelled_seqID)
    return unlabelled_seqID

# for unlabelled dataset (full_no_label), filter and stack rot_xyz into n (no of markers) x 3 matrix
def stack_matrix(unlabelled_df):

    # unlabelled_df must have columns frameID, rot_xyz_1, rot_xyz_2, rot_xyz_3
    unlabelled_grouped_df = unlabelled_df.groupby(['frameID'])[['rot_xyz_1', 'rot_xyz_2', 'rot_xyz_3']].apply(
        lambda x: x.to_numpy()
    ).reset_index(name='rot_xyz')

    # output df has col1 frameID, col2 rot_xyz
    return unlabelled_grouped_df


# create training data
def create_training(labelled_data, seed = 1):

    num_rows = labelled_data.shape[0]

    # stack coords of labelled into 8x3 matrices
    markers_matrix = labelled_data.iloc[:,12:36].to_numpy().reshape(num_rows,8,3)
    print(f'np.shape(markers_matrix): {np.shape(markers_matrix)}')

    # create new column for labels
    single_label = np.arange(8).reshape(1, 8, 1)
    labels = np.tile(single_label, (num_rows, 1, 1))
    print(f'np.shape(labels): {np.shape(labels)}')

    # convert into list for pandas dataframe
    markers_matrix_list = [mat for mat in markers_matrix]
    labels_list = [lab for lab in labels]

    # create new dataframe
    training_data = pd.DataFrame({
        'markers_matrix' : markers_matrix_list,
        'label' : labels_list
    })

    # # sanity check
    # print(training_data.iloc[0,0])
    # print(training_data.iloc[0, 1])

    # swap rows randomly so that the machine doesnt learn to label everything 0:8
    rng = np.random.default_rng(seed)
    def permute_row(row):
        perm = rng.permutation(8)
        row['markers_matrix'] = row['markers_matrix'][perm]
        row['label'] = row['label'][perm]
        return row

    training_data = training_data.apply(permute_row, axis=1)

    # # sanity check
    # print(training_data.iloc[0, 0])
    # print(training_data.iloc[0, 1])

    return training_data


if __name__ == "__main__":

    print(full_no_labels.info())
    sample_seq = unlabelled_seqID()[0]
    sample_df = subset_by_seqID(full_no_labels, [sample_seq])
    sample_df = stack_matrix(sample_df)
    print(f'{sample_df.info()}')
    print(f'{sample_df.iloc[1,1]}')

    # get list of unique seqID of full_bilateral_markers
    # full_bilateral_seqID = get_list_of_seqID(full_bilateral_markers)
    # print(full_bilateral_seqID)    # has length 1635

    # find subset of 04_09_038_1
    # sample_seqn = subset_by_seqID(full_bilateral_markers, ['04_09_038_1'])


    # create training data from 04_09_038_1
    # train_set = create_training(sample_seqn)

    # print(f'unlabelled_seqID: {unlabelled_seqID()}')