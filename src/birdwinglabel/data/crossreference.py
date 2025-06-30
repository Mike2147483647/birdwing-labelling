from data import full_no_labels, full_bilateral_markers, bilateral_markers
import numpy as np
import pandas as pd


# we want to produce M(t)

# assign new objects to keep origin untouched
labelled_df = full_bilateral_markers
unlabelled_df = full_no_labels
labelled_npy = bilateral_markers


# for labelled data
# remove duplicate rows
labelled_df = labelled_df.drop_duplicates()

# filter into frame with only frameID, seqID and rot_xyz
labelled_df = labelled_df.iloc[:,np.r_[0:2, 12:36]]

# reshape into matrix for each frame
labelled_df['rot_xyz_matrix'] = [
    labelled_df.iloc[i, 2:26].values.reshape(8,3)
    for i in range(labelled_df.shape[0])
]


# for unlabelled data
# filter unlabelled to have frames only also exist in labelled
shared_cols = ['frameID', 'seqID']
unlabelled_filtered_df = unlabelled_df.merge(labelled_df[shared_cols], on = shared_cols, how='inner')
# print(unlabelled_filtered_df.info())

# continue filtering into a frame with only frameID, seqID and rot_xyz
unlabelled_filtered_df = unlabelled_filtered_df.iloc[:,[0,1,7,8,9]]

# reshape into matrices of k rows x 3 columns for each frame
unlabelled_grouped_df = unlabelled_filtered_df.groupby(['frameID', 'seqID'])[['rot_xyz_1', 'rot_xyz_2', 'rot_xyz_3']].apply(
    lambda x: x.to_numpy()
).reset_index(name='rot_xyz')

# sort unlabelled data to have same order as labelled in frameID column
order_by_frameID = labelled_df['frameID']
# set implicitly the order in frameID of unlabelled data
unlabelled_grouped_df['frameID'] = pd.Categorical(
    # input unlabelled and order in labelled
    unlabelled_grouped_df['frameID'],
    categories=order_by_frameID,
    ordered=True
)
# sort the unlabelled data with the order set before and drop the old indices of rows
unlabelled_grouped_df = unlabelled_grouped_df.sort_values('frameID').reset_index(drop=True)



# create mapping from frameID to labelled matrix
labelled_matrix_dict = dict(
    zip(labelled_df['frameID'], labelled_df['rot_xyz_matrix'])
)

# define the matching function
def match_rows_to_labelled(unlabelled_matrix, frame_id):
    # this wont be null since we filtered unlabelled to have rows that shares the frameID with labelled
    labelled_matrix = labelled_matrix_dict.get(frame_id)
    return np.array([
        1 if np.any(np.all(row == labelled_matrix, axis=1)) else 0
        for row in unlabelled_matrix
    ])

# apply to each row in unlabelled_grouped_df
unlabelled_grouped_df['rot_xyz_mask'] = unlabelled_grouped_df.apply(
    lambda row: match_rows_to_labelled(row['rot_xyz'], row['frameID']),
    axis=1
)

# save the produced dataset
unlabelled_grouped_df.to_csv('cross-reference.csv', index=False)
