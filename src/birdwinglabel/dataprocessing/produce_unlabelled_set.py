from data import full_no_labels
import numpy as np
import pandas as pd


# assign new objects to keep origin untouched

unlabelled_df = full_no_labels
print(f'{unlabelled_df.info()}')
# Data columns (total 13 columns):
#  #   Column        Dtype
# ---  ------        -----
#  0   frameID       object
#  1   seqID         object
#  2   HorzDistance  float64
#  3   time          float64
#  4   xyz_1         float64
#  5   xyz_2         float64
#  6   xyz_3         float64
#  7   rot_xyz_1     float64
#  8   rot_xyz_2     float64
#  9   rot_xyz_3     float64
#  10  XYZ_1         float64
#  11  XYZ_2         float64
#  12  XYZ_3         float64

# Assign new object to keep origin untouched
unlabelled_df = full_no_labels.copy()

# Filter to keep frameID, seqID, rot_xyz_1, rot_xyz_2, rot_xyz_3
filtered_df = unlabelled_df[['frameID', 'seqID', 'rot_xyz_1', 'rot_xyz_2', 'rot_xyz_3']]

# Group by frameID and seqID, stack rot_xyz columns into a matrix [num_marker, 3] per frame
grouped = filtered_df.groupby(['frameID', 'seqID'])

rows = []
for (frameID, seqID), group in grouped:
    rot_xyz = group[['rot_xyz_1', 'rot_xyz_2', 'rot_xyz_3']].to_numpy()
    rows.append({'frameID': frameID, 'rot_xyz': rot_xyz, 'seqID': seqID})

final_df = pd.DataFrame(rows, columns=['frameID', 'rot_xyz', 'seqID'])

# Remove frames with empty rot_xyz
final_df = final_df[final_df['rot_xyz'].apply(lambda x: x.shape[0] > 0)].reset_index(drop=True)

print(f'final_df: {final_df.iloc[0,0]} \n{final_df.iloc[0,1]} \n{final_df.iloc[0,2]}')
# Export as unlabelled_df.pkl
final_df.to_pickle('unlabelled_df.pkl')