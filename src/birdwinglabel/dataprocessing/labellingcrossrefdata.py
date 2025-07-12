import pandas as pd
import numpy as np



unlabelled_df = pd.read_pickle('unlabelled_grouped_df.pkl')

# we now label unlabelled_df by turning entries with 1s into 1,...,8
# no need to care about gold_df actually, since we checked: for each data point (frameID)
# the first 8 rows of rot_xyz (matrix) in unlabelled is the same as its counterpart in gold

# for each frame, label 1-8 for those are 1 in mask, 0 otherwise
def create_labels(mask):
    labels = np.zeros_like(mask)
    idx = np.where(mask == 1)[0]
    labels[idx] = np.arange(1, len(idx)+1)
    return labels

labelled_df = pd.DataFrame({
    'frameID': unlabelled_df['frameID'],
    'rot_xyz': unlabelled_df['rot_xyz'],
    'labels': unlabelled_df['rot_xyz_mask'].apply(create_labels)
})



if __name__ == "__main__":

    # sanity check
    print(f'labelled_df info:')
    print(f'{labelled_df.info()}')
    print(
        f'sample: \nframeID: {labelled_df['frameID'][2]} \nmatrix: {labelled_df['rot_xyz'][2]} \nlabels: {labelled_df['labels'][2]}')

    labelled_df.to_pickle('labelled_df.pkl')

    gold_df = pd.read_pickle('gold_df.pkl')

    def check_all_matched():
        mismatch_count = 0
        for n in range(len(unlabelled_df)):
            unlabelled_matrix = unlabelled_df.iloc[n,1][:8]
            gold_matrix = gold_df.iloc[n,26]
            if not np.array_equal(unlabelled_matrix, gold_matrix):
                # print(f"Mismatch at index {n}")
                mismatch_count += 1

        if mismatch_count == 0:
            print("All first 8 rows match exactly.")
        else:
            print(f"Total mismatches: {mismatch_count}")



    check_all_matched()
    # print(np.array_equal(unlabelled_df.iloc[12356,1][:8],gold_df.iloc[12356,26]))

    def check_rot_xyz_match(row):
        # Extract columns 2-25 as a 1D array
        flat_cols = row.iloc[2:26].to_numpy()
        # Flatten rot_xyz_matrix
        flat_matrix = row['rot_xyz_matrix'].flatten()
        # Compare
        return np.array_equal(flat_cols, flat_matrix)


    # Apply to each row, result is a boolean Series
    matches = gold_df.apply(check_rot_xyz_match, axis=1)
    print(f'number of matches: {np.sum(matches)}')

    # for each data point (row in dfs), the matrix rot_xyz is in order as the markers labelled
    # first 8 rows of the matrix rot_xyz of unlabelled is the same as the gold

