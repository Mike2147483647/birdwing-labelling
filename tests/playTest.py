import birdwinglabel.data.data as birdData
import pandas as pd

# # checking the head of the loaded dataframes and arrays
# print(birdData.full_bilateral_markers.info())
# # print(birdData.bilateral_markers[:5])
# # full_bilateral_colnames = birdData.full_bilateral_markers.columns.tolist()
# # print(full_bilateral_colnames)
# print(birdData.full_no_labels.info())
# print(birdData.bilateral_frame.info())
# print(birdData.bilateral_frame[birdData.bilateral_frame['seqID'] == '04_09_038_1'])

df = pd.DataFrame({'A': [3, 1, 2], 'B': [9, 8, 7]})
df["A"] = pd.Categorical(df['A'], categories=[2,1,3], ordered=True)
df = df.sort_values('A').reset_index(drop=True)
print(df)