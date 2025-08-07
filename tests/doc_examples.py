import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from birdwinglabel.common.createtorchdataset import MarkerDataset, HotMarkerDataset

matrices = [np.random.rand(5,3) for _ in range(5)]
labels = [np.random.randint(0,9,5) for _ in range(5)]

data = pd.DataFrame(data = {'markers_matrix': matrices,
                               'label': labels
                           }
                    )
dataset = HotMarkerDataset(data, 9)
dataloader = DataLoader(dataset, batch_size=50)

