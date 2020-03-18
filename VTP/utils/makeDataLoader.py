from sktime.utils.load_data import load_from_tsfile_to_dataframe
import pandas as pd
import numpy as np

# convert to pandas dataframe and then numpy array

filePath = ""

train_x, train_y = load_from_tsfile_to_dataframe(filePath).to_numpy()
test_x, test_y = load_from_tsfile_to_dataframe(filePath).to_numpy()


