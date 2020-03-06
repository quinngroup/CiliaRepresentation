# convert to pandas dataframe

filePath = ""

from sktime.utils.load_data import load_from_tsfile_to_dataframe
train_x, train_y = load_from_tsfile_to_dataframe(filePath) 
test_x, test_y = load_from_tsfile_to_dataframe(filePath)


