# convert to pandas dataframe

from sktime.utils.load_data import load_from_tsfile_to_dataframe
train_x, train_y = load_from_tsfile_to_dataframe("../sktime/datasets/data/GunPoint/GunPoint_TRAIN.ts") 
test_x, test_y = load_from_tsfile_to_dataframe("../sktime/datasets/data/GunPoint/GunPoint_TEST.ts

