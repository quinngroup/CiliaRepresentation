from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    # source should be of shape (batch, 20, 10)
    def __init__(self, source):
        self.source = source
    
    def __getitem__(self, index):
        return self.source[index]
