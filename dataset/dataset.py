from torch.utils.data import Dataset
import torch


class OthelloDataset(Dataset):
    def __init__(self, train = True):
        self.PADDING_TOKEN = 0
        self.train = train

        if train:
            self.data = torch.load("dataset/train_dataset.pt")
        else:
            self.data = torch.load("dataset/test_dataset.pt")

        self.target = torch.full_like(self.data, self.PADDING_TOKEN)
        self.target[:, :-1] = self.data[:, 1:]

    def __len__(self):
        return self.data.size(dim=0)

    def __getitem__(self, item):
        x = self.data[item].long()
        y = self.target[item].long()

        return x, y