import torch.utils.data as data

# Define MyDataset
class MyDataset_1(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(MyDataset_1, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
