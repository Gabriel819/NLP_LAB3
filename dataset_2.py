import torch.utils.data as data

# Define MyDataset
class MyDataset_2(data.Dataset):
    def __init__(self, l2r_tensor, r2l_tensor, y_tensor, mask, orig_len_tensor):
        super(MyDataset_2, self).__init__()

        self.l2r = l2r_tensor
        self.r2l = r2l_tensor
        self.y = y_tensor
        self.mask = mask
        self.orig_len = orig_len_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.l2r[index], self.r2l[index], self.y[index], self.mask[index], self.orig_len[index]