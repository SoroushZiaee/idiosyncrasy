import numpy as np
import torch as ch
from torch.utils.data import Dataset


class Partition:
    """
    generate random indices dividing data into train, test, and val sets.
    saves the indices, so you can easily use the same partition scheme on
    multiple datasets with the same original order. ie:

        partion(images)
        partion(response)

        is equivalent to

        idx = random_index
        images[idx]
        responses[idx]

    """

    def __init__(self, ntotal, ntrain, ntest, nval, seed=0, idx=None):
        # always generate the same random partition, for now
        np.random.seed(seed)
        self.ntotal = ntotal
        self.ntrain = ntrain
        self.ntest = ntest
        self.nval = nval

        # so we can supply the idx if we want to use the same partition scheme
        if idx:
            self.idx = idx
        else:
            self.idx = np.random.choice(ntotal, ntotal, replace=False)

        self.train_idx = self.idx[:ntrain]

        test_and_val_idx = self.idx[ntrain:]
        self.test_idx = test_and_val_idx[:ntest]
        self.val_idx = test_and_val_idx[ntest:]

        # make sure none of the indices are overlapping
        assert 0 == len(
            set(self.train_idx)
            .intersection(set(self.test_idx))
            .intersection(set(self.val_idx))
        )

    def __call__(self, X):
        return {
            "train": X[self.train_idx],
            "test": X[self.test_idx],
            "val": X[self.val_idx],
        }


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.

    __getitem__ operates with any index through modulo
    """

    def __init__(self, data, transform=None):
        assert all([data[0].shape[0] == datum.shape[0] for datum in data])
        self.data = [ch.Tensor(datum) for datum in data]
        self.transform = transform

    def __getitem__(self, index):
        # modulo index by length of data, so that we can any index
        N = self.__len__()
        index = int(index % N)

        datum = [datum[index] for datum in self.data]
        datum[0] = self.transform(datum[0])
        return datum

    def __len__(self):
        return self.data[0].size(0)


class OldCustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    takes
        Stimuli : nd.array,
        Target : nd.array
    and returns [transform(stimuli), target]
    __getitem__ operates with any index through modulo
    """

    def __init__(self, X, Y, transform=None):
        assert X.shape[0] == Y.shape[0]
        self.X = ch.Tensor(X)
        self.Y = ch.Tensor(Y)
        self.transform = transform

    def __getitem__(self, index):
        # modulo index by length of data, so that we can any index
        N = self.__len__()
        index = index % N

        X = self.transform(self.X[index])
        Y = self.Y[index]
        return (X, Y)

    def __len__(self):
        return self.X.size(0)
