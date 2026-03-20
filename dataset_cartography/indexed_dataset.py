from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    """
    元のデータセットを変更せずにインデックスを取得するためのラッパークラス。
    (インデックス, 画像, 正解ラベル) の3つを返します。
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

    def __len__(self):
        return len(self.dataset)