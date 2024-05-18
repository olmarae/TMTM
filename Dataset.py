import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from utils import sample_mask


class Dataset_TMTM(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_index = torch.tensor(edge_index, dtype = torch.int64)
        edge_type = torch.load(self.root + "/edge_type.pt")
        bot_label = torch.load(self.root + "/label.pt")
        features = torch.load(self.root + "/features.pt")
        features = features.to(torch.float32)
        
        num_prop=features[:,[0,1,2,3,4,5,9,12,17,18,19,20,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44,45]]
        cat_prop=features[:,[6,7,8,10,11,13,14,15,16,21,27,38]]
        des_tensor = torch.load(self.root + "/des_tensor.pt")
        tweets_tensor = torch.load(self.root + "/tweets_tensor.pt")


        print("Dimensions of numerical properties:", num_prop.shape)
        print("Dimensions of categorical properties:", cat_prop.shape)
        print("Dimensions of description tensor:", des_tensor.shape)
        print("Dimensions of tweets tensor:", tweets_tensor.shape)


        features = torch.cat([cat_prop, num_prop, des_tensor, tweets_tensor], axis=1)
        data = Data(x=features, y =bot_label, edge_index=edge_index)
        data.edge_type = edge_type

        data.y2 = bot_label
        sample_number = len(data.y2)

        train_idx = range(int(0.7*sample_number))
        val_idx = range(int(0.7*sample_number), int(0.9*sample_number))
        test_idx = range(int(0.9*sample_number), int(sample_number))

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])