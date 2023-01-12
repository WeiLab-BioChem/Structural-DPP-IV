import os
import pickle as pkl

import torch
import torch.utils.data as Data
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import DataLoader

from data import Encode
from util.util_file import method_driver

now_stage = None
def get_data_from_disk(dataset_name: str, type_, index):
    # path: ../cooked_data/<dataset_name>/<type_>/<index>.pkl
    path = os.path.join('../cooked_data', get_dataset_name(dataset_name), type_, str(index) + '.pkl')
    if not os.path.exists(path):
        raise ValueError('Path not exists: {}'.format(path))
    with open(path, 'rb') as file:
        return pkl.load(file)


class SeqDataSet(Data.Dataset):
    def __init__(self, dataset_name, type_: str, dataset_len):
        self.dataset_name = dataset_name
        self.dataset_len = dataset_len
        if type_ not in ['train', 'test', 'valid']:
            raise ValueError('type_ must be train, test or valid')
        self.type_ = type_

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return get_data_from_disk(self.dataset_name, self.type_, idx)


def get_dataset_name(dataset_name: str) -> str:
    return dataset_name


class SeqDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.raw_train_data, self.raw_train_label = None, None
        self.raw_valid_data, self.raw_valid_label = None, None
        self.raw_test_data, self.raw_test_label = None, None

        self.processed_train_data, self.processed_train_label = None, None
        self.processed_valid_data, self.processed_valid_label = None, None
        self.processed_test_data, self.processed_test_label = None, None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        dataset_name = get_dataset_name(self.args.path_train_data)
        MAX_SEQ_LEN: int = self.args.max_seq_len
        # if dataset_name directly not in ../cooked_data, then create folder ../cooked_data/<dataset_name>
        if not os.path.exists(os.path.join('../cooked_data', dataset_name)):
            os.makedirs(os.path.join('../cooked_data', dataset_name))
            # and make subdir ../cooked_data/<dataset_name>/train, ../cooked_data/<dataset_name>/valid, ../cooked_data/<dataset_name>/test if not exist
            li: list[str] = ['train', 'valid', 'test']
            for i in li:
                if not os.path.exists(os.path.join('../cooked_data', dataset_name, i)):
                    os.makedirs(os.path.join('../cooked_data', dataset_name, i))
        use_cooked_data = self.args.use_cooked_data
        train_sub_set, valid_sub_set, test_sub_set = self.args.train_sub_set, self.args.valid_sub_set, self.args.test_sub_set
        if stage == 'fit' or stage is None:
            if dataset_name in ['DPP-IV']:
                self.raw_train_data, self.raw_train_label = method_driver(dataset_name, 'train', train_sub_set)
                self.raw_valid_data, self.raw_valid_label = method_driver(dataset_name, 'valid', valid_sub_set)
            else:
                raise ValueError('dataset_name {} not supported. Expected DPP-IV.'.format(dataset_name))

            train_set_len = Encode.construct_StructDataset_Sequence(dataset_name, 'train', self.raw_train_data,
                                                                    self.raw_train_label,
                                                                    use_cooked_data=use_cooked_data,
                                                                    max_seq_len=MAX_SEQ_LEN)
            valid_set_len = Encode.construct_StructDataset_Sequence(dataset_name, 'valid', self.raw_valid_data,
                                                                    self.raw_valid_label,
                                                                    use_cooked_data=use_cooked_data,
                                                                    max_seq_len=MAX_SEQ_LEN)
            self.train_dataset = SeqDataSet(dataset_name, 'train', train_set_len)
            self.valid_dataset = SeqDataSet(dataset_name, 'valid', valid_set_len)
            print('self.train_dataset', len(self.train_dataset))
            print('self.valid_dataset', len(self.valid_dataset))
        elif stage == 'test' or stage is None:
            global now_stage
            now_stage = 'test'
            self.raw_test_data, self.raw_test_label = method_driver(dataset_name, 'test', test_sub_set)
            # test_set_len = len(self.raw_test_data)
            test_set_len = Encode.construct_StructDataset_Sequence(dataset_name, 'test', self.raw_test_data, self.raw_test_label,
                                                    use_cooked_data=use_cooked_data, max_seq_len=MAX_SEQ_LEN, )
            self.test_dataset = SeqDataSet(dataset_name, 'test', test_set_len)
            print('self.test_dataset', len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)


if __name__ == '__main__':
    tmp_var = (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 1)
    pass
