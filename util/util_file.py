from typing import List

import yaml


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            li = line.split('\t')
            labels.append(int(li[1]))
            sequences.append(li[-1])

    return sequences, labels


def load_csv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            li = line.split(',')
            sequences.append(li[0])
            labels.append(int(li[1]))

    return sequences, labels


def load_DPP_IV_data(type_: str = 'train') -> (List[str], List[int]):
    # 读取DPP-IV数据集
    assert type_ in ['train', 'valid', 'test']
    if type_ == 'valid':
        type_ = 'test'
    if type_ == 'train':
        filename = '../data/DPP-IV/train/train.tsv'
    else:
        filename = '../data/DPP-IV/test/test.tsv'
    all_sequences, all_labels = load_tsv_format_data(filename)
    print("\n[INFO]\tmax len of dataset `DPP-IV {}`: {}".format(type_,
                                                                max([len(seq) for seq in all_sequences])))
    print("[INFO]\tmin len of dataset `DPP-IV {}`: {}".format(type_,
                                                              min([len(seq) for seq in all_sequences])))
    return all_sequences, all_labels


def read_yaml_to_dict(yaml_path: str):
    # 读取yaml并转为dict
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


# noinspection PyUnusedLocal
def method_driver(dataset_, type_, sub_set) -> (List[str], List[int]):
    all_sequences, all_labels = load_DPP_IV_data(type_)
    return all_sequences, all_labels


if __name__ == '__main__':
    pass
