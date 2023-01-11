import os
import pickle as pkl
import re
from typing import Iterable

import torch
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor
from tqdm import tqdm

from data import MGF_enhance

aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
           'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
           'W': 20, 'Y': 21, 'V': 22, 'X': 23}

MAX_SEQ_LEN = -1


def detect_max_index_pkl_in_path(path: str) -> int:
    """
    In given path, detect the max index of pkl file.
    Use this O(log n) method to detect max index.
    """
    file_list = os.listdir(path)
    if len(file_list) == 0:
        return -1
    file_list = [int(re.findall(r'\d+', i)[0]) for i in file_list]
    return max(file_list)


def store_one(dataset_name, type_, data_, label, index):
    """
    Here we read in `wb` mode, which means if the file exists, it will be overwritten.
    """
    # store as (tensor, int)
    tup = (data_, label)
    # create or overwrite file ../cooked_data/<dataset_name>/<type_>/<index>.pkl
    with open(os.path.join('..', 'cooked_data', dataset_name, type_, str(index) + '.pkl'), 'wb') as f:
        pkl.dump(tup, f)


def construct_MGFDataset(dataset_name: str, type_: str, sequences, labels,
                         cubeBiased=False, cubeBias=0.2, right_align=False,
                         use_cooked_data=False, max_seq_len=61) -> int:
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_seq_len
    if use_cooked_data:
        data_num = detect_max_index_pkl_in_path(os.path.join('..', 'cooked_data', dataset_name, type_)) + 1
        return data_num
    original_set_len = len(sequences)
    index = 0
    Labels = labels
    for i in tqdm(range(len(sequences))):
        MGFdata = construct_seq(sequences[i], cubeBiased=cubeBiased, cubeBias=cubeBias, right_align=right_align)
        store_one(dataset_name, type_, torch.FloatTensor(MGFdata), Labels[index], index)
        index += 1
    return original_set_len


def construct_MGFDataset_sequence(dataset_name: str, type_: str, sequences: list[str], labels, cubeBiased=False,
                                  cubeBias=0.2, right_align=False, use_cooked_data=False, max_seq_len: int = 90) -> int:
    """
    构建数据集的文件
    :return: data, label, 数据增广情况下的也会返回，如果是100， len就会是400
    for now, if you don't want use `kmer`, some changes in linear layer of MGFPaddingZero.py shall be made.
    Or else the script cannot run.
    """
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_seq_len
    if use_cooked_data:
        data_num = detect_max_index_pkl_in_path(os.path.join('..', 'cooked_data', dataset_name, type_)) + 1
        return data_num

    sequences_code = codePeptides(sequences)
    index = 0
    for i in tqdm(range(len(sequences))):
        MGFdata = construct_seq(sequences[i], cubeBiased=cubeBiased, cubeBias=cubeBias, right_align=right_align)
        store_one(dataset_name, type_, data_=(sequences_code[index], torch.FloatTensor(MGFdata)),
                  label=labels[index], index=index)
        index += 1
    return index


def construct_seq(sequence, cubeBiased=False, cubeBias=0.2, right_align=False):
    MGF = MGF_enhance
    MGFChannel = MGF.convert_to_graph_channel(sequence)
    MGFdata = MGF.convert_to_graph_channel_returning_maxSeqLenx15xfn(MGFChannel, cubeBiased=cubeBiased,
                                                                     maxSeqLen=MAX_SEQ_LEN, cubeBias=cubeBias,
                                                                     right_align=right_align)

    return MGFdata


# noinspection GrazieInspection
def codePeptides(peptideSeq: Iterable[str]) -> Tensor:
    pep_codes = []
    for pep in peptideSeq:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))

    return rnn_utils.pad_sequence(pep_codes, batch_first=True)


def code_one_peptide(peptideSeq: str) -> Tensor:
    pep = []
    for aa in peptideSeq:
        pep.append(aa_dict[aa])
    return torch.tensor(pep)


if __name__ == '__main__':
    pass
