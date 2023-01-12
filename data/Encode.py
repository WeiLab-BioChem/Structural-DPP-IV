import os
import pickle as pkl
import re
from typing import Iterable

import torch
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor
from tqdm import tqdm

from data import StructuralEncode

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


def construct_StructDataset_Sequence(dataset_name: str, type_: str, sequences: list[str], labels, cubeBiased=False,
                                     cubeBias=0.2, right_align=False, use_cooked_data=False,
                                     max_seq_len: int = 90) -> int:
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = max_seq_len
    if use_cooked_data:
        data_num = detect_max_index_pkl_in_path(os.path.join('..', 'cooked_data', dataset_name, type_)) + 1
        return data_num

    sequences_code = codePeptides(sequences)
    index = 0
    for i in tqdm(range(len(sequences))):
        StructedData = construct_seq(sequences[i], cubeBiased=cubeBiased, cubeBias=cubeBias, right_align=right_align)
        store_one(dataset_name, type_, data_=(sequences_code[index], torch.FloatTensor(StructedData)),
                  label=labels[index], index=index)
        index += 1
    return index


def construct_seq(sequence, cubeBiased=False, cubeBias=0.2, right_align=False):
    SE = StructuralEncode
    Channel = SE.convert_to_graph_channel(sequence)
    return SE.convert_to_graph_channel_returning_maxSeqLenx15xfn(Channel, cubeBiased=cubeBiased,
                                                                 maxSeqLen=MAX_SEQ_LEN, cubeBias=cubeBias,
                                                                 right_align=right_align)


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
    # WPX, WAX, WRX, and WVX
    all_possible_aa = 'ACDEFGHIKLMNPQRSTVWY'
    seqs = []
    for pre in ['WP', 'WA', 'WR', 'WV']:
        for aa in all_possible_aa:
            seqs.append(pre + aa)
    # WPI, WPK, WAS, WRK, WRR, WVI, WVR is False, others are True
    labels = [1] * 80
    for i, seq in enumerate(seqs):
        if seq in ['WPI', 'WPK', 'WAS', 'WRK', 'WRR', 'WVI', 'WVR']:
            labels[i] = 0
    construct_StructDataset_Sequence('DPP-IV', 'test', sequences=seqs, labels=labels, use_cooked_data=False, max_seq_len=90)
    pass
