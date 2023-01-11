from typing import List

import numpy as np
from rdkit import Chem

from util.util_dicts import pre_known_charges, aa_dict


# noinspection SpellCheckingInspection,PyTypeChecker
def atom_feature(atom) -> np.ndarray:
    """
    Features of atoms in base molecule.
    """
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O',
                                           'S']) +  # # whether there is an atom in the amino acid molecule
                    # 4 types of atoms
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3]) +  # # the degree of the atom in the molecule
                    one_of_k_encoding(atom.GetTotalNumHs(),
                                      [0, 1, 2, 3]) +  # # the total number of hydrogen on the atom
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3]) +  # # the implicit valence
                    [atom.GetIsAromatic()] +  # # whether the atom is aromatic
                    get_ring_info(atom) +  # # the size of atomic ring
                    [int(atom.GetHybridization()) + 1]  # # the hybridization of the atom
                    # # Formal charge of the atom is all 0, so we do not need to use this feature.
                    # # We will also use Gasteiger charge, but this feature will be added later.
                    # # Every time this function is called, the Gasteiger charge will be added after calling.
                    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_ring_info(atom):
    # This function is used to get the size of the ring to which the atom belongs.
    # But, it can only recognize the ring size of 5 and 6.
    ring_info_feature = []
    for i in range(5, 7):  # # base A: 5,6
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)

    return ring_info_feature


def norm_Adj(adjacency):
    """
    Obtain the normalized adjacency matrix.
    """
    I = np.array(np.eye(adjacency.shape[0]))  # # identity matrix
    adj_hat = adjacency + I

    # D^(-1/2) * (A + I) * D^(-1/2)
    D_hat = np.diag(np.power(np.array(adj_hat.sum(1)), -0.5).flatten(), 0)  # # degree matrix
    adj_Norm = adj_hat.dot(D_hat).transpose().dot(D_hat)  # # normalized adjacency matrix

    return adj_Norm


def norm_fea(features):
    """
    Obtain the normalized node feature matrix.
    """
    norm_fea_ = features / features.sum(1).reshape(-1, 1)  # # normalized node feature matrix

    return norm_fea_


def convert_to_graph_channel_returning_maxSeqLenx15xfn(cube: np.ndarray, maxSeqLen, cubeBiased=False,
                                                       cubeBias=0.2, right_align=False) -> np.ndarray:
    my_len = cube.shape[0]  # Note that the shape of cube is (seq_len, maxNumAtoms, 21)
    if my_len <= maxSeqLen:
        if not cubeBiased:
            # pad with 0s
            if not right_align:
                cube = np.pad(cube, ((0, maxSeqLen - my_len), (0, 0), (0, 0)), 'constant',
                              constant_values=0)
            else:
                cube = np.pad(cube, ((maxSeqLen - my_len, 0), (0, 0), (0, 0)), 'constant',
                              constant_values=0)
            return cube
        else:
            start_index = int(maxSeqLen * cubeBias)
            if start_index + my_len >= maxSeqLen:
                start_index = maxSeqLen - my_len
            cube = np.pad(cube, ((start_index, maxSeqLen - my_len - start_index), (0, 0), (0, 0)), 'constant',
                          constant_values=0)
            return cube
    else:
        raise ValueError(f"The length of the sequence is greater than {maxSeqLen}. This is not Expected.")


# noinspection PyUnresolvedReferences
def convert_to_graph_channel(seq: str):
    """
    Our Idea:
        This function is used to convert a sequence to a 3d ndarray.

    Args:
        seq: str of seq. For example, "MSEK"

    Returns:
        cube: shape(seq_len, maxNumAtoms, 21)
        Note that maxNumAtoms is 15, if the code below is not modified.
    """
    seq = seq.lower()
    maxNumAtoms = 15  # amino acid W has 15 atoms, so we set the maximum number of atoms to 15.
    # Molecules of bases from one sequence
    graphFeaturesOneSeq = []  # the features of the molecular graph of one sequence
    seqSMILES: List = [aa_dict[b] for b in seq]
    i_ = 0
    for aminoAcidSMILES in seqSMILES:
        aaMol = Chem.MolFromSmiles(aminoAcidSMILES)

        # Adjacency matrix
        AdjTmp = Chem.GetAdjacencyMatrix(aaMol)
        AdjNorm = norm_Adj(AdjTmp)
        # Node feature matrix (features of node (atom))
        if AdjNorm.shape[0] <= maxNumAtoms:

            # Preprocessing of feature
            graphFeature = np.zeros((maxNumAtoms, 21))
            nodeFeatureTmp = []  # Each element is a vector of features of an atom
            for atom in aaMol.GetAtoms():
                var_tmp = atom_feature(atom)
                nodeFeatureTmp.append(var_tmp)
            nodeFeatureNorm = norm_fea(np.asarray(nodeFeatureTmp))
            # add one new all-zero column to the end of the nodeFeatureNorm
            nodeFeatureNorm = np.insert(nodeFeatureNorm, nodeFeatureNorm.shape[1], 0, axis=1)
            for j_ in range(len(aaMol.GetAtoms())):
                # add partial charge to the last dimension
                nodeFeatureNorm[j_, -1] = pre_known_charges[seq[i_]][j_]

            # Molecular graph feature for one base
            seqLen = AdjNorm.shape[0]
            # assert seqLen == len(nodeFeatureTmp)
            # print("shape of nodeFeatureNorm: ", nodeFeatureNorm.shape)
            graphFeature[0:seqLen, 0:20] = np.dot(AdjNorm.T, nodeFeatureNorm)

            # the last dimension of the feature is the partial charge of the atom.
            # The partial charge of the atom is stored in ./UtilDicts.py
            # That is, in pre_known_charges dict.
            for i in range(len(aaMol.GetAtoms())):
                graphFeature[i, -1] = pre_known_charges[seq[i_]][i]

            # Append the molecular graph features for bases in order
            graphFeaturesOneSeq.append(graphFeature)
        i_ += 1

    # Molecular graph features for one sequence
    graphFeaturesOneSeq = np.asarray(graphFeaturesOneSeq, dtype=np.float32)
    return graphFeaturesOneSeq


if __name__ == '__main__':
    pass
