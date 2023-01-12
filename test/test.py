from module.lightning_data_module import get_data_from_disk
from module.lightning_frame_module import SeqLightningModule

# def construct_StructDataset_Sequence(dataset_name: str, type_: str, sequences: list[str], labels, cubeBiased=False,
#                                      cubeBias=0.2, right_align=False, use_cooked_data=False,
#                                      max_seq_len: int = 90) -> int:
#     global MAX_SEQ_LEN
#     MAX_SEQ_LEN = max_seq_len
#     if use_cooked_data:
#         data_num = detect_max_index_pkl_in_path(os.path.join('..', 'cooked_data', dataset_name, type_)) + 1
#         return data_num
#
#     sequences_code = codePeptides(sequences)
#     index = 0
#     for i in tqdm(range(len(sequences))):
#         StructedData = construct_seq(sequences[i], cubeBiased=cubeBiased, cubeBias=cubeBias, right_align=right_align)
#         store_one(dataset_name, type_, data_=(sequences_code[index], torch.FloatTensor(StructedData)),
#                   label=labels[index], index=index)
#         index += 1
#     return index

model = SeqLightningModule.load_from_checkpoint("../main/log/StructuralDPPIV/version_13/checkpoints/123.ckpt")
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
    one_piece = get_data_from_disk("DPP-IV", "test", i)
    predict = model(one_piece)
    print(f"seq: {seq}, label: {labels[i]}, predict: {predict}")