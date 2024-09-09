import argparse
import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import load_config
from config.load_constant import constant
from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule
from util import util_metric

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
if root_path not in sys.path:
    sys.path.append(root_path)




def run(args):
    pl.seed_everything(args.seed, workers=True)
    print(f"Seed set to: {args.seed}")
    data_module = SeqDataModule(args)
    data_module.setup('fit')

    model = SeqLightningModule(args)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_SE_epoch:.2f},{val_SP_epoch:.2f},{val_F1_epoch:.2f},{val_AUC_epoch:.2f}',
        monitor='val_ACC_epoch', save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_F1_epoch", min_delta=0.01, patience=100, verbose=False, mode="max")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    metric_df = util_metric.print_results(test_result)
    return metric_df


def use_dataset(config_dict, dataset_name, train_sub_set: str = None, valid_sub_set: str = None,
                test_sub_set: str = None):
    config_dict['dataset_name'] = dataset_name
    config_dict['path_test_data'] = dataset_name
    config_dict['path_train_data'] = dataset_name
    config_dict['path_valid_data'] = dataset_name
    config_dict['max_seq_len'] = 90
    config_dict['train_sub_set'] = train_sub_set
    config_dict['valid_sub_set'] = valid_sub_set
    config_dict['test_sub_set'] = test_sub_set
    assert config_dict['max_seq_len'] > 0


def start_single_train(data_type):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config_dict = load_config.load_default_args_dict(data_type)
    config_dict['max_epochs'] = 150
    config_dict['gpus'] = [0]  # using which GPU to train
    config_dict['batch_size'] = 32
    config_dict['lr'] = 0.00001
    config_dict['model'] = 'StructuralDPPIV'
    config_dict['log_dir'] = constant['path_log']
    use_dataset(config_dict, 'DPP-IV')
    config_dict['use_cooked_data'] = False
    # config_dict['use_cooked_data'] = False  # when you first run the code, set this to False
    args = argparse.Namespace(**config_dict)
    print('args', args)
    run(args)


if __name__ == '__main__':
    start_time = time.time()
    start_single_train('StructuralDPPIV')
    end_time = time.time()
    print('training time', (end_time - start_time) / 60, '(min)')
