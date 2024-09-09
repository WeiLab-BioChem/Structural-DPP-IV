import argparse
import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
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

model_ckpt = '../ckpt/model.ckpt'  # specify your checkpoint path here


def run_inference(args):
    pl.seed_everything(args.seed, workers=True)
    print(f"Seed set to: {args.seed}")

    data_module = SeqDataModule(args)
    data_module.setup('test')  # setup for testing data

    model = SeqLightningModule.load_from_checkpoint(model_ckpt, args=args)  # Load model from checkpoint

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    test_result = trainer.test(model=model, datamodule=data_module)

    metric_df = util_metric.print_results(test_result)  # Assuming you have a function to print metrics
    return metric_df


def use_dataset_for_test(config_dict, dataset_name, test_sub_set: str = None):
    config_dict['dataset_name'] = dataset_name
    config_dict['path_test_data'] = dataset_name
    config_dict['test_sub_set'] = test_sub_set
    config_dict['max_seq_len'] = 90
    assert config_dict['max_seq_len'] > 0


def start_single_test(data_type):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config_dict = load_config.load_default_args_dict(data_type)
    config_dict['gpus'] = [0]  # specify GPU for inference
    config_dict['batch_size'] = 32
    config_dict['lr'] = 0.00001
    config_dict['model'] = 'StructuralDPPIV'
    config_dict['log_dir'] = constant['path_log']
    use_dataset_for_test(config_dict, 'DPP-IV')

    # Add this line to ensure path_train_data exists during inference. Not used in inference.
    config_dict['path_train_data'] = config_dict['path_test_data']

    # Add these lines to ensure train_sub_set and valid_sub_set exist. Not used in inference.
    config_dict['train_sub_set'] = None
    config_dict['valid_sub_set'] = None

    config_dict['use_cooked_data'] = False
    args = argparse.Namespace(**config_dict)
    print('args', args)
    run_inference(args)



if __name__ == '__main__':
    start_time = time.time()
    start_single_test('StructuralDPPIV')
    end_time = time.time()
    print('testing time', (end_time - start_time) / 60, '(min)')
