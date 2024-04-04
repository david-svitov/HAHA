import argparse
import datetime
import os
import sys
from enum import Enum

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.general import instantiate_from_config
from torch.utils.tensorboard import SummaryWriter


class TrainingStage(Enum):
    INIT_TEXTURE = "INIT_TEXTURE"
    OPTIMIZE_GAUSSIANS = "OPTIMIZE_GAUSSIANS"
    FINETUNE_TEXTURE = "FINETUNE_TEXTURE"
    OPTIMIZE_OPACITY = "OPTIMIZE_OPACITY"
    FINETUNE_POSE = "FINETUNE_POSE"


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="Paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--test_mode",
        action='store_true',
        help="Only evaluate metrics from the checkpoint",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="Load pretrained weights from the checkpoint",
    )
    return parser


def generate_path_to_logs(config, opt, sequence_name):
    experiment_name = opt.base[0].split('/')[-1].split('.yaml')[0]
    time = datetime.datetime.now()
    run_name = sequence_name + time.strftime(f"-%Y_%m-%d_%H-%M")
    log_dir = os.path.join(config.logdir, experiment_name, run_name)
    return log_dir


def create_test_datasets(config):
    test_dataset = instantiate_from_config(config.test_dataloader)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.val_dataloader.batch_size,
        num_workers=config.val_dataloader.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    print("Test samples:", len(test_dataloader))
    return test_dataset, test_dataloader


def create_train_val_datasets(config):
    train_dataset = instantiate_from_config(config.train_dataloader)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_dataloader.batch_size,
        num_workers=config.train_dataloader.num_workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    print("Training samples:", len(train_dataloader))

    test_dataset = instantiate_from_config(config.val_dataloader)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.val_dataloader.batch_size,
        num_workers=config.val_dataloader.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    print("Validation samples:", len(test_dataloader))

    return train_dataset, train_dataloader, test_dataloader


def setup_tensorboard_logger(runner, config, opt, sequence_name):
    log_dir = generate_path_to_logs(config, opt, sequence_name)
    if opt.test_mode:
        log_dir += '-test'
    os.makedirs(log_dir, exist_ok=False)
    runner.logger = SummaryWriter(log_dir)


def setup_callbacks(runner, config):
    callbacks = []
    for callback_config in config.callbacks.values():
        callbacks.append(instantiate_from_config(callback_config))
    runner.set_callbacks(callbacks)


def main(args):
    parser = get_parser()

    opt, unknown = parser.parse_known_args(args)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    runner = instantiate_from_config(config.runner)
    runner.to(runner.device)

    setup_callbacks(runner, config)

    if opt.pretrained:
        runner.load_checkpoint(opt.pretrained)

    if opt.test_mode:
        test_dataset, test_dataloader = create_test_datasets(config)
        setup_tensorboard_logger(runner, config, opt, test_dataset.sequence_name)

        runner.initialize_optimizable_pose(test_dataset)
        runner.fit_pose(test_dataloader)
        runner.test(test_dataloader)
    else:
        train_dataset, train_dataloader, val_dataloader = create_train_val_datasets(config)
        setup_tensorboard_logger(runner, config, opt, train_dataset.sequence_name)

        runner.initialize_optimizable_pose(train_dataset)
        runner.fit(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main(sys.argv)
