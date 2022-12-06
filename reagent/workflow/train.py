# ---------------------------------------------------------------------
# Implements custom pipeline for ReAgent
# Author Neeraj Jain
# --------------------------------------------------------------------

import argparse
import os
import pprint
import json
import time
import logging

import torch
from torch.backends import cudnn
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from prodict import Prodict

from reagent.core.tensorboardX import summary_writer_context
from reagent.training.dqn_trainer import DQNTrainer
from reagent.workflow.types import RewardOptions
from reagent.model_managers.discrete.mobile_dqn import MobileDQN
from reagent.workflow.types import ReaderOptions, ResourceOptions
from reagent.data.my_data_module import MyDataModule

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Read json config file
    with open(args.cfg, "r") as f:
        config = json.load(f)
        # convert it into dot notation
    cfg = Prodict.from_dict(config)

    mname = cfg.MODEL.name
    dname = cfg.DATASET.dataset
    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    final_output_dir = "".join(["output","/",dname,"/",mname,"/",cfg_name])

    logger.info(pprint.pformat(args))
    logger.info(json.dumps(cfg, indent=4, sort_keys=False))

    # cudnn related setting
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    writer = SummaryWriter()
    # logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

    # create model manager
    model_manager = MobileDQN()

    reward_options = RewardOptions()
    # Build trainer
    trainer_module = model_manager.build_trainer(
        cfg,
        use_gpu=1,
        reward_options=reward_options,
    )
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    warmstart_input_path = final_output_dir
    reader_options = ReaderOptions()

    resource_options = ResourceOptions()

    data_module = None
    num_epochs = cfg.TRAIN.epochs
    dset = cfg.DATASET.dataset
    train_path = cfg.DATASET.train
    eval_path = cfg.DATASET.valid
    test_path = cfg.DATASET.test
    bsize = cfg.TRAIN.batch_size
    workers = cfg.TRAIN.workers

    # Create data module
    data_module = MyDataModule(train_path, eval_path, test_path, bsize, workers)
    logger.info("Loading dataset {}".format(dset))
    start = time.time()
    data_module.setup()
    end = time.time()
    logger.info("Time Taken: {} Mins".format((end - start) / 60))
    # Train and validate the model
    with summary_writer_context(writer):
        train_output, lightning_trainer = model_manager.train(
            trainer_module,
            None,   # train_dset,
            None,   # eval_dset,
            None,
            data_module,
            num_epochs,
            reader_options,
            resource_options,
            checkpoint_path=warmstart_input_path,
        )

    return


if __name__ == "__main__":
    main()
