"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)


import lavis.tasks as tasks
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.runners.runner_base import RunnerBase, worker_init_fn

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset

import ray
import random
import numpy as np
import torch.backends.cudnn as cudnn


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


@registry.register_runner("runner_ray")
class RunnerRay(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    # def __init__(self, cfg, task, model, datasets, job_id):
    def __init__(self, cfg, job_id):

        self.config = cfg
        self.job_id = job_id


        # init_distributed_mode(cfg.run_cfg)

        setup_seeds(cfg)

        # set after init_distributed_mode() to only log on master.
        setup_logger()

        cfg.pretty_print()
        
        # Here read your custom config file and pass it to t

        # with torch.cuda.amp.autocast(dtype=torch.float32):
        task = tasks.setup_task(cfg)
        datasets = task.build_datasets(cfg)
        model = task.build_model(cfg)

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    # self._wrapped_model = DDP(
                    #     self._model, device_ids=[self.config.run_cfg.gpu], find_unused_parameters=True
                    # )
                    self._wrapped_model = ray.train.torch.prepare_model(self._model,
                                                                        parallel_strategy = "ddp",
                                                                        parallel_strategy_kwargs= {"find_unused_parameters": True})
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def resume(self):
        return self.config.run_cfg.get("resume", False)


    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        if not self.evaluate_only and self.resume:
            # check for the latest checkpoint i.e checkpoint_latest.pth
            # if not found, check if there are checkpoint_*.pth files
            last_ckpt = Path(self.output_dir) / "checkpoint_latest.pth"
            if last_ckpt.exists():
                ckpts = [last_ckpt]
            else:
                # ckpts are of from checkpoint_*.pth, i.e checkpoint_5.pth, checkpoint_10.pth, ... etc we pick the last one
                ckpts = sorted(Path(self.output_dir).glob("checkpoint_*.pth"), key=lambda x: int(x.stem.split('_')[-1]))
            if len(ckpts) > 0:
                print(f"found ckpts: {ckpts}")
                print(f"picked ckpt: {ckpts[-1]}")
                self._load_checkpoint(str(ckpts[-1]))

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(split_name=split_name, cur_epoch=cur_epoch)

            self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(dataset, wds.DataPipeline):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                    worker_init_fn=worker_init_fn,
                )

                loader = ray.train.torch.prepare_data_loader(loader)

                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i]) for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

