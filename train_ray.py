"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import argparse

from lavis.common.config import Config
from lavis.common.utils import now

from lavis.runners import RunnerRay
import ray
from ray.train.torch import TorchTrainer
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # replace some settings in the used config
    parser.add_argument("--replace_cfg", nargs="+", help="replace some settings in the used config", default=None)
    # parser.add_argument("--job_id", default=None, help="job id")
    # python train.py --cfg-path configs/cont_train.yaml \
    # --replace-cfg run_cfg.seed=1 run_cfg.local_rank=0 --job-id 1

    args = parser.parse_args()

    return args

def run_training(config, job_id):
    # Instantiate your RunnerRay with all necessary arguments.
    runner = RunnerRay(config, job_id)
    # Optionally, log the starting configuration
    print("Starting training with RunnerRay...")
    runner.train()

def train_func(train_loop_config):

    args = train_loop_config["args"]
    cfg = Config(args)

    job_id = now() if cfg.run_cfg.job_id is None else cfg.run_cfg.job_id

    run_training(cfg, job_id)

def main():

    args = parse_args()
    args.cfg_path = Path(os.getenv("PWD")) / args.cfg_path
    cfg = Config(args)

    assert cfg.run_cfg.job_id is not None, "job_id must be provided in the config file."

    scaling_config = ray.train.ScalingConfig(num_workers=cfg.run_cfg.ray_num_gpus, 
                                             use_gpu=True,
                                             resources_per_worker={cfg.run_cfg.ray_gpu_type: 1} if cfg.run_cfg.ray_gpu_type != "any" else None,
                                             )

    # Resolve storage path to an absolute file URI to satisfy PyArrow's expectations.
    base_output_dir = (Path(os.getenv("REPO_DIR")) / "lavis"/ cfg.run_cfg.output_dir).resolve()
    storage_path = (base_output_dir / cfg.run_cfg.job_id / "ray").as_uri()

    # Configure the trainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"args": args},  # you can pass additional config options here
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            storage_path=storage_path,
            failure_config=ray.train.FailureConfig(max_failures=10),
        ),

    )

    # Run the training job.
    results = trainer.fit()
    print("Training results:", results)


if __name__ == "__main__":
    main()
