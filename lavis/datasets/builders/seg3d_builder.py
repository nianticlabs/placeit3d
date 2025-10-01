"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.placeit3d_datasets import PlaceIt3DDataset

@registry.register_builder("placeit3d")
class ThreeDReferBuilder(BaseDatasetBuilder):
    train_dataset_cls = PlaceIt3DDataset
    eval_dataset_cls = PlaceIt3DDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults.yaml"}
