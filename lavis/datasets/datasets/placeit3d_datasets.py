"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import glob
import h5py
import torch
import random
import math
import pathlib
import os
import numpy as np
import os.path as osp
import pointgroup_ops
import torch_scatter

from loguru import logger
from typing import Tuple
from lavis.datasets.datasets.base_dataset import BaseDataset


class PlaceIt3DDataset(BaseDataset):
    def __init__(self, text_processor, pts_root, ann_paths, our_cfg=None):
        super().__init__(text_processor, pts_root, ann_paths)

        self.hdf5_file = None
        self.mode = 4

        if "train" in ann_paths[0]:
            self.prefix = "train"
            self.training = True
        elif "val" in ann_paths[0]:
            self.prefix = "val"
            self.training = False
        elif "test" in ann_paths[0]:
            self.prefix = "val"
            self.training = False
        self.with_label = True

        if ann_paths[0].endswith(".h5"):
            repo_dir = os.environ.get("REPO_DIR", "")
            h5_path = ann_paths[0]
            if repo_dir and not osp.isabs(h5_path):
                h5_path = osp.join(repo_dir, h5_path)
            self.hdf5_file = h5_path
            h5 = self.read_h5()
            entry_paths = self.get_entry_paths()
            self.annotation = [i for i in range(len(entry_paths))]

        # Normalize pts_root by prefixing REPO_DIR when provided and pts_root is relative
        repo_dir = os.environ.get("REPO_DIR", "")
        if repo_dir and not osp.isabs(self.pts_root):
            self.pts_root = osp.join(repo_dir, self.pts_root)

        self.sp_filenames = self.get_sp_filenames()
        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.aug = True

        # Get the superpoints type
        self.uniform_superpoints_root = osp.join(self.pts_root, "superpoints")
        self.asset_pointbert_features_root = osp.join(
            self.pts_root, "pointbert_embeddings")

        # Read the asset sizes json
        json_path = pathlib.Path(self.pts_root, 'object_sizes.json')
        if not json_path.is_file():
            raise FileNotFoundError(f"The JSON file {json_path} does not exist or is not a file.")
        with open(json_path, 'r') as f:
            self.asset_sizes = json.load(f)

    def get_sp_filenames(self):
        filenames = glob.glob(
            osp.join(self.pts_root, 'scannetv2', self.prefix, '*' + '_refer.pth'))
        assert len(filenames) > 0, f'Empty dataset. Check the path: {self.pts_root}'
        filenames = sorted(filenames)
        return filenames

    def read_h5(self):
        logger.info(f"Loading placement data from {self.hdf5_file}...")
        hdf5_file = self.hdf5_file
        file_path = pathlib.Path(hdf5_file)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"The file {file_path} does not exist or is not a file.")
        try:
            h5 = h5py.File(file_path, 'r')
        except Exception as e:
            raise RuntimeError(f"Failed to load HDF5 file: {e}")

        return h5

    def get_entry_paths(self):
        h5 = self.read_h5()

        entry_paths = []
        for asset_name, asset_group in h5.items():
            if asset_name != "metadata" and isinstance(asset_group, h5py.Group):
                for entry_name, entry_group in asset_group.items():
                    if isinstance(entry_group, h5py.Group) and len(entry_group) > 0:
                        entry_paths.append(f"{asset_name}/{entry_name}")

        return entry_paths

    def read_placement_data(self):
        # logger.info(f"Loading placement data from {self.hdf5_file}...")
        hdf5_file = self.hdf5_file
        self.h5 = self.read_h5()

        if self.h5 is None:
            raise RuntimeError("HDF5 file was not loaded properly.")

        if "metadata" not in self.h5:
            raise ValueError(f"Metadata group not found in {hdf5_file}.")

        if "relation_types" not in self.h5["metadata"]:
            raise ValueError(
                f"Dataset 'relation_types' not found in metadata group of {hdf5_file}.")

        self.relation_types = [
            rt.decode('utf-8') if isinstance(rt, bytes) else rt
            for rt in self.h5["metadata"]["relation_types"][()]
        ]
        # print(f"Loaded {len(self.relation_types)} relation types.")

        self.entry_paths = self.get_entry_paths()

        if not self.entry_paths:
            raise ValueError(f"No valid entry groups found in {hdf5_file}.")

        # print(f"Initialized dataset with {len(self.entry_paths)} entries.")

    def get_asset_size_from_json(self, asset_id):
        asset_key = self.find_matching_asset_key(asset_id)
        if asset_key and asset_key in self.asset_sizes:
            return self.asset_sizes[asset_key]["size"]
        return None
    
    def find_matching_asset_key(self, asset_id):
        asset_suffix = asset_id.split('_')[-1]  # Extract unique asset identifier
        for key in self.asset_sizes.keys():
            if asset_suffix in key:
                return key
        return None
    
    def get_example_data(self, idx):
        """
        Fetch and decode a sample from the dataset, including the number of constraints,
        asset id, and scene id.
        """
        entry_path = self.entry_paths[idx]
        entry_group = self.h5[entry_path]

        # Extract asset name from the entry path.
        asset_id = entry_path.split('/')[0]  # e.g., "scene0191_00_asset1"

        # Extract scene id from asset_id.
        # Assuming the asset id is in the format "sceneID_assetName", where sceneID is the first two tokens.
        parts = asset_id.split('_')
        if len(parts) >= 3:
            scene_id = '_'.join(parts[:2])
        else:
            scene_id = asset_id  # fallback if the naming convention is different

        asset_group = self.h5[asset_id]

        if "asset_size" in asset_group:
            asset_size = asset_group["asset_size"][:].tolist()
        else:
            asset_size = self.get_asset_size_from_json(asset_id)

        relationships = entry_group["relationships"][:]
        prompt = entry_group["prompt"][()]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        rotations = entry_group["rotations"][:]
        intersection_sizes = entry_group["intersection_sizes"][:]
        pointcloud_ids = entry_group["pointcloud_ids"][:]
        valid_rotations = entry_group["valid_rotations"][:]

        # Decode relationships and count constraints.
        decoded_relationships = []
        # Initialize counter for non-"plausible" relationships.
        num_constraints = 1

        for rel in relationships:
            code = rel[0]
            relation_name = self.relation_types[code]
            if relation_name != "plausible":
                num_constraints += 1
            anchors = [int(anchor) for anchor in rel[1:] if anchor != -1]
            decoded_relationships.append({
                "relation": relation_name,
                "anchors": anchors
            })

        decoded_valid_rotations = []
        for mask in valid_rotations:
            valid_rots = [rotations[i]
                          for i in range(len(rotations)) if (mask & (1 << i))]
            decoded_valid_rotations.append(valid_rots)

        return {
            "prompt": prompt,
            "relationships": decoded_relationships,
            "intersection_sizes": intersection_sizes.tolist(),
            "pointcloud_ids": pointcloud_ids.tolist(),
            "valid_rotations": decoded_valid_rotations,
            "asset_size": asset_size,
            "num_constraints": num_constraints,
            "asset_id": asset_id,
            "scene_id": scene_id,
        }

    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * 50
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(
            instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, np.eye(3)

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        # This augmentation method has potential issues and only rot should be used
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m), m

    def get_cropped_inst_label(
        self, instance_label: np.ndarray, valid_idxs: np.ndarray
    ) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def get_ref_mask(self, instance_label, superpoint, object_id):
        ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(
            ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask

    def __len__(self):
        return len(self.annotation)

    def get_anchor_mask(self, instance_label, superpoint, anchor_ids):
        ref_lbl = torch.zeros(instance_label.shape[0])
        instance_label = torch.from_numpy(instance_label)

        for object_id in anchor_ids:
            ref_lbl += (
                instance_label == object_id
            ).float()  # Convert boolean mask to float before accumulation

        # Ensure binary mask (instead of > 0.5)
        ref_lbl = (ref_lbl > 0).float()

        gt_spmask = torch_scatter.scatter_mean(ref_lbl, superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl

        return gt_pmask, gt_spmask

    def get_input_description(self, data_dict, asset_size_cm):
        description = data_dict["prompt"]

        return self.text_processor(description)

    def get_asset_encoding(self, asset_size, asset_size_raw=None, asset_id=None):
        """ Get the asset encoding based on the asset_encoding type
        args:
            asset_id: str, the asset id
            asset_size: str, the asset size, ex. "10 10 10 cm"
        """
        assert asset_id is not None, "Asset id is required for pointbert encoding"
        assert asset_size_raw is not None, "Asset size raw is required to be used with the pointbert encoding"

        # Load the asset features
        asset_feature = torch.load(
            osp.join(self.asset_pointbert_features_root, f"{asset_id}.pt"))

        asset_size = torch.tensor(asset_size_raw)

        return asset_size, asset_feature

    def create_rotation_gt(self, valid_rotations, gt_pmask, pos_inds):
        """
        Create a binary tensor indicating valid rotations for each sample.

        Args:
            data_dict (dict): Contains "valid_rotations", a list of lists with valid rotation angles.
            gt_pmask (list or tensor): Ground truth mask determining the number of samples.
            pos_inds (list or tensor): Indices mapping each sample to its corresponding valid rotations.

        Returns:
            torch.Tensor: A (num_samples, 8) tensor where valid rotations are marked as 1.0.
        """

        # Define the mapping from angles to indices
        angle_to_index = {0: 0, 45: 1, 90: 2,
                          135: 3, 180: 4, 225: 5, 270: 6, 315: 7}

        # Extract valid rotations (angles) from the data dictionary
        num_samples = len(gt_pmask)  # Determine number of samples

        # Initialize the rotation ground truth tensor with 0
        rot_gt = torch.full((num_samples, 8), 0, dtype=torch.float32)

        # Convert angles to indices
        valid_indices = [torch.tensor(
            [angle_to_index[angle] for angle in angles]) for angles in valid_rotations]

        # Convert the repeat counts to a tensor
        repeat_counts = torch.tensor(
            [len(vr) for vr in valid_indices], dtype=torch.long)

        # Expand pos_inds to match the number of valid rotations per sample
        pos_idx_expanded = torch.tensor(
            pos_inds, dtype=torch.long).repeat_interleave(repeat_counts)

        # Flatten and concatenate valid rotation indices
        rot_indices = torch.cat(valid_indices).long()

        # Assign 1.0 to valid rotation positions
        rot_gt[pos_idx_expanded, rot_indices] = 1.0

        return rot_gt

    def __getitem__(self, index: int) -> Tuple:
        example_id = self.annotation[index]
        data_dict = self.get_example_data(example_id)

        ann_id = example_id
        scan_id = data_dict["scene_id"]
        asset_id = data_dict.get("asset_id", None)
        if asset_id is not None:
            asset_id = asset_id.split("_")[-1]
        raw_asset_id = asset_id

        # The actual size in meters as list of floats
        asset_size_raw = data_dict.get("asset_size", None)
        if asset_size_raw is not None:
            asset_size = " ".join([str(int(x*100))
                                  for x in asset_size_raw]) + " cm"

        anchor_ids = []
        for el in data_dict["relationships"]:
            for anchor_id in el["anchors"]:
                anchor_ids.append(anchor_id)

        #####################################################
        # Get the input text prompt and the text answer
        #####################################################
        description = self.get_input_description(data_dict, asset_size_cm=asset_size)
        question_template = random.choice(self.short_question_list)
        question = question_template.format(description=description)
        answers = [random.choice(self.answer_list)]

        #####################################################
        # Get the asset features to be used in training
        #####################################################
        asset_encoding_size, asset_encoding_feature = self.get_asset_encoding(
            asset_size, asset_id=raw_asset_id, asset_size_raw=asset_size_raw)

        #####################################################
        # load point cloud
        #####################################################
        for fn in self.sp_filenames:
            if scan_id in fn:
                sp_filename = fn
                break
        data = self.load(sp_filename)

        data = self.transform_test(*data)
        xyz, xyz_middle, rgb, _, _, instance_label, _ = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()

        #####################################################
        # Get the superpoints to be used
        #####################################################
        superpoint = np.load(
            osp.join(self.uniform_superpoints_root, f"{scan_id}_sp_1024_euclid.npy"))
        superpoint = torch.from_numpy(superpoint)

        # Get the center of each superpoint cluster
        superpoint_centers = torch_scatter.scatter_mean(
            coord_float, superpoint, dim=0)

        #####################################################
        # Get the gt coarse (superpoint) and fine (full point cloud) grained masks
        #####################################################
        pos_inds = data_dict["pointcloud_ids"]

        gt_pmask = torch.zeros(coord.shape[0]).float()
        gt_pmask[pos_inds] = 1.0

        gt_spmask = torch_scatter.scatter_mean(gt_pmask, superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.25).float()

        #####################################################
        # Get the anchor mask
        #####################################################
        anchor_gt_pmask, anchor_gt_spmask = self.get_anchor_mask(
            instance_label, superpoint, anchor_ids)

        #####################################################
        # Get the rotation gt
        #####################################################
        rot_gt_pmask = self.create_rotation_gt(
            data_dict["valid_rotations"], gt_pmask, pos_inds)

        # Now create the gt for the superpoints
        rot_gt_spmask = torch_scatter.scatter_mean(
            rot_gt_pmask, superpoint, dim=0)
        rot_gt_spmask = (rot_gt_spmask > 0.25).float()

        # Find the max number of points inside a superpoint
        max_points_in_a_superpoint = torch_scatter.scatter_sum(
            torch.ones_like(superpoint), superpoint, dim=0).max().item()
        
        ret_data = {
            "ann_id": ann_id,
            "scan_id": scan_id,
            "coord": coord,
            "coord_float": coord_float,
            "feat": feat,
            "superpoint": superpoint,
            "asset_id": asset_id,
            "gt_pmask": gt_pmask,
            "gt_spmask": gt_spmask,
            "text_input": question,
            "answers": answers,
            "asset_encoding_size": asset_encoding_size,
            "asset_encoding_feature": asset_encoding_feature,
            "superpoint_centers": superpoint_centers,
            "max_points_in_a_superpoint": max_points_in_a_superpoint,
            "rot_gt_spmask": rot_gt_spmask,
            "anchor_gt_spmask": anchor_gt_spmask,
            "asset_ids": asset_id,
        }
        
        if self.prefix == "val":
            ret_data["rot_gt_pmask"] = rot_gt_pmask
            ret_data["anchor_gt_pmask"] = anchor_gt_pmask

        return ret_data

    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, asset_ids, gt_pmasks, gt_spmasks, text_inputs, answers_list, asset_encoding_sizes, asset_encoding_features, superpoints_centers, object_ids, rot_gt_spmasks, anchor_gt_spmasks = ([
        ], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        
        val_rot_gt_pmasks = []
        val_anchor_gt_pmasks = []

        batch_offsets = [0]
        dense_batch_offsets = [0]  # start = 0
        n_answers = []
        raw_superpoints = []
        superpoint_bias = 0
        dense_point_bias = 0

        max_points_in_a_superpoint_list = []

        for i, data in enumerate(batch):
            ann_id = data["ann_id"]
            scan_id = data["scan_id"]
            coord = data["coord"]
            coord_float = data["coord_float"]
            feat = data["feat"]
            src_superpoint = data["superpoint"]
            asset_id = data["asset_id"]
            gt_pmask = data["gt_pmask"]
            gt_spmask = data["gt_spmask"]
            text_input = data["text_input"]
            answers = data["answers"]
            asset_encoding_size = data["asset_encoding_size"]
            asset_encoding_feature = data["asset_encoding_feature"]
            src_superpoint_centers = data["superpoint_centers"]
            max_points_in_a_superpoint = data["max_points_in_a_superpoint"]
            rot_gt_spmask = data["rot_gt_spmask"]
            anchor_gt_spmask = data["anchor_gt_spmask"]
            asset_id = data["asset_ids"]

            n_points = coord.shape[0]
            assert n_points == gt_pmask.shape[0]

            # Without the superpoint bias
            raw_superpoints.append(src_superpoint)

            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            dense_point_bias = dense_point_bias + n_points
            dense_batch_offsets.append(dense_point_bias)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(
                torch.cat(
                    [torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1)
            )
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            object_ids.append(-1)
            max_points_in_a_superpoint_list.append(max_points_in_a_superpoint)

            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            rot_gt_spmasks.append(rot_gt_spmask)
            anchor_gt_spmasks.append(anchor_gt_spmask)

            answers_list.extend(answers)
            text_inputs.append(text_input)
            n_answers.append(len(answers))
            superpoints_centers.append(src_superpoint_centers)
            asset_ids.append(asset_id)

            if asset_encoding_size is not None:
                asset_encoding_sizes.append(asset_encoding_size)
            if asset_encoding_feature is not None:
                asset_encoding_features.append(asset_encoding_feature)
            if "rot_gt_pmask" in data:
                val_rot_gt_pmasks.append(data["rot_gt_pmask"])
            if "anchor_gt_pmask" in data:
                val_anchor_gt_pmasks.append(data["anchor_gt_pmask"])

        batch_offsets = torch.tensor(
            batch_offsets, dtype=torch.int)  # int [B+1]
        dense_batch_offsets = torch.tensor(
            dense_batch_offsets, dtype=torch.int)  # int [B+1]
        # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]

        raw_superpoints = torch.cat(raw_superpoints, 0).long()
        superpoints_centers = torch.cat(
            superpoints_centers, 0)  # float [B*N, 3]
        feats = torch.cat((feats, coords_float), dim=1)
        
        # voxelize
        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            coords, len(batch), self.mode)

        ret_dict = {
            "ann_ids": ann_ids,
            "scan_ids": scan_ids,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "feats": feats,
            "superpoints": superpoints,
            "batch_offsets": batch_offsets,
            "object_ids": object_ids,
            "gt_pmasks": gt_pmasks,
            "gt_spmasks": gt_spmasks,
            "answer": answers_list,
            "text_input": text_inputs,
            "n_answers": torch.LongTensor(n_answers),
            "superpoints_centers": superpoints_centers,
            "dense_batch_offsets": dense_batch_offsets,
            "raw_superpoints": raw_superpoints,
            "max_points_in_a_superpoint": max_points_in_a_superpoint_list,
            "rot_gt_spmasks": rot_gt_spmasks,
            "anchor_gt_spmasks": anchor_gt_spmasks,
            "asset_ids": asset_ids,
        }

        if len(asset_encoding_sizes) > 0:
            asset_encoding_sizes = torch.stack(asset_encoding_sizes)
            ret_dict["asset_encoding_sizes"] = asset_encoding_sizes

        if len(asset_encoding_features) > 0:
            asset_encoding_features = torch.cat(asset_encoding_features, dim=0)
            ret_dict["asset_encoding_features"] = asset_encoding_features
        
        if len(val_rot_gt_pmasks) > 0:
            val_rot_gt_pmasks = torch.stack(val_rot_gt_pmasks)
            ret_dict["val_rot_gt_pmasks"] = val_rot_gt_pmasks
        
        if len(val_anchor_gt_pmasks) > 0:
            val_anchor_gt_pmasks = torch.stack(val_anchor_gt_pmasks)
            ret_dict["val_anchor_gt_pmasks"] = val_anchor_gt_pmasks

        return ret_dict

    def __len__(self):
        return len(self.annotation)


QUESTION_LIST = [
    "Identify the region in the given 3D scene and the asset according to the description: {description}.",
    "Given the 3D scene and the asset, determine the region based on the description: {description}.",
    "Respond with the segmentation mask of the exact region: {description}.",
]

ANSWER_LIST = [
    "It is [SEG] [ROT] [ANC].",
    "Sure, [SEG] [ROT] [ANC].",
    "Sure, it is [SEG] [ROT] [ANC].",
    "Sure, the segmentation result is [SEG] [ROT] [ANC].",
    "[SEG] [ROT] [ANC].",
]
