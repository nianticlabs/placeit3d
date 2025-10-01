import h5py
from torch.utils.data import Dataset
import pathlib
import json
import tqdm
import argparse
import numpy as np
import torch
import json
from pathlib import Path
import trimesh


class PlacementDataset(Dataset):
    def __init__(self, hdf5_file,
                 asset_sizes_json=None):
        file_path = pathlib.Path(hdf5_file)
        if not file_path.is_file():
            raise FileNotFoundError(f"The file {file_path} does not exist or is not a file.")
        try:
            self.h5 = h5py.File(file_path, 'r')
        except Exception as e:
            raise RuntimeError(f"Failed to load HDF5 file: {e}")
        if self.h5 is None:
            raise RuntimeError("HDF5 file was not loaded properly.")

        if "metadata" not in self.h5:
            raise ValueError(f"Metadata group not found in {hdf5_file}.")

        if "relation_types" not in self.h5["metadata"]:
            raise ValueError(f"Dataset 'relation_types' not found in metadata group of {hdf5_file}.")

        self.relation_types = [
            rt.decode('utf-8') if isinstance(rt, bytes) else rt
            for rt in self.h5["metadata"]["relation_types"][()]
        ]
        print(f"Loaded {len(self.relation_types)} relation types.")

        self.entry_paths = []
        print("Loading dataset...")
        for asset_name, asset_group in self.h5.items():
            if asset_name != "metadata" and isinstance(asset_group, h5py.Group):
                for entry_name, entry_group in asset_group.items():
                    if isinstance(entry_group, h5py.Group) and len(entry_group) > 0:
                        self.entry_paths.append(f"{asset_name}/{entry_name}")

        if not self.entry_paths:
            raise ValueError(f"No valid entry groups found in {hdf5_file}.")

        print(f"Initialized dataset with {len(self.entry_paths)} entries.")


        if asset_sizes_json is None:
            json_path = pathlib.Path("data/object_sizes.json")
        else:
            json_path = pathlib.Path(asset_sizes_json)
        if not json_path.is_file():
            raise FileNotFoundError(f"The JSON file {asset_sizes_json} does not exist or is not a file.")

        with open(json_path, 'r') as f:
            self.asset_sizes = json.load(f)

    def __len__(self):
        return len(self.entry_paths)

    def __getitem__(self, idx):
        entry_path = self.entry_paths[idx]
        entry_group = self.h5[entry_path]

        asset_id = entry_path.split('/')[0]
        parts = asset_id.split('_')
        if len(parts) >= 3:
            scene_id = '_'.join(parts[:2])
        else:
            scene_id = asset_id

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

        decoded_relationships = []
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
            valid_rots = [rotations[i] for i in range(len(rotations)) if (mask & (1 << i))]
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


def load_and_align_mesh(mesh_path, alignment_path):
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices

    lines = open(alignment_path).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((vertices.shape[0], 4))
    pts[:, 0:3] = vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], vertices[:, 3:]],
                                            axis=1)
    vertices = aligned_mesh_vertices
    return vertices


def run(pred_path, superpoint_path, pcd_path, mesh_path):
    pred = []
    rots = [360 / 8. * i for i in range(8)]

    result_json = []
    for idx in tqdm.tqdm(range(len(test_dataset))):
        with h5py.File(pred_path, "r") as hf:
            # Read string datasets
            scan_ids = hf["scan_ids"][:].astype(str)  # Convert bytes to string

            # Read integer datasets
            num_superpoints = hf["num_superpoints"][:]

            pred_rots = hf["pred_rots"][idx]
            if pred_rots.shape[0] > 0:
                pred_rots = pred_rots.reshape(num_superpoints[idx], 8)
            else:
                pred_rots = np.zeros((num_superpoints[idx],
                                    8), dtype=np.float32)
                pred_rots[:, 0] = 1.0

            # Read variable-length float dataset
            pred_sp = hf["pred_sp"][idx]

        asset_height = test_dataset[idx]["asset_size"][2]

        superpoints = np.load(superpoint_path / f'{scan_ids[idx]}_sp_1024_euclid.npy')
        scene = scan_ids[idx]
        xyz = load_and_align_mesh(mesh_path=mesh_path / scene / f"{scene}_vh_clean_2.ply",
                                  alignment_path=mesh_path / scene / f"{scene}.txt")

        best_sp = pred_sp.argmax()
        best_rot = pred_rots[best_sp].argmax()

        top_sp = xyz[superpoints == best_sp].mean(0)
        nearest_point_idx = np.linalg.norm(top_sp - xyz, axis=1).argmin()
        nearest_point = xyz[nearest_point_idx]

        final_pred = nearest_point.copy()
        final_pred[2] += asset_height / 2

        pred.append((test_dataset[idx]['pointcloud_ids'] == nearest_point_idx).any())

        if idx % 10 == 0:
            print(pred_path, np.array(pred).mean())

        entry = dict(
            entry_id=int(idx),
            dataset_idx=int(-1),
            tx=float(final_pred[0]),
            ty=float(final_pred[1]),
            tz=float(final_pred[2]),
            rotation=float(rots[best_rot])
        )
        result_json.append(entry)

    save_path = pred_path.parent / (pred_path.stem + '.json')
    with open(save_path, 'w') as f:
        json.dump(result_json, f)

    print(save_path, np.array(pred).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate placement predictions')
    parser.add_argument('--pred_path', type=str, help='Path to the predictions file')
    parser.add_argument('--test_h5', type=str, help='Path to the test h5 file')
    parser.add_argument('--superpoint_path', type=str, help='Path to the superpoints')
    parser.add_argument('--pcd_path', type=str, help='Path to the point clouds')
    parser.add_argument('--mesh_path', type=str, help='Path to the meshes')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pred_path = Path(args.pred_path)
    superpoint_path = Path(args.superpoint_path)
    pcd_path = Path(args.pcd_path)
    mesh_path = Path(args.mesh_path)
    test_h5 = Path(args.test_h5)

    test_data = Path(test_h5)
    test_dataset = PlacementDataset(test_data)

    run(pred_path, superpoint_path, pcd_path, mesh_path)