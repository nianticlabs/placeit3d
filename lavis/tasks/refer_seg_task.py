"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import h5py
import os
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.models.placewizard_model.seg_loss import get_iou
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm
from datetime import datetime

@registry.register_task("placeit3d_seg")
class PlaceIt3DSegTask(BaseTask):

    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()
        
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")
        
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt
        )

    def valid_step(self, model, samples):
        result = model.predict_seg(
            samples=samples,
            answer_list=None,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        
        pred_spmask = result['placement_masks'][-1].squeeze()
        
        result = dict(
            scan_id=samples["scan_ids"][0],
            asset_id=samples["asset_ids"][0],
            num_points=samples["superpoints"].shape[0],
            num_superpoints=samples["superpoints"].max().cpu().numpy()+1,
            pred_rots= result["rotation_angles_logits"][0].cpu().numpy() if "rotation_angles_logits" in result else np.array([]),
            pred_sp=pred_spmask.cpu().numpy(),
            description=samples["text_input"][0]
        )
        
        return [{"result": result}]
    
    def after_evaluation(self, val_result, split_name, epoch):
        num_samples = len(val_result)

        save_path =  os.path.join(registry.get_path("result_dir"), f"{split_name}_predictions.h5")     
        
        with h5py.File(save_path, "w") as hf:
            str_dt = h5py.string_dtype(encoding="utf-8")  # String dtype for HDF5
            float_vlen_dt = h5py.vlen_dtype(float)  # Variable-length float dtype

            scan_ids = hf.create_dataset("scan_ids", (num_samples,), dtype=str_dt)
            asset_ids = hf.create_dataset("asset_ids", (num_samples,), dtype=str_dt)
            num_points = hf.create_dataset("num_points", (num_samples,), dtype=int)
            num_superpoints = hf.create_dataset("num_superpoints", (num_samples,), dtype=int)
            pred_sp = hf.create_dataset("pred_sp", (num_samples,), dtype=float_vlen_dt)
            pred_rots = hf.create_dataset("pred_rots", (num_samples,), dtype=float_vlen_dt, compression="gzip")
            descriptions = hf.create_dataset("descriptions", (num_samples,), dtype=str_dt)
            
            logger.info("Exporting the results to HDF5 file...")
            for i, result in tqdm(enumerate(val_result)):
                scan_ids[i] = result["result"]["scan_id"]
                asset_ids[i] = result["result"]["asset_id"]
                
                num_points[i] = result["result"]["num_points"]
                num_superpoints[i] = result["result"]["num_superpoints"]
                
                pred_rots[i] = result["result"]["pred_rots"].flatten()
                pred_sp[i] = result["result"]["pred_sp"].flatten()
                descriptions[i] = result["result"]["description"]

        print(f"Saving the results to {save_path}")