import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from .attention import SelfAttentionLayer, CrossAttentionLayer, FFN


class PlacementDecoder(nn.Module):
    def __init__(
        self,
        # decoder=None,
        d_text=512,
        num_layer=6,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        attn_mask=True,
        media=32,
        our_cfg=None
    ):
        super().__init__()
        
        self.num_layer = num_layer
        self.attn_mask = attn_mask

        self.input_proj = nn.Sequential(
            nn.Linear(media, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.x_placement_mask = nn.Sequential(
            nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        ##################################################
        ######        Projecting LLM Tokens         ######
        ##################################################
        self.loc_token_proj = nn.Linear(d_text, d_model)
        self.placement_out_norm = nn.LayerNorm(d_model)

        self.rot_token_proj = nn.Linear(d_text, d_model)

        self.anc_token_proj = nn.Linear(d_text, d_model)
        self.x_anchor_mask = nn.Sequential(
            nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.anchor_out_norm = nn.LayerNorm(d_model)

        ##################################################
        ######         Self/Cross Attention         ######
        ##################################################
        self.sa_layers = nn.ModuleList([])
        self.sa_ffn_layers = nn.ModuleList([])

        self.ca_layers = nn.ModuleList([])
        self.ca_ffn_layers = nn.ModuleList([])

        for i in range(num_layer):
            self.sa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.sa_ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
            self.ca_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.ca_ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))

        # Predicting the rotation angles
        self.sp_proj_rot = nn.Sequential(
            nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.rot_prime_proj = nn.Sequential(nn.Linear(
            d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(),)
        self.rotation_angles_head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 8))
            
        self.asset_proj = nn.Sequential(
            nn.Linear(2048, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sp_asset_proj_instance = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sp_asset_proj_placement = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sp_asset_proj_anchor = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sp_asset_proj_rot = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(
                max_len - cur_len, x.shape[1]).to(x.device)], dim=0)

            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask

    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool()  # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) ==
                                   attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def placement_prediction_head(self, query, mask_feats, batch_mask):
        query = self.placement_out_norm(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_masks, attn_masks

    def anchor_prediction_head(self, query, mask_feats, batch_mask):
        query = self.anchor_out_norm(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_masks, attn_masks

    def forward(self,
                sp_feats,
                batch_offsets,
                loc_token_features=None,
                rot_token_features=None,
                anc_token_feature=None,
                asset_features=None,
                **kwargs):
        B = len(batch_offsets) - 1
        x = sp_feats

        ##########################################################
        # Get instance features used in the cross-attention with the llm queries
        ##########################################################
        inst_feats = self.input_proj(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)    
        
        asset_feature_proj = self.asset_proj(asset_features)
        inst_feats = torch.cat([inst_feats, asset_feature_proj.repeat(1, inst_feats.size(1), 1)], dim=-1)
        inst_feats = self.sp_asset_proj_instance(inst_feats)
        
        ##########################################################
        # Project the llm queries
        ##########################################################
        loc_token_query = self.loc_token_proj(loc_token_features)
        if not self.training: # only during the inference
            loc_token_query = loc_token_query[:1, :, :] # Keep only the first token in case the llm predicts multiple placement tokens
        
        anc_token_query = self.anc_token_proj(anc_token_feature)
        if not self.training: # only during the inference
            anc_token_query = anc_token_query[:1, :, :] # Keep only the first token in case the llm predicts multiple anchor tokens
        
        rot_token_query = self.rot_token_proj(rot_token_features)
        if not self.training: # only during the inference
            rot_token_query = rot_token_query[:1, :, :] # Keep only the first token in case the llm predicts multiple rotation tokens
        ##########################################################
        # Get the mask features used in the prediction heads for the placement and the anchor
        ##########################################################
        placement_mask_feats = self.x_placement_mask(x)
        placement_mask_feats, _ = self.get_batches(
            placement_mask_feats, batch_offsets)
        
        placement_mask_feats = torch.cat([placement_mask_feats, asset_feature_proj.repeat(1, placement_mask_feats.size(1), 1)], dim=-1)
        placement_mask_feats = self.sp_asset_proj_placement(placement_mask_feats)
    
        placement_prediction_masks = []

        anchor_mask_feats = self.x_anchor_mask(x)
        anchor_mask_feats, _ = self.get_batches(
            anchor_mask_feats, batch_offsets)
            
        anchor_mask_feats = torch.cat([anchor_mask_feats, asset_feature_proj.repeat(1, anchor_mask_feats.size(1), 1)], dim=-1)
        anchor_mask_feats = self.sp_asset_proj_anchor(anchor_mask_feats)
            
        anchor_prediction_masks = []

        ##########################################################
        # 0-th prediction for the placement
        ##########################################################
        placement_pred_masks, loc_attn_masks = self.placement_prediction_head(
            loc_token_query, placement_mask_feats, batch_mask)
        placement_prediction_masks.append(placement_pred_masks)

        ##########################################################
        # 0-th prediction for the anchor
        ##########################################################
        anchor_pred_masks, anchor_attn_masks = self.anchor_prediction_head(
            anc_token_query, anchor_mask_feats, batch_mask)
        anchor_prediction_masks.append(anchor_pred_masks)

        ##########################################################
        # Construct the queries
        ##########################################################
        cur_idx = 0

        queries = loc_token_query
        loc_query_idx = cur_idx
        cur_idx += 1

        queries = torch.cat([queries, rot_token_query], dim=1)
        rot_query_idx = cur_idx
        cur_idx += 1

        queries = torch.cat([queries, anc_token_query], dim=1)
        anc_query_idx = cur_idx
        cur_idx += 1

        ##########################################################
        # multi-round
        ##########################################################
        for i in range(self.num_layer):
            # None means all the llm tokens can attend to each other
            queries = self.sa_layers[i](queries, None)
            queries = self.sa_ffn_layers[i](queries)

            # Now we need to construct the attn masks
            # The attn masks for the loc and the rot tokens are the same
            # but different from the attn masks for the anchor token
            attention_masks = loc_attn_masks  # [B, 1, n_sp]
            # Repeat the loc_attn_masks for the rot token
            attention_masks = torch.cat(
                [attention_masks, loc_attn_masks], dim=1)

            attention_masks = torch.cat(
                [attention_masks, anchor_attn_masks], dim=1)

            queries, _ = self.ca_layers[i](
                inst_feats, queries, batch_mask, attention_masks)
            queries = self.ca_ffn_layers[i](queries)

            # Do the placement prediction
            placement_query = queries[:, loc_query_idx, :].view(B, 1, -1)
            placement_pred_masks, _ = self.placement_prediction_head(
                placement_query, placement_mask_feats, batch_mask)
            _, loc_attn_masks = self.placement_prediction_head(
                placement_query, placement_mask_feats, batch_mask)
            placement_prediction_masks.append(placement_pred_masks)

            anchor_query = queries[:, anc_query_idx, :].view(B, 1, -1)
            anchor_pred_masks, _ = self.anchor_prediction_head(
                anchor_query, anchor_mask_feats, batch_mask)
            _, anchor_attn_masks = self.anchor_prediction_head(
                anchor_query, anchor_mask_feats, batch_mask)
            anchor_prediction_masks.append(anchor_pred_masks)

        # Rotation angles predictions per superpoint
        rot_query = queries[:, rot_query_idx, :].view(B, 1, -1)
        rot_query_prime = self.rot_prime_proj(rot_query)
        
        sp_feats_prime = self.sp_proj_rot(x)
        sp_feats_prime, _ = self.get_batches(sp_feats_prime, batch_offsets)
        
        sp_feats_prime = torch.cat([sp_feats_prime, asset_feature_proj.repeat(1, sp_feats_prime.size(1), 1)], dim=-1)
        sp_feats_prime = self.sp_asset_proj_rot(sp_feats_prime)
            
        sp_feats_combined = torch.cat(
            [rot_query_prime.repeat(1, sp_feats_prime.size(1), 1), sp_feats_prime], dim=-1)
        rotation_angles_logits = self.rotation_angles_head(
            sp_feats_combined)

        result = {
            'placement_masks':
            placement_pred_masks,
            'batch_mask':
            batch_mask,
            'aux_placement_masks': placement_prediction_masks[:-1]
        }

        result['rotation_angles_logits'] = rotation_angles_logits
        result['anchor_masks'] = anchor_pred_masks
        result['aux_anchor_masks'] = anchor_prediction_masks[:-1]

        return result
