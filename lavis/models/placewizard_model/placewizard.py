import logging
import os
import contextlib
import torch
import torch.nn as nn
import ipdb
import torch.nn.functional as F
import gorilla
from loguru import logger
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.dist_utils import download_cached_file
from lavis.models.placewizard_model.Qformer import BertConfig, BertLMHeadModel
from lavis.models.placewizard_model.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.placewizard_model.placement_decoder import PlacementDecoder
from lavis.models.placewizard_model.point_extractor import PointExtractor
from lavis.models.placewizard_model.seg_loss import Criterion
from lavis.common.utils import is_url


@registry.register_model("placewizard")
class PlaceWizard(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        apply_lemmatizer=False,
        point_encoder_cfg=None,
        mask_decoder_cfg=None,
        seg_criterion_cfg=None,
        pred_confidence=0.5,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_seg() result with lemmas.
        """
        super().__init__()

        self._build_scene_encoder(point_encoder_cfg)
        self._build_asset_encoder()
        self._build_placement_decoder(mask_decoder_cfg)

        assert point_encoder_cfg["media"] * 2 == mask_decoder_cfg["media"]

        # Encode (x, y, z) → feature space
        self.sp_xyz_encoder = nn.Linear(3, point_encoder_cfg["media"])
        # For fusion of the encoded pc features with the positional embeddings
        self.feature_fusion = nn.Sequential(
            nn.Linear(mask_decoder_cfg["media"],
                        mask_decoder_cfg["media"]),
            nn.ReLU(inplace=True),
            nn.Linear(mask_decoder_cfg["media"], mask_decoder_cfg["media"])
        )

        self._build_qformer(input_feature_dim=mask_decoder_cfg["media"], num_query_tokens=num_query_token)
        self._build_llm(t5_model)

        # Freeze the llm
        # Except the input and the output embeddings
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self._build_proj_layers(apply_lemmatizer, mask_decoder_cfg)

        #################################################
        # Construct the criterion
        #################################################
        self.criterion = Criterion(**seg_criterion_cfg)
        self.pred_confidence = pred_confidence

        # Finally make sure the trainable parameters are set correctly
        self._set_trainable_parameters()

    def _build_scene_encoder(self, point_encoder_cfg):
        #################################################
        # Construct the 3D scene point encoder
        #################################################
        self.encoder = PointExtractor(**point_encoder_cfg)

        # Load the pretrained checkpoint
        repo_dir = os.environ.get("REPO_DIR", "")
        ckpt_rel = "checkpoints/spformer_encoder_uniform_superpoints.pth"
        encoder_checkpoint_path = os.path.join(repo_dir, ckpt_rel) if repo_dir else ckpt_rel
        logger.info(
            f"Loading 3D scene encoder checkpoint from {encoder_checkpoint_path}")
        gorilla.load_checkpoint(
            self.encoder, encoder_checkpoint_path, strict=False, map_location='cpu')

    def _build_asset_encoder(self):
        #################################################
        # Construct the appropriate asset feature projector
        #################################################
        logger.info(
            f"The asset feature is represented as a pointbert extracted feature shape [768] (maxpooled from point bert).")
        self.asset_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048)
        )

        self.asset_size_proj = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048)
        )

        self.embedd_asset_feature = True

    def _build_placement_decoder(self, placement_decoder_cfg):
        #################################################
        # Construct the appropiate Mask Decoder
        #################################################
        # 32 for the xyz embedding and 32 for the encoded features
        placement_decoder_cfg["media"] = placement_decoder_cfg["media"] * 2
        logger.info(
            f"Injecting xyz coordinates to the scene features. The new media size is {placement_decoder_cfg['media']}.")

        self.placement_decoder = PlacementDecoder(**placement_decoder_cfg)

    def _build_qformer(self, input_feature_dim, num_query_tokens):
        #################################################
        # Construct the Query Former
        #################################################
        self.pc_adapter = nn.Linear(input_feature_dim, 1408)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_tokens, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

    def _build_llm(self, t5_model):
        #################################################
        # Construct the T5 Tokenizer and Model
        #################################################
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config)

        # Add all the possible special tokens that we are goind to use so that we have forward compatibility
        # with different experiments, also all the experiments from now on will add these special tokens in the
        # same order, so that each special token id will the same
        n_new_tokens = 0
        n_new_tokens += self.t5_tokenizer.add_tokens("[SEG]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[ROT]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[ANC]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[ASSET_START]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[ASSET_END]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[SCENE_START]")
        n_new_tokens += self.t5_tokenizer.add_tokens("[SCENE_END]")

        self.seg_token_idx = self.t5_tokenizer(
            "[SEG]", add_special_tokens=False).input_ids[0]
        self.rot_token_idx = self.t5_tokenizer(
            "[ROT]", add_special_tokens=False).input_ids[0]
        self.anchor_token_idx = self.t5_tokenizer(
            "[ANC]", add_special_tokens=False).input_ids[0]
        
        # the below tokens are only put for compatibility with previous ablations
        # which didn't pan out. we are not using them.
        self.asset_start_idx = self.t5_tokenizer(
            "[ASSET_START]", add_special_tokens=False).input_ids[0]
        self.asset_end_idx = self.t5_tokenizer(
            "[ASSET_END]", add_special_tokens=False).input_ids[0]
        self.scene_start_idx = self.t5_tokenizer(
            "[SCENE_START]", add_special_tokens=False).input_ids[0]
        self.scene_end_idx = self.t5_tokenizer(
            "[SCENE_END]", add_special_tokens=False).input_ids[0]

        # Update the size of the input and the output embeddings of the llm
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        # Intialize the added tokens with the mean of the embeddings of the tokens that are already in the model
        assert n_new_tokens == 7, "The number of new tokens added should be 5 as we are not yet doing pretraining step."
        if n_new_tokens > 0:
            input_embeddings = self.t5_model.get_input_embeddings().weight.data
            output_embeddings = self.t5_model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-n_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-n_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-n_new_tokens:] = input_embeddings_avg
            output_embeddings[-n_new_tokens:] = output_embeddings_avg

    def _build_proj_layers(self, apply_lemmatizer, mask_decoder_cfg):
        # Setup the Qformer projection layer to the T5 model
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        in_dim = self.t5_model.config.hidden_size
        out_dim = mask_decoder_cfg["d_text"]

        # Setting up the SEG/ROT/ANC token projection layers
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

        self.rot_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

        self.anc_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
            
    def _set_trainable_parameters(self):
        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.placement_decoder.train()
        for param in self.placement_decoder.parameters():
            param.requires_grad = True

        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.rot_hidden_fcs.train()
        for param in self.rot_hidden_fcs.parameters():
            param.requires_grad = True

        self.anc_hidden_fcs.train()
        for param in self.anc_hidden_fcs.parameters():
            param.requires_grad = True

        self.sp_xyz_encoder.train()
        self.feature_fusion.train()

        # Setting the parameters of the added network to be trainable (just in case)
        for param in self.sp_xyz_encoder.parameters():
            param.requires_grad = True
        for param in self.feature_fusion.parameters():
            param.requires_grad = True

        # make sure that the qformer model is trainable
        for param in self.Qformer.parameters():
            assert param.requires_grad

        for param in self.asset_proj.parameters():
            param.requires_grad = True
        for param in self.asset_size_proj.parameters():
            param.requires_grad = True

        for param in self.pc_adapter.parameters():
            param.requires_grad = True

        for param in self.t5_proj.parameters():
            param.requires_grad = True

    def get_scene_features(self, samples):
        sp_feats, _ = self.encoder(samples)  # (B*M, D)

        sp_xyz_coords = samples["superpoints_centers"]  # (B*M, 3)
        sp_xyz_encoded = self.sp_xyz_encoder(sp_xyz_coords)  # (B*M, 32)
        sp_fused_features = torch.cat(
            [sp_feats, sp_xyz_encoded], dim=-1)  # (B*M, 32*2)
        sp_fused_features = self.feature_fusion(
            sp_fused_features)  # (B*M, 32*2)
        samples["sp_feats"] = sp_fused_features

        x_feat, batch_mask = self.placement_decoder.get_batches(
            sp_fused_features, samples["batch_offsets"])
        pc_embeds = x_feat
        pc_embeds = self.pc_adapter(pc_embeds)
        image_atts = (~batch_mask).long()

        return samples["sp_feats"], None, pc_embeds, image_atts, batch_mask

    def forward(self, samples):
        batch_size = samples["batch_offsets"].shape[0] - 1

        with self.maybe_autocast():
            answer = samples["answer"]
            text_input = samples["text_input"]
            n_answers = samples["n_answers"]

            #################################################
            # Getting the 3D scene features
            #################################################
            _, p_dense_features, pc_embeds, image_atts, batch_mask = self.get_scene_features(
                samples)

        #################################################
        # Compressing the scene features using Q-Former
        #################################################
        query_tokens = self.query_tokens.expand(
            pc_embeds.shape[0], -1, -1)  # 768 #2, 32, 768
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # This carries the 32 scene token features
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)

        #################################################
        # Adding the asset features when applicable
        #################################################
        asset_features = self.asset_proj(
            samples["asset_encoding_features"]).unsqueeze(1)

        asset_feature_size_emb = self.asset_size_proj(
            samples["asset_encoding_sizes"]
        ).unsqueeze(1)
        
        asset_features = asset_features + asset_feature_size_emb
        
        samples["asset_features"] = asset_features
        inputs_t5 = torch.cat([inputs_t5, asset_features], dim=1)

        atts_t5 = torch.ones(
            inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        #################################################
        # Constructing the input to the T5 model
        #################################################
        if self.prompt:
            text_input = [self.prompt.format(question)
                          for question in text_input]
        else:
            text_input = text_input

        dtype = torch.float32

        with torch.cuda.amp.autocast(dtype=dtype):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=400,
                return_tensors="pt",
            ).to(pc_embeds.device)
            output_tokens = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(pc_embeds.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, n in enumerate(n_answers):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n

            batch_input_tokens_input_ids = torch.stack(
                batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(
                batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat(
                [batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(
                batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            #################################################
            # Forward the LLM
            #################################################
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True
            )
            llm_out = outputs["decoder_hidden_states"][-1]

            seg_token_index = targets == self.seg_token_idx
            seq_out = llm_out[seg_token_index]
            text_features = self.text_hidden_fcs(seq_out).unsqueeze(1)
            samples["loc_token_features"] = text_features

            rot_token_index = targets == self.rot_token_idx
            rot_out = llm_out[rot_token_index]
            rot_features = self.rot_hidden_fcs(rot_out).unsqueeze(1)
            samples["rot_token_features"] = rot_features

            anchor_token_index = targets == self.anchor_token_idx
            anchor_out = llm_out[anchor_token_index]
            anchor_features = self.anc_hidden_fcs(anchor_out).unsqueeze(1)
            samples["anc_token_feature"] = anchor_features

        #################################################
        # Placement Decoder
        #################################################
        out = self.placement_decoder(**samples)

        seg_loss, log_vars = self.criterion(pred=out,
                                            gt_spmasks=samples["gt_spmasks"],
                                            gt_anchor_spmasks=samples["anchor_gt_spmasks"],
                                            gt_rotation_angles=samples["rot_gt_spmasks"],) 

        loss = outputs.loss
        loss = loss + seg_loss
        return {"loss": loss, "log_vars": log_vars}

    def predict_seg(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        batch_size = samples["batch_offsets"].shape[0] - 1

        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            text_input = samples["text_input"]

            #################################################
            # Getting the 3D scene features
            #################################################
            _, p_dense_features, pc_embeds, image_atts, batch_mask = self.get_scene_features(
                samples)

        #################################################
        # Compressing the scene features using Q-Former
        #################################################
        query_tokens = self.query_tokens.expand(
            pc_embeds.shape[0], -1, -1)  # 768 #2, 32, 768
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # This carries the 32 scene token features
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)

        #################################################
        # Adding the asset features when applicable
        #################################################
        asset_features = self.asset_proj(
            samples["asset_encoding_features"]).unsqueeze(1)

        asset_feature_size_emb = self.asset_size_proj(
            samples["asset_encoding_sizes"].unsqueeze(1)
        )
        asset_features = asset_features + asset_feature_size_emb
        samples["asset_features"] = asset_features
        inputs_t5 = torch.cat([inputs_t5, asset_features], dim=1)

        atts_t5 = torch.ones(
            inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(text_input, str):
            text_input = [text_input]

        prompt = self.prompt

        if prompt:
            text_input = [prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(
                input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_hidden_states=True
                # for description, also use repetition penalty = 1.5
            )

            llm_out = outputs['decoder_hidden_states'][-1][-1]
            
            seg_mask = outputs["sequences"][:, 1:] == self.seg_token_idx
            seg_out = llm_out[seg_mask].mean(axis=0, keepdim=True)

            if seg_out.shape[0] == 0:
                # only allow batch size = 1
                h_s = self.t5_model.config.hidden_size
                seg_out = torch.zeros((1, h_s)).cuda()

            text_features = self.text_hidden_fcs(seg_out).unsqueeze(1)
            samples["loc_token_features"] = text_features
            
            rot_token_index = outputs["sequences"][:, 1:] == self.rot_token_idx
            rot_out = llm_out[rot_token_index]
            
            if rot_out.shape[0] == 0:
                # only allow batch size = 1
                h_s = self.t5_model.config.hidden_size
                rot_out = torch.zeros((1, h_s)).cuda()
            
            rot_features = self.rot_hidden_fcs(rot_out).unsqueeze(1)
            samples["rot_token_features"] = rot_features

            anchor_token_index = outputs["sequences"][:, 1:] == self.anchor_token_idx
            anchor_out = llm_out[anchor_token_index]
            
            if anchor_out.shape[0] == 0:
                # only allow batch size = 1
                h_s = self.t5_model.config.hidden_size
                anchor_out = torch.zeros((1, h_s)).cuda()
            
            anchor_features = self.anc_hidden_fcs(anchor_out).unsqueeze(1)
            samples["anc_token_feature"] = anchor_features

        #################################################
        # Placement Decoder
        #################################################
        result = self.placement_decoder(**samples)

        return result

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        mask_decoder_cfg = cfg.get("mask_decoder_cfg")
        seg_criterion_cfg = cfg.get("seg_criterion_cfg")
        pred_confidence = cfg.get("pred_confidence")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        prompt = cfg.get("prompt", "")
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            apply_lemmatizer=apply_lemmatizer,
            point_encoder_cfg=point_encoder_cfg,
            mask_decoder_cfg=mask_decoder_cfg,
            seg_criterion_cfg=seg_criterion_cfg,
            pred_confidence=pred_confidence
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(
            1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(
            mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
