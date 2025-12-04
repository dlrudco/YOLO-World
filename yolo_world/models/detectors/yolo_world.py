# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats, _ = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats,
                        txt_masks) = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, txt_masks

import transformers
import re
from llava.constants import IGNORE_INDEX
@MODELS.register_module()
class YOLLMDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 lmm=None,
                 lmm_max_token_length=512,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        self.lmm_max_token_length = lmm_max_token_length
        
        if lmm is not None:
            from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
            from llava.model.multimodal_projector.builder import vision_projector_with_pos_proj
            
            self.lmm = LlavaQwenForCausalLM.from_pretrained(lmm).half()
            # self.lmm = checkpoint_wrapper(self.lmm)
            self.lmm.requires_grad_(False)
            self.lmm.config.use_cache = False
            
            self.lmm_tokenizer = transformers.AutoTokenizer.from_pretrained(
                lmm,
                cache_dir=None, 
                model_max_length=self.lmm_max_token_length, 
                padding_side="right")
            if hasattr(self.lmm.model, 'mm_projector'):
                del self.lmm.model.mm_projector
            if hasattr(self.lmm.model, 'vision_tower'):
                del self.lmm.model.vision_tower
            
            self.lmm.config.tokenizer_padding_side = self.lmm_tokenizer.padding_side
            self.lmm.config.tokenizer_model_max_length = self.lmm_max_token_length
            
            resolution_list = [80, 40, 20]
            channel_list = [192, 384, 576]
            self.qformer = nn.ModuleList()
            self.k_mapper = nn.ModuleList()
            self.v_mapper = nn.ModuleList()
            self.qformer_queries = nn.ParameterList()
            for i in range(len(channel_list)):
                self.k_mapper.append(nn.Linear(channel_list[i],896))
                self.v_mapper.append(nn.Linear(channel_list[i],896))
                self.qformer.append(nn.MultiheadAttention(896, 8))
                self.qformer_queries.append(nn.Parameter(
                torch.zeros(1, 32, 896), requires_grad=True
                ))

            # yv, xv = torch.meshgrid([torch.range(0, 1, 1/self.feature_map_size), torch.range(0, 1, 1/self.feature_map_size)])
            # grid = torch.stack((xv, yv), 2).view(self.feature_map_size+1, self.feature_map_size+1, 2)
            # self.grid_box = torch.cat([grid[:-1, :-1], grid[1:, 1:]], dim=-1).flatten(0, 1)
        

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        if self.lmm is not None:
            breakpoint()
            input_ids = [
                torch.tensor(data_samples['input_id'], dtype=torch.long, device=img_feats[0].device) for data_samples in batch_data_samples['conversation']
            ]
            labels = [
                torch.tensor(data_samples['label'], dtype=torch.long, device=img_feats[0].device) for data_samples in batch_data_samples['conversation']
            ]
            
            if self.lmm_tokenizer.pad_token_id is None:
                self.lmm_tokenizer.pad_token_id = 0 
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.lmm_tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.lmm_max_token_length]
            labels = labels[:, :self.lmm_max_token_length]
            attention_mask=input_ids.ne(self.lmm_tokenizer.pad_token_id)
            lmm_imput_dict = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

            image_queries = []
            query_masks = []
                                                        
            for i in range(len(img_feats)):
                feature_map = img_feats[i]
                feature_map = feature_map.flatten(2).permute(2,0,1)  # b, hw, c
                k = self.k_mapper[i](feature_map)
                v = self.v_mapper[i](feature_map)
                q = self.qformer_queries[i].permute(1,0,2).expand(-1, k.shape[1], -1)
                feature_map  = self.qformer[i](q, k, v)[0]
                image_queries.append(feature_map.half())
                query_masks.append((torch.ones(feature_map.shape[:-1], device=feature_map.device, dtype=torch.bool)))

                        
            image_queries = torch.cat(image_queries, dim=0).split(dim=1, split_size=1)
            image_queries = [iq.squeeze(1) for iq in image_queries]
            query_masks = torch.cat(query_masks, dim=0).split(dim=1, split_size=1)
            query_masks = [qm.squeeze(1) for qm in query_masks]
            lmm_imput_dict['input_ids'] = lmm_imput_dict['input_ids'].to(img_feats[0].device)
            lmm_imput_dict['labels'] = lmm_imput_dict['labels'].to(img_feats[0].device)
            lmm_imput_dict['attention_mask'] = lmm_imput_dict['attention_mask'].to(img_feats[0].device)
            lmm_imput_dict['image_queries'] = image_queries
            lmm_imput_dict['query_masks'] = query_masks
                        
            self.lmm.eval() 
            with torch.autocast('cuda', enabled=True):
                loss_lmm = self.lmm.detection_forward(**lmm_imput_dict)
            losses['loss_lmm_image'] = loss_lmm.loss
        
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats,
                        txt_masks) = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, txt_masks


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
