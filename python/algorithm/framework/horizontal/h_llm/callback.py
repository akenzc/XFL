# Copyright 2022 The XFL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import inspect
from pathlib import Path
from typing import Union
from copy import deepcopy

import torch
import torch.nn as nn
import transformers

from accelerate import (
    dispatch_model, infer_auto_device_map
)
from accelerate.utils import get_balanced_memory
from accelerate.hooks import (
    AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
)
from transformers import (
    TrainerCallback, TrainingArguments, TrainerState, TrainerControl
)
from peft import PeftModel, PromptLearningConfig
from peft.utils import (
    get_peft_model_state_dict, set_peft_model_state_dict
)

from algorithm.core.horizontal.aggregation.api import (
    get_aggregation_root_inst, get_aggregation_leaf_inst
)
from common.utils.logger import logger
from service.fed_config import FedConfig


class AssistTrainerCallback(TrainerCallback):
    def __init__(self,
                 agg_steps: int,
                 sec_conf: dict,
                 root_id: str,
                 leaf_ids: list[str],
                 init_params: bool = False,
                 peft_type: str = "LORA"):
        super().__init__()
        self.agg_steps = agg_steps
        self.agg_steps_list = []
        if not (0 < agg_steps <= 1):
            raise ValueError("agg_steps must be between 0 and 1.")
        self.agg_inst = get_aggregation_root_inst(sec_conf, root_id, leaf_ids)
        self.init_params = init_params
        self.peft_type = peft_type

    def on_train_begin(self,
                       args: TrainingArguments,
                       state: TrainerState,
                       control: TrainerControl,
                       model: Union[transformers.PreTrainedModel, torch.nn.Module],
                       **kwargs):
        if self.init_params:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    torch.nn.init.uniform_(m.weight.data)
                    
        args.logging_steps = state.max_steps + 1
        args.save_steps = state.max_steps + 1
        
        adapters_weights = get_adapter_state_dict(model, self.peft_type)
        self.agg_inst.broadcast(adapters_weights)

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    model: Union[transformers.PreTrainedModel, torch.nn.Module],
                    tokenizer,
                    **kwargs):
        
        if not self.agg_steps_list:
            i = self.agg_steps
            while i < 1:
                self.agg_steps_list.append(round(i, 4))
                i += self.agg_steps
            self.agg_steps_list.append(1)
            self.steps_list = [math.ceil(i * state.max_steps)
                               for i in self.agg_steps_list]
            if len(self.steps_list) != len(set(self.steps_list)):
                raise ValueError("agg_steps is too small or overlapping, try a larger one.")
            logger.info(f"Aggergate model by steps: {self.agg_steps_list}")

        if state.global_step in self.steps_list:
            idx = self.steps_list.index(state.global_step)
            factor = 1

            adapters_weights = get_adapter_state_dict(model, self.peft_type)
            logger.info(f"gather and aggregating..., global_step={state.global_step}")
            new_adapters_weights = self.agg_inst.aggregate(
                parameters=adapters_weights, parameters_weight=factor)
            
            set_adapter_state_dict(model, self.peft_type, new_adapters_weights)
            logger.info(f"broadcasting..., global_step={state.global_step}")
            self.agg_inst.broadcast(new_adapters_weights)
            
            if args.output_dir and args.save_strategy != 'no':
                model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}")
            control.should_log = True


class LabelTrainerCallback(TrainerCallback):
    def __init__(self,
                 agg_steps: Union[float, int],
                 sec_conf: dict,
                 root_id: str,
                 leaf_ids: list[str],
                 init_params: bool = False,
                 peft_type: str = "LORA"):
        super().__init__()
        self.agg_steps = agg_steps
        self.agg_steps_list = []
        if not (0 < agg_steps <= 1):
            raise ValueError("agg_steps must be between 0 and 1.")
        self.is_standalone = False if FedConfig.get_assist_trainer() else True
        if not self.is_standalone:
            self.agg_inst = get_aggregation_leaf_inst(sec_conf, root_id, leaf_ids)
        self.init_params = init_params
        self.peft_type = peft_type

    def on_train_begin(self,
                       args: TrainingArguments,
                       state: TrainerState,
                       control: TrainerControl,
                       model: Union[transformers.PreTrainedModel, torch.nn.Module],
                       train_dataloader,
                       **kwargs):
        if self.init_params:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    torch.nn.init.uniform_(m.weight.data)

        args.logging_steps = state.max_steps + 1
        args.save_steps = state.max_steps + 1
        
        if not self.is_standalone:
            new_adapters_weights = self.agg_inst.download()
            set_adapter_state_dict(model, self.peft_type, new_adapters_weights)

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    model: Union[transformers.PreTrainedModel, torch.nn.Module],
                    **kwargs):
            
        if not self.agg_steps_list:
            i = self.agg_steps
            while i < 1:
                self.agg_steps_list.append(round(i, 4))
                i += self.agg_steps
            self.agg_steps_list.append(1)
            self.steps_list = [math.ceil(i * state.max_steps)
                               for i in self.agg_steps_list]
            if len(self.steps_list) != len(set(self.steps_list)):
                raise ValueError(f"agg_steps is too small or overlapping.")
            logger.info(f"Aggergate model by steps: {self.agg_steps_list}")

        if state.global_step in self.steps_list:
            idx = self.steps_list.index(state.global_step)
            if not self.is_standalone:
                factor = 1
                
                adapters_weights = get_adapter_state_dict(model, self.peft_type)
                
                # DP-PEFT (Differential Privacy) noise injection
                # In DP scenarios, noise proportional to the sensitivity of LoRA weights 
                # should be added right here to adapters_weights before sending.
                # Assuming noise_multiplier is given via env or config, here we add a standard DP clip-and-noise mechanism
                # if DP config is present. For now, we simulate the DP noise injection.
                dp_noise_multiplier = float(FedConfig.get_env("DP_NOISE_MULTIPLIER", 0.0))
                dp_clip_norm = float(FedConfig.get_env("DP_CLIP_NORM", 1.0))
                
                if dp_noise_multiplier > 0.0:
                    logger.info(f"Applying DP-LoRA noise: multiplier={dp_noise_multiplier}, clip_norm={dp_clip_norm}")
                    # Calculate global L2 norm of the update
                    total_norm = 0.0
                    for k, v in adapters_weights.items():
                        total_norm += torch.sum(v ** 2).item()
                    total_norm = math.sqrt(total_norm)
                    
                    clip_coef = dp_clip_norm / (total_norm + 1e-6)
                    clip_coef = min(1.0, clip_coef)
                    
                    for k, v in adapters_weights.items():
                        # Clip
                        clipped_v = v * clip_coef
                        # Add Gaussian Noise
                        noise = torch.normal(
                            mean=0.0, 
                            std=dp_noise_multiplier * dp_clip_norm, 
                            size=v.shape, 
                            device=v.device, 
                            dtype=v.dtype
                        )
                        adapters_weights[k] = clipped_v + noise
                
                logger.info(f"uploading..., global_step={state.global_step}")
                self.agg_inst.upload(adapters_weights, factor)
                logger.info(f"downloading..., global_step={state.global_step}")
                new_adapters_weights = self.agg_inst.download()
                set_adapter_state_dict(model, self.peft_type, new_adapters_weights)

            if args.output_dir and args.save_strategy != 'no':
                model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}")

            control.should_log = True


def get_adapter_state_dict(model: PeftModel, peft_type: str, **kwargs):
    adapter_name = model.active_adapter
    adapters_weights = get_peft_model_state_dict(
        model, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
    )

    for k, v in adapters_weights.items():
        adapters_weights[k] = deepcopy(v).to('cpu')
    return adapters_weights


def set_adapter_state_dict(model: PeftModel, peft_type: str, adapters_weights: dict, **kwargs):
    adapter_name = model.active_adapter
    # load the weights into the model
    set_peft_model_state_dict(model, adapters_weights, adapter_name=adapter_name)
    if (
        (getattr(model, "hf_device_map", None) is not None)
        and (len(set(model.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
        and len(model.peft_config) == 1
    ):
        device_map = kwargs.get("device_map", "auto")
        max_memory = kwargs.get("max_memory", None)
        offload_dir = kwargs.get("offload_folder", None)
        offload_index = kwargs.get("offload_index", None)

        dispatch_model_kwargs = {}
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = model._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        dispatch_model(
            model,
            device_map=device_map,
            offload_dir=offload_dir,
            **dispatch_model_kwargs,
        )
        hook = AlignDevicesHook(io_same_device=True)
        if isinstance(model.peft_config[adapter_name], PromptLearningConfig):
            remove_hook_from_submodules(model.prompt_encoder)
        add_hook_to_module(model.get_base_model(), hook)