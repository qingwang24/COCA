#!/usr/bin/env python
# coding=utf-8

import logging
import os
my_env = os.environ.copy()  

# my_env["PATH"] = "/home4/intern/ycpan4/miniconda3/envs/mcts_flash/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:" + my_env["PATH"] # 防止镜像没有module命令无法load gcc
# os.environ['CUDA_HOME'] = '/opt/lib/cuda-12.1'
# os.environ.update(my_env)
os.environ['TRITON_CACHE_DIR'] = '/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/sft/.triton/autotune'
os.environ['CXX'] = 'g++'
os.environ["WANDB_DISABLED"] = 'true' # 防止镜像没有module命令无法load gcc
os.environ["MAX_JOBS"] = '8'
os.environ['TORCH_EXTENSIONS_DIR'] = "/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/sft/.cache/v5"
os.environ.update(my_env)


import sys
# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
import numpy as np
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import libs.configs.default as cfg
from libs.data.critique import Dataset, DataCollator, SeqSampler
from libs.utils.comm import distributed, get_rank, get_world_size

import torch
import transformers
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.models.critique import CritiqueModel
from pretrainer import PreTrainer as Trainer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Qwen2ForTokenClassification,
    AutoProcessor,
    Pix2StructVisionModel,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from safetensors.torch import load_file
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def get_num_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    logger.info(f"trainable params: {int(trainable_params / 1e6)}M || all params: {int(all_param / 1e6)}M || trainable: {100 * trainable_params / all_param}%")


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # init datasets, collator and batch_sampler
    # cfg.train_seq_length=2048
    batch_sampler = SeqSampler(cfg.train_dataset_meta_path, get_world_size(), get_rank(), cfg.train_batch_size, cfg.train_seq_length, epoch=0)
    train_dataset = Dataset(cfg.train_dataset_meta_path, cfg.train_pixel_meta_path, seq_length=cfg.train_seq_length, tokenizer=cfg.qwen_tokenizer)
    data_collator = DataCollator(cfg.qwen_tokenizer.pad_token_id)
    
    # 加载Qwen2VL
    model = CritiqueModel.from_pretrained(cfg.qwen_checkpoint_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    for param in model.visual.parameters():
        param.requires_grad = False
    # model = model.to(torch.bfloat16)
    
    use_lora = False
    if use_lora:
        # 配置LoRA
        logger.info(f"Init Lora...")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=64,  # Lora 秩, 控制Lora更新的维度
            lora_alpha=16,  # Lora alaph，控制Lora微调强度
            lora_dropout=0.05,  # Dropout 比例
            bias="none",
        )
        # 获取LoRA模型
        peft_model = get_peft_model(model, config)
        model = peft_model
    else:
        model = model

    # 打开最后一层
    # for name, param in model.named_parameters():
    #     if '.layers.27.' in name or '.layers.26.' in name or '.layers.25.' in name:
    #         param.requires_grad = True
    
    get_num_params(model)
    logger.info(f"model dtype: {model.dtype}")
    if cfg.load_only_parameters and os.path.exists(cfg.load_only_parameters_path):
        state_dict = load_file(cfg.load_only_parameters_path)
        model.load_state_dict(state_dict, strict=True)
        logger.info('Load params from {}'.format(cfg.load_only_parameters_path))
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=data_collator,
        batch_sampler=batch_sampler
    )
    
    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()