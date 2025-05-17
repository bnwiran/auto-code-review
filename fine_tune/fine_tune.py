import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig

from helper import compute_module_sizes

load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fine_tune")


@dataclass
class ScriptArguments:
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )

    dataset: Optional[str] = field(
        default="/AI/datasets/code_reviewer",
        metadata={"help": "The preference dataset to use."},
    )

    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_r: Optional[int] = field(default=8)

    model: Optional[str] = field(
        default="TinyLlama/TinyLlama-1.1B-step-50K-105b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    model_dtype: Optional[str] = field(
        default="float32",
        metadata={"help": "The model dtype to use. E.g. float16, bfloat16, etc."},
    )

    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    per_device_train_batch_size: Optional[int] = field(default=1)
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    max_seq_length: Optional[int] = field(default=1024)


def _create_model(args: ScriptArguments):
    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        token=hf_access_token
    )

    logger.info(f"Model size: {compute_module_sizes(model)[''] / 1e9:.2f}GB")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        token=hf_access_token
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def _create_trainer(args, train_dataset: Dataset, model, tokenizer):
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        fp16=args.fp16,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        group_by_length=args.group_by_length,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    return trainer


def _create_gen_batches_train(args):
    def _gen_batches_train():
        dataset = load_dataset(args.dataset, streaming=True, split="train")

        for sample in iter(dataset):
            # Extract instruction and input from the sample
            formatted_prompt = _get_prompt(sample)
            yield {'text': formatted_prompt}

    return _gen_batches_train


def _get_prompt(sample):
    instruction = str(sample['instruction'])
    input_text = str(sample['text'])
    out_text = str(sample['target'])
    if input_text is None or input_text == "":
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            f"{str(out_text)}"
            f"<|eot_id|><|end_of_text|>"
        )
    else:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{str(out_text)}"
            f"<|eot_id|><|end_of_text|>"
        )

    return "".join(formatted_prompt)


def main(args):
    dataset = Dataset.from_generator(_create_gen_batches_train(args))
    model, tokenizer = _create_model(args)
    trainer = _create_trainer(args, dataset, model, tokenizer)

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
