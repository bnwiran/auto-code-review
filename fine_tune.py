import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")


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

    dataset_name: Optional[str] = field(
        default="/AI/datasets/code_reviewer",
        metadata={"help": "The preference dataset to use."},
    )

    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_r: Optional[int] = field(default=8)

    model_name: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    model_dtype: Optional[str] = field(
        default="float32",
        metadata={"help": "The model dtype to use. E.g. float16, bfloat16, etc."},
    )

    output_dir: str = field(
        default="./results_packing",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


def _create_model(args: ScriptArguments):
    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        torch_dtype=args.model_dtype,
        token=hf_access_token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        token=hf_access_token
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def _create_trainer(args, train_dataset, model, tokenizer):
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    return trainer


def main(args):
    train_dataset = load_dataset(args.dataset_name, split="train")
    model, tokenizer = _create_model(args)
    trainer = _create_trainer(args, train_dataset, model, tokenizer)

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
