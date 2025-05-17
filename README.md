# Auto Code Review

## Description

## Installation

### Prerequisites
- Python version required Python 3.11.x

### Steps
#### Clone the repository
```sh
git clone https://github.com/bnwiran/auto-code-review.git
cd auto-code-review
```

#### Create virtual environment
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

#### Install dependencies
```sh
pip install -r requirements.txt
```

## Setup
To prepare the Code Review/Comment Generation dataset, run the following command:
```sh
python prepare_dataset.py \
--ds_src <code_reviewer_comment_generation_source> \
--ds_dest <code_reviewer_destination> \
--ds_name <dataset_name>
```
This command will create the code_reviewer HF dataset.

### Fine-tuning
To fine-tune a model, run the following command:
```sh
python fine_tune/fine_tune.py --dataset_name <dataset_name>
```

### Optional Arguments

| Argument       | Type     | Default Value                | Description                                                                 |
|----------------|----------|------------------------------|-----------------------------------------------------------------------------|
| `bf16`         | `bool`   | `False`                      | Enables bf16 training.                                                     |
| `fp16`         | `bool`   | `False`                      | Enables fp16 training.                                                     |
| `dataset`      | `str`    | `/AI/datasets/code_reviewer` | The preference dataset to use.                                             |
| `lora_alpha`   | `int`    | `16`                         | LoRA alpha parameter.                                                      |
| `lora_dropout` | `float`  | `0.0`                        | LoRA dropout rate.                                                         |
| `lora_r`       | `int`    | `8`                          | LoRA rank parameter.                                                       |
| `model`        | `str`    | `meta-llama/Llama-3.2-1B`    | The model to train from the Hugging Face hub.                              |
| `model_dtype`  | `str`    | `float32`                    | The model dtype to use (e.g., float16, bfloat16).                          |
| `output_dir`   | `str`    | `./output`                   | The output directory for model predictions and checkpoints.                |