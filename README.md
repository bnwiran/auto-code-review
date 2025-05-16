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