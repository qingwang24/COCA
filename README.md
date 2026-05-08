# COCA

This repository hosts the open-source project **COCA**.

GitHub: [https://github.com/qingwang24/COCA](https://github.com/qingwang24/COCA)

## Project Structure

```text
COCA/
├── README.md
├── code/              # Source code for cold start and COCA training.
│   ├── cold_start/    # Cold-start data processing, SFT, and self-play code.
│   ├── coca_train/    # COCA training and iterative GRPO training code.
│   └── readme.md      # Quick pointers to key training scripts.
└── prompts/           # Prompt templates and examples for COCA and evaluation.
    ├── README.md
    ├── coca_prompts.md
    ├── gpt4o_evaluation_prompt.md
    └── gpt4o_output_examples.md
```

## Current Status

- `code/`: contains the current implementation for cold-start training and COCA iterative training.
- `prompts/`: contains prompt templates and output examples used by COCA and GPT-4o evaluation.

## Code

The main code entry points are summarized in [`code/readme.md`](code/readme.md):

- Iterative reasoning: `code/cold_start/runner/vllm/self_play.py`
- Cold-start SFT: `code/cold_start/runner/vllm/train.py`
- COCA training: `code/coca_train/off_policy_GRPO/grpo_iter_new.py`

## Prompts

- [`prompts/README.md`](prompts/README.md): overview of prompt files.
- [`prompts/gpt4o_evaluation_prompt.md`](prompts/gpt4o_evaluation_prompt.md): system prompt for using GPT-4o as an evaluator.
- [`prompts/gpt4o_output_examples.md`](prompts/gpt4o_output_examples.md): JSON output examples for GPT-4o evaluation.
- [`prompts/coca_prompts.md`](prompts/coca_prompts.md): critic and actor prompts used in the COCA method.

## Notes

Runtime outputs, logs, checkpoints, and local cache files are intentionally excluded from version control.
