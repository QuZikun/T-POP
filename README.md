# T-POP: Test-Time Personalization with Online Preference Feedback

[![arXiv](https://img.shields.io/badge/arXiv-2509.24696-b31b1b.svg)](https://arxiv.org/abs/2509.24696) [![Paper](https://img.shields.io/badge/OpenReview-T--POP-blue)](https://openreview.net/forum?id=Y3tTG28OO9)

## Overview

T-POP performs test-time personalization of a frozen LLM by training a lightweight reward model online with preference feedback collected during decoding. At each generation step, an exploration/exploitation arm pair is combined via a Multi-Armed Bandit objective to steer the model toward a target attribute (e.g. *creative*, *concise*, *verbose*).

The core algorithm is implemented in `model.py` and `imple_armo.py`. We test the code on Qwen, Llama series of LLMs; it may run for other series, but we have not tested them.

## Setup

```bash
git clone https://github.com/QuZikun/T-POP.git
cd T-POP
conda create -y --name tpop python=3.10
conda activate tpop
pip install -r requirements.txt
```

## Datasets

We preprocessed the following four datasets as our benchmark, which are placed in the `data` folder.

```
data
  ├── HelpSteer_train.json
  ├── UltraFeedback_truthful_qa.json
  ├── UltraFeedback_ultrachat.json
  └── personal_preference_eval_preference_data.json
```

You can also use your own dataset by defining a json file of the following format:

```
{
    "query": "What is the best mobile phone brand currently?"
},
...
```

## Usage

```bash
bash script/run_armo.sh
```

Before running, edit the variables at the top of `script/run_armo.sh` — at minimum `OPENAI_API_KEY`, `DATA_FILE`, `ATTRIBUTE`, and the device assignments (`LLM_DEVICE`, `REWARD_MODEL_DEVICE`, `EMBEDDING_MODEL_DEVICE`).

Or call the script directly:

```bash
python imple_armo.py \
   --llm_path meta-llama/Llama-3.1-8B-Instruct \
   --embedding_model_path Qwen/Qwen3-Embedding-0.6B \
   --openai_api_key "<your_api_key>" \
   --preference_model_name openai/gpt-4o-2024-08-06 \
   --preference_api_base_url https://openrouter.ai/api/v1 \
   --data_file data/personal_preference_eval_preference_data.json \
   --attribute creative \
   --train_samples 100 \
   --reward_weight 1.0 \
   --output_dir results/example \
   --normalize_sentence_embeddings
```

Please refer to `imple_armo.py` for documentation on what each configuration does.

## Evaluation

We implement two types of evaluation, GPT win rate and RM score.

### GPT win rate

After producing T-POP responses (`ours_file`) and a baseline file with the same queries (`baseline_file`), compute the pairwise win rate judged by GPT:

```bash
python winrate_evaluator.py \
   --ours_file     results/example/responses/responses_armo_rw_1.0.jsonl \
   --baseline_file path/to/baseline.jsonl \
   --attribute     creative \
   --openai_api_key "<your_api_key>" \
   --judge_model   openai/gpt-4o-2024-08-06 \
   --api_base_url  https://openrouter.ai/api/v1
```

### RM score

We also provide an implementation of RM evals with [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1). RM scoring is performed automatically at the end of `imple_armo.py` and written to `<output_dir>/result.txt`. Quick lookup:

```bash
grep "ArmoRM Score:" <output_dir>/result.txt
```

## BibTex

If you find our paper / code helpful, please consider citing our work 📝 and starring this repository ⭐️!

```
@misc{qu2025tpoptesttimepersonalizationonline,
      title={T-POP: Test-Time Personalization with Online Preference Feedback},
      author={Zikun Qu and Min Zhang and Mingze Kong and Xiang Li and Zhiwei Shang and Zhiyong Wang and Yikun Ban and Shuang Qiu and Yao Shu and Zhongxiang Dai},
      year={2025},
      eprint={2509.24696},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24696},
}
```
