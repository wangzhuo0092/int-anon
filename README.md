# Int-Rule

This repository provides an anonymized implementation for extracting internal judgment signals from LLM judges and recalibrating them with lightweight rule-based calibration.

The code supports two stages:

1. **Layer-wise internal signal extraction** from open-source LLM judges.
2. **Rule recalibration and evaluation** using calibration subsets.

The repository also includes scripts for direct judgment, reasoning-based judgment, and API-based log-probability extraction.

---

## Repository Structure

```text
Int-Rule-anon/
├── data/
│   └── main/
│       ├── BiGGen-Bench-human-eval.json
│       ├── chatbot_arena_digit.jsonl
│       ├── flask.json
│       ├── helpsteer.json
│       └── mt_bench_digit.jsonl
├── int-rule/
│   ├── get_api_outputs.py
│   ├── get_pointwise_internal.py
│   ├── main.py
│   ├── optimize_layer_weights.py
│   ├── utils_compare.py
│   └── utils.py
├── results/
│   ├── flask/
│   └── valid/
├── scripts/
│   ├── run_direct.sh
│   └── run_reasoning.sh
├── requirements.txt
└── README.md
```

---

## Environment Setup

We recommend using Python 3.10 or later.

```bash
conda create -n int-rule python=3.10
conda activate int-rule
pip install -r requirements.txt
```

If PyTorch installation depends on the CUDA version of your machine, install PyTorch following the official instructions first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

The main datasets are placed under:

```text
data/main/
```

The expected files are:

```text
BiGGen-Bench-human-eval.json
chatbot_arena_digit.jsonl
flask.json
helpsteer.json
mt_bench_digit.jsonl
```

The repository also contains example result files under:

```text
results/flask/
results/valid/flask/
```

These files can be used to run the calibration and evaluation pipeline without re-running the full extraction stage.

---

## Stage 1: Extract Layer-wise Internal Signals

For open-source LLM judges, use:

```bash
python int-rule/get_pointwise_internal.py \
  --model_name_or_path /path/to/model \
  --input_file data/main/flask.json \
  --save_dir results \
  --points 5 \
  --batch_size 4 \
  --max_new_tokens 20 \
  --temperature 0 \
  --dtype bfloat16
```

This script generates a result file under:

```text
results/<dataset_name>/<model_name>_logits.json
```

Each output record contains:

- `idx`: original sample index
- `prompt`: input prompt
- `human_score`: human annotation
- `direct_socre`: final-layer argmax score
- `weighted_socre`: final-layer expected score
- `weighted_direct_socre`: average direct score across layers
- `df`: layer-wise logits, probabilities, direct scores, and expected scores

The legacy keys `direct_socre` and `weighted_socre` are kept for compatibility with existing result files.

---

## Direct Judgment Setting

To run extraction in the direct setting:

```bash
bash scripts/run_direct.sh
```

Before running, edit the following fields in the script:

```bash
PROJECT_DIR=/path/to/project
MODEL_CACHE_DIR=/path/to/model/cache
PYTHON_BIN=/path/to/python
CUDA_DEVICES=0
```

The direct setting uses short generation and expects the model to output the final judgment directly.

---

## Reasoning Judgment Setting

To run extraction in the reasoning setting:

```bash
bash scripts/run_reasoning.sh
```

This setting uses:

```bash
--with_feedback
```

and allows longer generation before extracting the final score token.

---

## API-based Log-probability Extraction

For closed-source or API-based models, use:

```bash
python int-rule/get_api_outputs.py \
  --input_file data/main/flask.json \
  --save_dir results \
  --model_name gpt-4o \
  --top_logprobs 20 \
  --temperature 0
```

Set the API key with:

```bash
export OPENAI_API_KEY="your_api_key"
```

The output file is saved as:

```text
results/<dataset_name>/<model_name>_api_logprobs.json
```

For API models, the `logits` field stores candidate-label top log-probabilities at the detected score-token position, not full-vocabulary logits.

---

## Stage 2: Rule Recalibration and Evaluation

After extraction, run:

```bash
RESULT_ROOT=results python int-rule/main.py
```

This script evaluates the following methods:

- `Raw`
- `Int-Logit-Avg`
- `Int-Logit-W`
- `Int-Prob-Avg`
- `Int-Prob-W`
- `Final-Rule`
- `Final-Rule++`
- `Int-Rule`
- `Int-Rule++`

The output files are saved under:

```text
results/compare_rerun/
```

For API-based results, outputs are saved under:

```text
results/compare_api_test/
```

---

## Calibration Settings

The calibration subset sizes are:

```text
20, 40, 80, 160, 320
```

The optimization is repeated with five random seeds:

```text
42, 123, 3407, 2020, 2026
```

The main hyperparameters are:

```text
Layer-weight training:
- optimizer: Adam
- epochs: 2
- learning rate: 1e-2
- minimum learning rate: 1e-3
- batch size: 8

Latent initialization:
- steps: 2000
- learning rate: 1e-2
- L-BFGS refinement: enabled when applicable

Rule recalibration:
- outer iterations: 50
- update steps per stage: 100
- learning rate: 1e-2
```

---

## Outputs

The main evaluation script produces:

```text
correlation.csv
prob_metrics.csv
mean_metrics.csv
uncertainty_metrics.csv
```

and analysis payloads under:

```text
analysis_payload/
```

The metrics include:

- correlation with human mean scores
- accuracy
- negative log-likelihood
- cross-entropy against empirical human distributions
- calibration error
- soft calibration error
- binned Jensen-Shannon divergence
- expected-score MSE and MAE

---

## Notes

- The scripts use local model paths. Replace `/path/to/model` and `/path/to/model/cache` with your own local paths when running.
- Large model checkpoints and raw model caches are not included.
- API keys should be provided through environment variables.
