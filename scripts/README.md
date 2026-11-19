# Scripts

This folder contains the scripts for the study on contextual meaning variation in MWEs in Galician. We compared multiple approaches: logistic classifiers trained on contextual embeddings from transformer models, classifiers trained on static embeddings, and LLM-based prompting experiments with both English and Galician instructions.

## Dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```


## Files

### `train_regressors.py` - Transformer Embeddings Classification
Trains logistic regressors on contextualized transformer embeddings with hyperparameter tuning.

- Loads pre-computed transformer embeddings (safetensors format) extracted using the script from [this repository](https://github.com/marcospln/Galician_MWE_dataset).
- Hyperparameter optimization via GridSearchCV on dev set
- Separate evaluation per dataset and combined (noun_adj + verb_obj) training

The script expects the following structure on `--embeddings_folder`:
```
<embeddings_folder>/
├── noun_amod/
│   └── {model_name}/
│       ├── mean_pooled.safetensors
│       └── sum_pooled.safetensors
└── verb_obj/
    └── {model_name}/
        ├── mean_pooled.safetensors
        └── sum_pooled.safetensors
```

#### Command-line Arguments
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--dataset_folder` | str | Folder containing TSV datasets | `"dataset/"` |
| `--embeddings_folder` | str | Folder containing embedding files | `"embeddings/"` |
| `--output` | str | Output parquet filename | `"regressors.parquet"` |
| `--models` | str | Space-separated list of model names | all available |
| `--log_level` | str | Logging level | `INFO` |

#### Output
Parquet file specified by `--output`, containing classification metrics per model and embedding type.

---

### `get_baselines.py` - Baseline Evaluation
Evaluates two alternative approaches for performance analysis:
-  **Static embeddings**: runs a classiflication procedure similar to `train_regressors.py` but using static embeddings from FastText models. The script expects a `fasttext_models` folder with the corresponding `.vec` files.

- **LLM prompting**: runs a few-shot classification procedure using English instructions.

#### Command-line Arguments
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--dataset_folder` | str | Folder containing TSV datasets | `"dataset/"` |
| `--output_folder` | str | Folder for saving results | `"results/"` |
| `--baseline` | str | Which baseline(s): `prompting`, `static`, `all` | `all` |
| `--log_level` | str | Logging level | `INFO` |

#### Output
Parquet files saved to `--output_folder`:
- `prompting.parquet` - LLM evaluation metrics (if `--baseline` is `prompting` or `all`)
- `fasttext.parquet` - FastText evaluation metrics (if `--baseline` is `static` or `all`)

---

### `prompting_gl.py` - Galician Instructions Prompting
Same procedure as the LLM prompting baseline on `get_baselines.py`, but with Galician instructions.

#### Output
`prompting_GL.parquet` file saved to `--output_folder` containing LLM evaluation metrics with Galician instructions.

---

### `analyses.py` - Data Analysis and Visualization
Performs analysis of classification results across all models and generates visualizations.

The scripts expects the following structure in `--results_folder`:
```
results/
├── {dataset}/               # MWEGenCorp and/or SentDisjCorp, defined in DATASETS constant
│   ├── fasttext.parquet
│   ├── prompting.parquet
│   └── regressors.parquet
```
Note: if you want to include both the results from the English and Galician prompting instructions, you must merge `prompting_GL.parquet` into `prompting.parquet` before running the analyses.

#### Command-line Arguments
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--results_folder` | str | Folder containing results parquet files | `"results/"` |
| `--save_files` | flag | Save LaTeX tables and image files | False |
| `--log_level` | str | Logging level | `INFO` |

#### Output
- Summary statistics and model comparisons
- Visualizations and LaTeX tables (`--save_files` enabled)

---
### `utils.py` - Utility Functions
Provides shared helper functions for data loading, logging, and model evaluation.

---

## Workflow
The typical workflow:

```
1. Data preparation (ensure datasets and embeddings folders follow the required structure)
                ↓
2. Run embeddings classification →  train_regressors.py
                ↓
3. Run baselines
    ├─ get_baselines.py
    └─ prompting_gl.py
                ↓
4. Analyze results          →  analyses.py
```