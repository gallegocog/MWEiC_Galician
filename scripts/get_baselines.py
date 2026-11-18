"""MWEiC Classification Baselines Evaluation.

Evaluates both prompting (LLMs) and static embeddings (fasttext) baselines
on multiword expression classification.

This script includes two classification approaches:
  1. Prompting: Uses LLMs with few-shot examples
  2. Static Embeddings: Uses fasttext embeddings with logistic regression

Results are saved as Parquet files.
"""
import argparse
import logging
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_dataset, setup_logging, tune_hyperparameters


# ================================================================================
#   Constants
# ================================================================================

SPLITS = ["train", "dev", "test"]
MWE_PATTERNS = ['noun_adj', 'verb_obj']
PROMPTING_MODELS = [
    'Qwen/Qwen3-8B',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct'
]
INSTRUCTIONS = (
        "Your task is to classify a multiword expression (MWE) in Galician occurring in two contexts as conveying the same or different meaning.\n"
        "Reply only with T (True) if the MWE conveys the same meaning in both sentences, or F (False) if it has different senses in each sentence.\n"
        "The input consists on three elements: the MWE, sentence 1, and sentence 2. Multiword expressions may occur with different surface forms due to morphological inflection.\n"
        "Output must contain only one character: T or F.\n"
        "VERY IMPORTANT: DO NOT include the MWE, the sentences or any explanation in the output. Just T or F.\n"
    )
STATIC_MODELS = [
    "cc.gl.300",
    "fasttext_sg_300d_w5"
]
PARAM_GRID = [
    {
        "solver": ["liblinear"],
        "l1_ratio": [1.0, 0.0],
        "C": [0.001, 0.01, 0.1],
    }
]
BATCH_SIZE = 32


# ================================================================================
#   Functions
# ================================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate prompting and static embeddings baselines on MWEiC datasets'
    )
    parser.add_argument('--dataset_folder', type=str, default="dataset/",
                        help='Folder containing TSV datasets (default: "dataset/")')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    parser.add_argument('--output_folder', type=str, default='results/',
                        help='Output folder for results (default: "results/")')
    parser.add_argument('--baseline', type=str, default='all',
                        choices=['prompting', 'static', 'all'],
                        help='Which baseline(s) to evaluate: prompting, static, or all (default: all)')
    return parser.parse_args()


def normalize_sentence(sentence: str) -> str:
    """Normalize sentence using NFC unicode normalization."""
    return unicodedata.normalize("NFC", sentence)


def build_examples(mweic_df: pd.DataFrame) -> dict:
    """Build few-shot examples (positive and negative) from dataset."""

    if not all(label in mweic_df['label'].values for label in ['T', 'F']):
        raise ValueError("DataFrame must contain both 'T' and 'F' labels")
    
    examples = {"T": None, "F": None}

    for label in ["T", "F"]:
        row = mweic_df[mweic_df['label'] == label].sample(n=1, random_state=42).iloc[0]

        mwe = normalize_sentence(row['mwe'])
        sent1 = normalize_sentence(row['sent_ex1'])
        sent2 = normalize_sentence(row['sent_ex2'])

        examples[label] = {
            "input": (
                f"MWE: {mwe}\n"
                f"Oración 1: {sent1}\n"
                f"Oración 2: {sent2}"
            ),
            "output": label
        }

    return examples


def get_model_response_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_batch: list[list],
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """Generate model responses from a batch of chat messages."""

    # Apply chat template to all messages
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True
        )
        for messages in messages_batch
    ]

    all_outputs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        try:
            # Generate responses
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    top_k=0,
                    top_p=0,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract generated tokens
            batch_outputs = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            decoded = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            all_outputs.extend(decoded)

        finally:
            del model_inputs

    return all_outputs


def evaluate_predictions(
    labels: np.ndarray, predictions: list
) -> dict[str, float]:
    """Compute classification metrics (accuracy, precision, recall, F1).

    Args:
        labels: Ground truth labels.
        predictions: Predicted labels.

    Returns:
        Dict with per-label metrics and macro averages.
    """
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    result = {
        "accuracy": report["accuracy"],

        **{
            f"{metric}_{label}": report[str(label)][
                "f1-score" if metric == "f1" else metric
            ]
            for label in sorted(set(labels))
            for metric in ["precision", "recall", "f1"]
        },

        **{
            f"{metric}_macro": report["macro avg"][
                "f1-score" if metric == "f1" else metric
            ]
            for metric in ["precision", "recall", "f1"]
        },
    }

    return result


def get_prompting_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: dict,
    instructions: str,
    examples: dict,
    batch_size: int = 32,
) -> dict:
    """Evaluate prompting baseline with few-shot examples (batched).

    Args:
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        dataset: Dict with 'test' split DataFrame.
        instructions: Prompt instructions string.
        examples: Dict with 'T'/'F' few-shot examples.
        batch_size: Number of examples per batch (default: 32).

    Returns:
        Metrics dict from evaluate_predictions().
    """
    # Build all prompts and messages
    messages_list = []
    for entry in dataset['test'].itertuples():
        mwe = normalize_sentence(entry.mwe)
        sent1 = normalize_sentence(entry.sent_ex1)
        sent2 = normalize_sentence(entry.sent_ex2)

        prompt = (
            f"Instruction:\n{instructions}\n\n"
            f"Example Input (positive):\n{examples['T']['input']}\n\n"
            f"Example Output (positive): {examples['T']['output']}\n\n"
            f"Example Input (negative):\n{examples['F']['input']}\n\n"
            f"Example Output (negative): {examples['F']['output']}\n\n"

            f"Input:\n"
            f"MWE: {mwe}\n"
            f"Oración 1: {sent1}\n"
            f"Oración 2: {sent2}\n"
            f"Output: "
        )

        messages_list.append([
            {"role": "system", "content": "You are an expert on understanding multiword expressions in Galician."},
            {"role": "user", "content": prompt},
        ])
    
    # Process in batches
    predictions = get_model_response_batched(model, tokenizer, messages_list, batch_size)
    
    result = evaluate_predictions(dataset['test']['label'].values, predictions)

    return result


def get_word2vec_embeddings(
    model: KeyedVectors, sents: pd.Series
) -> np.ndarray:
    """Generate mean word2vec embeddings for sentences.

    Args:
        model: Loaded KeyedVectors model.
        sents: Series of sentences to embed.

    Returns:
        Array of shape (n_sents, embedding_dim) containing mean embeddings per sentence.
    """
    formatted = (
        sents
        .map(normalize_sentence)
        .str.replace(r'([^\w\s.,;:])', r' \1 ', regex=True)
        .str.replace(r'([.,;:])', r' \1 ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    ).tolist()

    embs = []
    for sent in formatted:
        sent_embs = []
        tokens = sent.split()
        for token in tokens:
            if token in model:
                sent_embs.append(model[token])
        if sent_embs:
            sent_embs_mean = np.mean(sent_embs, axis=0)
        else:
            sent_embs_mean = []
        embs.append(sent_embs_mean)

    return np.array(embs)


def get_static_baseline(
    dataset_embs: dict, dataset_labels: dict
) -> tuple[StandardScaler, object, dict]:
    """Evaluate static fastText embeddings with logistic regression.

    Trains a logistic regression model on scaled embeddings using grid search
    for hyperparameter tuning. Evaluation uses the test split.

    Args:
        dataset_embs: Dict with fastText embeddings for each split.
        dataset_labels: Dict with labels for each split.

    Returns:
        Tuple of (scaler, estimator, metrics_dict) where:
        - scaler: StandardScaler fitted on training embeddings
        - estimator: Fitted logistic regression from best grid search configuration
        - metrics_dict: Dict with accuracy, precision, recall, F1 scores
    """
    # Standardize embeddings
    scaler = StandardScaler()
    train_embs_scaled = scaler.fit_transform(dataset_embs['train'])
    dev_embs_scaled = scaler.transform(dataset_embs['dev'])
    test_embs_scaled = scaler.transform(dataset_embs['test'])

    # Hyperparameter tuning on dev set
    grid = tune_hyperparameters(
        train_embs_scaled, dataset_labels['train'],
        dev_embs_scaled, dataset_labels['dev'],
        PARAM_GRID
    )

    # Evaluate on test set
    estimator = grid.best_estimator_
    predictions = estimator.predict(test_embs_scaled)
    result = evaluate_predictions(dataset_labels['test'], predictions)

    del dataset_embs

    return scaler, estimator, result


# ================================================================================
#   Main code
# ================================================================================


def main():
    """Main pipeline: evaluate prompting and static embeddings baselines."""
    start_time = time.time()

    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("MWEiC Classification Baselines Evaluation")
    logging.info(f"Dataset folder: {args.dataset_folder}")
    logging.info(f"Output folder: {args.output_folder}")
    logging.info(f"Baseline filter: {args.baseline}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    dataset_folder = Path(args.dataset_folder)

    all_patterns = {}
    for pattern in MWE_PATTERNS:
        all_patterns[pattern] = {}
        for split in SPLITS:
            dataset_path = dataset_folder / f"{pattern}_{split}.tsv"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            all_patterns[pattern][split] = load_dataset(dataset_path, sep="\t")
    
    all_patterns['combined'] = {
        split: pd.concat(
            [
                all_patterns[pattern][split]
                for pattern in MWE_PATTERNS
            ],
            ignore_index=True,
        )
        for split in SPLITS
    }

    # 1. Prompting baseline
    prompting_results = []
    if args.baseline in ['prompting', 'all']:
        logging.info("PROMPTING BASELINE")

        for model_name in PROMPTING_MODELS:
            logging.info(f"Processing model: {model_name}")

            try:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                logging.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load model {model_name}: {e}")
                continue

            try:
                for pattern in tqdm(all_patterns.keys(), desc=f"  Datasets", leave=False):
                    dataset = all_patterns[pattern]
                    examples = build_examples(dataset['train'])

                    try:
                        result = get_prompting_baseline(
                            model, tokenizer, dataset, INSTRUCTIONS, examples
                        )
                        prompting_results.append(
                            {
                                "baseline": "prompting",
                                "model": model_name,
                                "tested_on": pattern,
                                **result,
                            }
                        )
                    except Exception as e:
                        logging.error(f"Error processing {model_name} on {pattern}: {e}")
                        continue
            finally:
                del model
                del tokenizer
        
        # Save prompting results
        if prompting_results:
            prompting_output = output_folder / "prompting_EN.parquet"
            prompting_df = pd.DataFrame(prompting_results)
            prompting_df.to_parquet(prompting_output, index=False)
            logging.info(f"{len(prompting_results)} prompting results saved to: {prompting_output}")

    # 2. Static embeddings baseline
    static_results = []
    if args.baseline in ['static', 'all']:
        logging.info("FASTTEXT BASELINE")

        for model_name in STATIC_MODELS:
            logging.info(f"Processing model: {model_name}")

            model_path = Path() / "fasttext_models" / f"{model_name}.vec"
            model = KeyedVectors.load_word2vec_format(model_path, binary=False)

            model_embs = {
                pattern: {} for pattern in MWE_PATTERNS
            }

            # Compute embeddings for all datasets
            for pattern in MWE_PATTERNS:
                dataset = all_patterns[pattern]
                for split in SPLITS:
                    sents1 = dataset[split]['sent_ex1']
                    sents2 = dataset[split]['sent_ex2']

                    embs1 = get_word2vec_embeddings(model, sents1)
                    embs2 = get_word2vec_embeddings(model, sents2)
                    embs_concat = np.concatenate([embs1, embs2], axis=1)
                    model_embs[pattern][split] = embs_concat

            # Train and evaluate on separated datasets
            for pattern in MWE_PATTERNS:
                try:
                    dataset_embs = model_embs[pattern]
                    dataset_labels = {split: all_patterns[pattern][split]['label'].values for split in SPLITS}

                    # Train and evaluate on single dataset
                    _, _, result = get_static_baseline(dataset_embs, dataset_labels)
                    static_results.append(
                        {
                            "baseline": "fasttext",
                            "model": model_name,
                            "trained_on": pattern,
                            "tested_on": pattern,
                            **result,
                        }
                    )
                except Exception as e:
                    logging.error(f"Error processing {model_name} on {pattern}: {e}")
                    continue

            # Train and evaluate on combined dataset
            try:
                combined_embs = {
                    split: np.concatenate(
                        [model_embs[pattern][split] for pattern in MWE_PATTERNS],
                        axis=0
                    )
                    for split in SPLITS
                }
                combined_labels = {
                    split: np.concatenate(
                        [
                            all_patterns[pattern][split]['label'].values
                            for pattern in MWE_PATTERNS
                        ],
                        axis=0
                    )
                    for split in SPLITS
                }

                combined_scaler, combined_estimator, combined_result = get_static_baseline(
                    combined_embs, combined_labels
                )
                static_results.append(
                    {
                        "baseline": "fasttext",
                        "model": model_name,
                        "trained_on": "combined",
                        "tested_on": "combined",
                        **combined_result,
                    }
                )
                
                # Evaluate across patterns: model trained on combined, tested on separated pattern
                for pattern in MWE_PATTERNS:
                    try:
                        test_embs_scaled = combined_scaler.transform(model_embs[pattern]['test'])
                        test_labels = all_patterns[pattern]['test']['label'].values

                        predictions = combined_estimator.predict(test_embs_scaled)
                        result = evaluate_predictions(test_labels, predictions)

                        static_results.append(
                            {
                                "baseline": "fasttext",
                                "model": model_name,
                                "trained_on": "combined",
                                "tested_on": pattern,
                                **result,
                            }
                        )
                    except Exception as e:
                        logging.error(f"Error evaluating {model_name} across {pattern}: {e}")
                        continue

            except Exception as e:
                logging.error(f"Error processing combined dataset for {model_name}: {e}")
            
            finally:
                del model, model_embs

        # Save static results
        if static_results:
            static_output = output_folder / "fasttext.parquet"
            static_df = pd.DataFrame(static_results)
            static_df.to_parquet(static_output, index=False)
            logging.info(f"{len(static_results)} static results saved to: {static_output}")

    elapsed_time = time.time() - start_time
    logging.info("Evaluation completed")
    logging.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()