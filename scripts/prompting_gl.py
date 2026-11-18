"""MWEiC Classification Prompting Evaluation (galician instructions).

Evaluates prompting (LLM) baselines on Galician multiword expression classification tasks.
Uses few-shot prompting with LLM models to classify MWE meanings as same or different.
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_dataset, setup_logging
from get_baselines import normalize_sentence, get_model_response_batched, evaluate_predictions


# ================================================================================
#   Constants
# ================================================================================

SPLITS = ["train", "dev", "test"]
MWE_DATASETS = ['noun_adj', 'verb_obj']
PROMPTING_MODELS = [
    'Qwen/Qwen3-8B',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct'
]
INSTRUCTIONS = (
        "A túa tarefa é clasificar unha expresión multipalabra (MWE) en galego que aparece en dous contextos, indicando se presenta o mesmo significado ou un significado diferente.\n"
        "Responde só con V (Verdadeiro) se a MWE presenta o mesmo significado nas dúas oracións, ou F (Falso) se ten sentidos diferentes en cada oración.\n"
        "A entrada contén tres elementos: a MWE, a oración 1 e a oración 2. As expresións multipalabra poden aparecer en diferentes formas debido á inflexión morfolóxica.\n"
        "A saída debe ser unha única letra: V ou F.\n"
        "MOI IMPORTANTE: NON incúas a MWE, as oracións, nin ningunha explicación na saída. Só V ou F.\n"
    )
BATCH_SIZE = 32


# ================================================================================
#   Functions
# ================================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate prompting experiment on MWEiC datasets'
    )
    parser.add_argument('--dataset_folder', type=str, default="dataset/",
                        help='Folder containing TSV datasets (default: "dataset/")')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    parser.add_argument('--output_folder', type=str, default='results/',
                        help='Output folder for results (default: "results/")')
    return parser.parse_args()


def build_examples(dataset: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Build few-shot examples (positive and negative) from dataset."""

    if not all(label in dataset['label'].values for label in ['T', 'F']):
        raise ValueError("Dataset must contain both 'T' and 'F' labels")
    
    examples = {"T": None, "F": None}

    for label in ["T", "F"]:
        row = dataset[dataset['label'] == label].sample(n=1, random_state=42).iloc[0]

        example_label = "V" if label == "T" else "F"

        mwe = normalize_sentence(row['mwe'])
        sent1 = normalize_sentence(row['sent_ex1'])
        sent2 = normalize_sentence(row['sent_ex2'])

        examples[label] = {
            "input": (
                f"MWE: {mwe}\n"
                f"Oración 1: {sent1}\n"
                f"Oración 2: {sent2}"
            ),
            "output": example_label
        }

    return examples


def get_prompting_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: dict[str, pd.DataFrame],
    instructions: str,
    examples: dict[str, dict[str, str]],
    batch_size: int = 32,
) -> dict[str, object]:
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
            f"Instrución:\n{instructions}\n\n"
            f"Exemplo de entrada (positivo):\n{examples['T']['input']}\n\n"
            f"Exemplo de saída (positivo): {examples['T']['output']}\n\n"
            f"Exemplo de entrada (negativo):\n{examples['F']['input']}\n\n"
            f"Exemplo de saída (negativo): {examples['F']['output']}\n\n"

            f"Entrada:\n"
            f"MWE: {mwe}\n"
            f"Oración 1: {sent1}\n"
            f"Oración 2: {sent2}\n"
            f"Saída: "
        )

        messages_list.append([
            {"role": "system", "content": "You are an expert on understanding multiword expressions in Galician."},
            {"role": "user", "content": prompt},
        ])
    
    # Process in batches
    predictions = get_model_response_batched(model, tokenizer, messages_list, batch_size)

    labels = ["V" if l == "T" else "F" for l in dataset['test']['label'].values]
    result = evaluate_predictions(labels, predictions)

    return result


# ================================================================================
#   Main code
# ================================================================================


def main() -> None:
    """Main pipeline: evaluate prompting experiment"""

    start_time = time.time()

    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("MWEiC Classification Baselines Evaluation")
    logging.info(f"Dataset folder: {args.dataset_folder}")
    logging.info(f"Output folder: {args.output_folder}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    dataset_folder = Path(args.dataset_folder)

    all_datasets = {}
    for dataset_type in MWE_DATASETS:
        all_datasets[dataset_type] = {}
        for split in SPLITS:
            dataset_path = dataset_folder / f"{dataset_type}_{split}.tsv"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            all_datasets[dataset_type][split] = load_dataset(dataset_path, sep="\t")
    
    all_datasets['combined'] = {
        split: pd.concat(
            [
                all_datasets[dataset_type][split]
                for dataset_type in MWE_DATASETS
            ],
            ignore_index=True,
        )
        for split in SPLITS
    }

    # 1. Prompting baseline
    prompting_results = []
    for model_name in PROMPTING_MODELS:
        logging.info(f"Processing model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        try:
            for dataset_type in tqdm(all_datasets.keys(), desc=f"  Datasets", leave=False):
                dataset = all_datasets[dataset_type]
                examples = build_examples(dataset['train'])

                try:
                    result = get_prompting_baseline(
                        model, tokenizer, dataset, INSTRUCTIONS, examples
                    )
                    prompting_results.append(
                        {
                            "baseline": "prompting",
                            "model": model_name,
                            "dataset": dataset_type,
                            **result,
                        }
                    )
                except Exception as e:
                    logging.error(f"Error processing {model_name} on {dataset_type}: {e}")
                    continue
        finally:
            del model
            del tokenizer

    # Save results to parquet
    prompting_output = output_folder / "prompting_GL.parquet"
    prompting_df = pd.DataFrame(prompting_results)
    prompting_df.to_parquet(prompting_output, index=False)
    logging.info(f"{len(prompting_results)} prompting results saved to: {prompting_output}")


    elapsed_time = time.time() - start_time
    logging.info("Evaluation completed")
    logging.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")
    logging.info(f"Total results generated: {len(prompting_results)} entries")
    logging.info(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    main()