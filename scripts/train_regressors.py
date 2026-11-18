import argparse
import ast
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import *


# ================================================================================
#   Constants
# ================================================================================

SPLITS = ["train", "dev", "test"]
EMBEDDING_TYPES = ['comp', 'cont', 'cls']
COMPOUND_TYPES = ['mean_comp', 'mean_incl_cont', 'mean_incl_cls',
                  'sum_comp', 'sum_incl_cont', 'sum_incl_cls']
POOLING_METHODS = ["mean", "sum"]
MWE_PATTERNS = ['noun_adj', 'verb_obj']
PARAM_GRID = [
    {
        "solver": ["liblinear"],
        "l1_ratio": [1.0, 0.0],
        "C": [0.001, 0.01, 0.1],
    }
]


# ================================================================================
#   Functions
# ================================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train logistic regressors on MWE embeddings for WIC task'
    )
    parser.add_argument('--dataset_folder', type=str, default="dataset/",
                        help='Folder containing TSV datasets (default: "dataset/")')
    parser.add_argument('--embeddings_folder', type=str, default="embeddings/",
                        help='Folder containing embedding files (default: "embeddings/")')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to process. If not specified, processes all available models (default: None)')
    parser.add_argument('--output', type=str, default='regressors.parquet',
                        help='Output parquet filename (default: "regressors.parquet")')
    return parser.parse_args()


def create_compound_embeddings(embs: dict[str, torch.Tensor], pooling_method: str) -> dict[str, torch.Tensor]:
    """Create compound embeddings by combining comp, cont, cls with given pooling method."""
    if pooling_method not in ["mean", "sum"]:
        raise ValueError(f"pooling_method must be 'mean' or 'sum', got '{pooling_method}'")
    
    if pooling_method == "mean":
        cont_combined = (embs['comp'] + embs['cont']) / 2
        cls_combined = (embs['comp'] + embs['cls']) / 2
    else:  # sum
        cont_combined = embs['comp'] + embs['cont']
        cls_combined = embs['comp'] + embs['cls']

    compound_embs = {
        f"{pooling_method}_comp": embs['comp'],
        f"{pooling_method}_incl_cont": cont_combined,
        f"{pooling_method}_incl_cls": cls_combined
    }

    return compound_embs


def get_layer_embeddings(
    embs: dict[str, dict[str, torch.Tensor]],
    layer_idx: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract embeddings for a specific layer across all pooling methods."""
    return {
        pooling: {
            t: tensor[layer_idx]
            for t, tensor in inner.items()
        }
        for pooling, inner in embs.items()
    }


def prepare_embeddings(
    embs: dict[str, dict[str, torch.Tensor]],
    layer: int,
    metadata: dict[str, dict],
    pattern: dict
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract layer embeddings, create compound embeddings and pair them for training."""

    # Extract layer-specific embeddings
    layer_embs = get_layer_embeddings(embs, layer)

    # Create compound representations
    compound_embs = {}
    for pooling in layer_embs.keys():
        compound_embs.update(
            create_compound_embeddings(layer_embs[pooling], pooling)
        )
    
    # Build paired tensors for each compound type and split
    assert metadata['mean']['sent_ids'] == metadata['sum']['sent_ids'], "Sentence IDs must match across embedding types"
    emb_sent_ids = ast.literal_eval(metadata['mean']['sent_ids'])

    prepared_embs = {comp_type:{split: {} for split in SPLITS} for comp_type in compound_embs.keys()}
    for comp_type in compound_embs.keys():
        for split in SPLITS:
            sent_id_pairs = list(zip(pattern[split]["id_ex1"], pattern[split]["id_ex2"]))
            paired_embs = build_paired_tensor(
                compound_embs[comp_type], emb_sent_ids, sent_id_pairs
            )
            prepared_embs[comp_type][split] = paired_embs

    return prepared_embs


def run_experiment(
    train_embs: torch.Tensor,
    dev_embs: torch.Tensor,
    train_labels: np.ndarray,
    dev_labels: np.ndarray,
    test_sets: list[dict[str, object]],
    model_name: str,
    comp_type: str,
    layer: int,
    train_on: str,
    param_grid: list[dict[str, list]] = PARAM_GRID,
) -> list[dict[str, object]]:
    """Train a regressor and evaluate it on multiple test sets."""
    regressor, scaler = train_regressor(
            train_embs, train_labels,
            dev_embs, dev_labels,
            param_grid,
        )

    results = []
    for test_set in test_sets:
        result = evaluate_test(
            regressor, scaler,
            test_set["test_embs"], test_set["test_labels"],
        )
        results.append({
            "trained_on": train_on,
            "tested_on": test_set["test_on"],
            "model": model_name,
            "comp_type": comp_type,
            "layer": layer,
            "n_train_samples": len(train_labels),
            "n_dev_samples": len(dev_labels),
            "n_test_samples": len(test_set["test_labels"]),
            **result,
        })

    del regressor, scaler
    return results


# ================================================================================
#   Main code
# ================================================================================


def main() -> None:
    """Train and evaluate logistic regressors on compound embeddings."""
    start_time = time.time()

    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting Logistic Regression Training Pipeline")
    logging.info(f"Dataset folder: {args.dataset_folder}")
    logging.info(f"Embeddings folder: {args.embeddings_folder}")
    logging.info(f"Output file: {args.output}")

    # Load datasets
    logging.info("Loading datasets...")
    dataset_folder = Path(args.dataset_folder)
    all_patterns = {
        pattern: {
            split: load_dataset(dataset_folder / f"{pattern}_{split}.tsv", sep="\t")
            for split in SPLITS
        }
        for pattern in MWE_PATTERNS
    }
    all_labels = {
        pattern: {
            split: all_patterns[pattern][split]["label"].values
                    for split in SPLITS
        }
        for pattern in MWE_PATTERNS
    }
    combined_labels = {
        split: np.concatenate([
            all_labels[pattern][split]
            for pattern in MWE_PATTERNS
        ], axis=0)
        for split in SPLITS
    }

    # Filter models if specified
    embs_path = Path(args.embeddings_folder)
    available_models = sorted([p.name for p in embs_path.glob("*/*") if p.is_dir()])
    
    if args.models is not None:
        models = sorted(set(args.models) & set(available_models))
        if not models:
            logging.warning(f"No specified models found in available models. Specified: {args.models}, Available: {available_models}")
        else:
            logging.info(f"Processing specified models: {models}")
    else:
        models = set(available_models)
        logging.info(f"Processing all available models: {models}")

    # Process models
    all_results = []
    pbar_models = tqdm(models, desc="Models", colour="green")
    
    for model_name in pbar_models:
        logging.info(f"Processing model: {model_name}")
        model_start = time.time()

        # Load model embeddings     
        model_embs = {
            pattern: {}
            for pattern in MWE_PATTERNS
        }
        embs_metadata = {
            pattern: {}
            for pattern in MWE_PATTERNS
        }
        for pattern in MWE_PATTERNS:
            for pooling in POOLING_METHODS:
                emb_file = Path(args.embeddings_folder) / pattern / model_name / f"{pooling}_pooled.safetensors"                
                embs, metadata = load_embeddings(emb_file, EMBEDDING_TYPES)
                model_embs[pattern][pooling] = embs
                embs_metadata[pattern][pooling] = metadata


        # Process layers
        n_layers = model_embs[MWE_PATTERNS[0]][POOLING_METHODS[0]]['comp'].shape[0]
        for layer_idx in tqdm(range(n_layers), desc="Layers"):

            layer_embs = {}

            # Prepare embeddings for each pattern
            for pattern in MWE_PATTERNS:
                layer_embs[pattern] = prepare_embeddings(
                    model_embs[pattern],
                    layer_idx,
                    embs_metadata[pattern],
                    all_patterns[pattern],
                )

            # Train and evaluate for separated patterns
            for pattern in MWE_PATTERNS:
                pattern_embs = layer_embs[pattern]
                pattern_labels = all_labels[pattern]

                for compound_type in COMPOUND_TYPES:
                    embs = pattern_embs[compound_type]

                    all_results.extend(
                        run_experiment(
                            embs["train"],
                            embs["dev"],
                            pattern_labels["train"],
                            pattern_labels["dev"],
                            [
                                {
                                    "test_on": pattern,
                                    "test_embs": embs["test"],
                                    "test_labels": pattern_labels["test"],
                                }
                            ],
                            model_name,
                            compound_type,
                            layer_idx,
                            pattern,
                        )
                    )
            
            # Train on combined patterns, evaluate on combined and separated
            for compound_type in COMPOUND_TYPES:

                combined_embs = {
                    split: torch.cat(
                        [layer_embs[pattern][compound_type][split] for pattern in MWE_PATTERNS],
                        dim=0
                    )
                    for split in SPLITS
                }
                
                test_sets = [
                    {
                        "test_on": "combined",
                        "test_embs": combined_embs["test"],
                        "test_labels": combined_labels["test"],
                    }
                ]
                test_sets.extend([
                    {
                        "test_on": pattern,
                        "test_embs": layer_embs[pattern][compound_type]["test"],
                        "test_labels": all_labels[pattern]["test"],
                    }
                    for pattern in MWE_PATTERNS
                ])
                all_results.extend(
                    run_experiment(
                        combined_embs["train"],
                        combined_embs["dev"],
                        combined_labels["train"],
                        combined_labels["dev"],
                        test_sets,
                        model_name,
                        compound_type,
                        layer_idx,
                        "combined",
                    )
                )

            del layer_embs

        # Save results after each model
        model_elapsed = time.time() - model_start
        logging.info(f"Model {model_name} completed in {model_elapsed:.2f}s")
        
        del model_embs, embs_metadata
        gc.collect()
        
        results_df = pd.DataFrame(all_results)
        results_df.to_parquet(args.output)
        logging.debug(f"Saved {len(all_results)} results to {args.output}")

    elapsed_time = time.time() - start_time
    logging.info("Pipeline completed successfully!")
    logging.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")
    logging.info(f"Total results generated: {len(all_results)} entries")
    logging.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()