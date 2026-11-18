import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler


# ================================================================================
#   Data utilities
# ================================================================================


def setup_logging(log_level: str) -> None:
    """Configure logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# ================================================================================
#   Data loading
# ================================================================================


def load_dataset(file: str, sep: str = ',') -> pd.DataFrame:
    """Load dataset from CSV/TSV file removing unnamed columns.
    
    Args:
        file: Path to the CSV/TSV file.
        sep: Column separator (default: ',').
        
    Returns:
        DataFrame with unnamed columns removed.
        
    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    try:
        if not Path(file).exists():
            raise FileNotFoundError(f"Dataset file not found: {file}")

        df = pd.read_csv(file, sep=sep)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        logging.info(f"Successfully loaded {file} with {len(df)} entries")
        
        return df

    except Exception as e:
        logging.error(f"Error loading dataset from {file}: {str(e)}")
        raise


def load_embeddings(file: str, embeddings: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load embeddings and metadata from .safetensor file.
    
    Args:
        file: Path to .safetensor file containing embeddings
        embeddings: Keys of embeddings to load
        
    Returns:
        Tuple of (embeddings dict, metadata dict with sent_ids).
        
    Raises:
        FileNotFoundError: If embeddings file does not exist.
    """
    try:
        if not Path(file).exists():
            raise FileNotFoundError(f"Embeddings file not found: {file}")

        embs = {}
        with safe_open(file, framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                if key in embeddings:
                    embs[key] = f.get_tensor(key)

        logging.debug(f"Successfully loaded {file} with {len(embs)} tensors")
        logging.debug(f"Loaded embeddings shapes: {dict((k, v.shape) for k, v in embs.items())}")

        return embs, metadata

    except Exception as e:
        logging.error(f"Error loading embeddings from {file}: {e}")
        raise


# ================================================================================
#   Logistic regression
# ================================================================================


def build_paired_tensor(
    embs: torch.Tensor,
    available_sent_ids: list[str],
    sent_id_pairs: list[tuple[str, str]],
) -> torch.Tensor:
    """Build paired embeddings tensor from sentence ID pairs.
    
    Args:
        embs: Tensor of shape (n_sentences, vector_dim)
        available_sent_ids: List of sentence IDs in embeddings tensor
        sent_id_pairs: List of (sent_id_1, sent_id_2) tuples to extract
    
    Returns:
        Tensor of shape (n_pairs, 2, vector_dim) where vector_dim=2 stacks
        the two example embeddings side-by-side
    
    Raises:
        ValueError: If any sent_id in pairs not found in available_sent_ids
    """
    # Map sentence IDs to tensor indices
    id_to_idx = {sid: i for i, sid in enumerate(available_sent_ids)}

    # Convert sentence ID pairs to tensor indices
    try:
        idx1 = torch.tensor(
            [id_to_idx[sid1] for sid1, _ in sent_id_pairs],
            dtype=torch.long,
            device=embs.device,
        )
        idx2 = torch.tensor(
            [id_to_idx[sid2] for _, sid2 in sent_id_pairs],
            dtype=torch.long,
            device=embs.device,
        )

    except Exception as e:
        raise RuntimeError(f"Error building tensor indices: {e}")

    emb1 = embs[idx1, :]  # (n_pairs, vector_dim)
    emb2 = embs[idx2, :]  # (n_pairs, vector_dim)
    paired = torch.stack([emb1, emb2], dim=1)  # (n_pairs, 2, vector_dim)

    return paired


def generate_features(pairs_tensor: torch.Tensor) -> np.ndarray:
    """Concatenate paired embeddings into single feature vector.
    
    Args:
        pairs_tensor: Tensor of shape (n_samples, 2, vector_dim) containing
                      two embeddings per sample
    
    Returns:
        NumPy array of shape (n_samples, 2*vector_dim)
    
    Raises:
        ValueError: If tensor shape != (n_samples, 2, embed_dim)
    """
    if pairs_tensor.dim() != 3 or pairs_tensor.shape[1] != 2:
        raise ValueError(f"Expected shape (n_samples, 2, embed_dim), got {pairs_tensor.shape}")
    
    v1 = pairs_tensor[:, 0, :].numpy()
    v2 = pairs_tensor[:, 1, :].numpy()
    return np.concatenate([v1, v2], axis=1)


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    param_grid: list[dict[str, list]]
) -> GridSearchCV:
    """Tune hyperparameters using train+dev.
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training labels
        X_dev: Development features for validation (n_dev, n_features)
        y_dev: Development labels
        param_grid: List of dicts specifying hyperparameter combinations
        
    Returns:
        Fitted GridSearchCV object with best_estimator_ and best_params_.
        
    Raises:
        AssertionError: If feature dimensions or sample counts don't match
    """
    assert X_train.shape[1] == X_dev.shape[1], "Feature dimension mismatch"
    assert X_train.shape[0] == len(y_train), "Train sample count mismatch"
    assert X_dev.shape[0] == len(y_dev), "Dev sample count mismatch"

    # Combine datasets with PredefinedSplit
    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.concatenate([y_train, y_dev])
    splits = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_dev))
    ])
    ps = PredefinedSplit(splits)

    # Evaluate hyperparameters
    scorer = 'accuracy'

    grid = GridSearchCV(
        LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
        param_grid=param_grid,
        cv=ps,
        scoring=scorer,
        n_jobs=-1
    )
    grid.fit(X_combined, y_combined)

    return grid


def train_regressor(
    embs_train: torch.Tensor,
    y_train: np.ndarray,
    embs_dev: torch.Tensor,
    y_dev: np.ndarray,
    param_grid: list[dict[str, list]]
) -> tuple[GridSearchCV, StandardScaler]:
    """Train logistic regressor with GridSearch hyperparameter tuning on dev set.
    
    Args:
        embs_train: Training embeddings tensor (n_train, 2, embedding_dim).
        y_train: Training labels.
        embs_dev: Dev embeddings for validation.
        y_dev: Dev labels.
        param_grid: List of dicts specifying hyperparameter combinations.
    
    Returns:
        Tuple of (fitted GridSearchCV, fitted StandardScaler).
    """
    X_train = generate_features(embs_train)
    X_dev = generate_features(embs_dev)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)

    grid = tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_grid)

    return grid, scaler


def evaluate_test(
    grid: GridSearchCV,
    scaler: StandardScaler,
    embs_test: torch.Tensor,
    y_test: np.ndarray
) -> dict[str, float | str | None]:
    """Predict on test set and return hyperparams + classification metrics.
    
    Args:
        grid: Fitted GridSearchCV with best_estimator_.
        scaler: Fitted StandardScaler.
        embs_test: Test embeddings (n_test, 2, embedding_dim).
        y_test: Test labels.
    
    Returns:
        Dict with keys: l1_ratio, C, solver, dev_score, accuracy, precision/recall/f1 per label.
    """
    X_test = generate_features(embs_test)
    X_test = scaler.transform(X_test)

    y_pred = grid.best_estimator_.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "l1_ratio": grid.best_params_.get("l1_ratio", None),
        "C": grid.best_params_.get("C", None),
        "solver": grid.best_params_.get("solver", None),
        "dev_score": grid.best_score_,
        "accuracy": report["accuracy"],

        **{
            f"{metric}_{label}": report[str(label)][
                "f1-score" if metric == "f1" else metric
            ]
            for label in sorted(set(y_test))
            for metric in ["precision", "recall", "f1"]
        },

        **{
            f"{metric}_macro": report["macro avg"][
                "f1-score" if metric == "f1" else metric
            ]
            for metric in ["precision", "recall", "f1"]
        },
    }