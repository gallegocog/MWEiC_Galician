import argparse
import logging
import time
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import setup_logging


# ================================================================================
#   Constants
# ================================================================================

DATASETS = ["SentDisjCorp", "MWEGenCorp"]
PATTERN_ORDER = ['noun-adj', 'verb-obj', 'combined']
RESULTS_MODELS_ORDER = ['bert-GL', 'Goldfish-GL', 'Carvalho-8B', 'Qwen2.5-7B', 'Qwen3-8B']
FASTTEXT_MODELS_ORDER = ['cc.gl.300', 'fasttext-sg-300d-w5']
PROMPT_MODELS_ORDER = ['Qwen3-8B', 'Qwen2.5-7B-Inst', 'Llama-3.2-3B-Inst']


# ================================================================================
#   Functions
# ================================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze MWE classification results and create visualizations'
    )
    parser.add_argument('--results_folder', type=str, default="/results",
                        help='Folder containing results parquet files (default: "results")')
    parser.add_argument('--save_files', action='store_true',
                        help='Whether to save generated .tex and img files (default: False)')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    return parser.parse_args()


def format_dataframes(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and recode some columns for clarity and consistency."""
    df = df.copy()
    df = df.replace({
        "trained_on": {
            "noun_adj": "noun-adj",
            "verb_obj": "verb-obj",
        },
        "tested_on": {
            "noun_adj": "noun-adj",
            "verb_obj": "verb-obj",
        },
        "model": {
            "bert-base-gl-cased": "bert-GL",
            "glg_latn_1000mb": "Goldfish-GL",
            "Llama-Carvalho-PT-GL": "Carvalho-8B",
            "Qwen3-8B": "Qwen3-8B",
            "Qwen2.5-7B": "Qwen2.5-7B",
            "Qwen/Qwen3-8B": "Qwen3-8B",
            "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Inst",
            "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Inst",
            "cc.gl.300": "cc.gl.300",
            "fasttext_sg_300d_w5": "fasttext-sg-300d-w5",
        },
        "comp_type": {
            "mean_comp": "mean",
            "sum_comp": "sum",
            "mean_incl_cont": "mean\n(cont)",
            "mean_incl_cls": "mean\n(cls)",
            "sum_incl_cont": "sum\n(cont)",
            "sum_incl_cls": "sum\n(cls)",
        }
    })

    return df


def create_summary_table(
    results_df: pd.DataFrame,
    group_cols: List[str],
    model_order: List[str],
    pattern_order: List[str] = PATTERN_ORDER
) -> pd.DataFrame:
    """Create a summary table with max accuracy per group."""
    results_df = results_df.copy()
    table = (
        results_df
        .groupby(group_cols, as_index=False)['accuracy']
        .max()
        .round(3)
    )
    
    for col in ['trained_on', 'tested_on']:
        if col in table.columns:
            table[col] = pd.Categorical(
                table[col],
                categories=pattern_order,
                ordered=True
            )

    table['model'] = pd.Categorical(
        table['model'],
        categories=model_order,
        ordered=True
    )

    table = table.sort_values(
        by=group_cols
    )

    return table


def create_best_metric_table(
    results_df: pd.DataFrame,
    metric: str,
    model_order: List[str] = RESULTS_MODELS_ORDER,
    pattern_order: List[str] = PATTERN_ORDER
) -> pd.DataFrame:
    """Extract best metric value per model-pattern combination."""
    group_cols = ["trained_on", "tested_on", "model"]
    extra_cols = ["comp_type", "layer"]

    idx = results_df.groupby(group_cols)[metric].idxmax()
    result = results_df.loc[idx, group_cols + [metric] + extra_cols].copy()

    for col in ["trained_on", "tested_on"]:
        result[col] = pd.Categorical(
            result[col],
            categories=pattern_order,
            ordered=True
        )

    result["model"] = pd.Categorical(
        result["model"],
        categories=model_order,
        ordered=True
    )

    result = result.sort_values(["trained_on", "tested_on", "model"]).reset_index(drop=True)

    result[metric] = result[metric].round(3)

    return result


def plot_boxplot(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotation: float = 0,
    ylim: Optional[Tuple[float, float]] = None,
    showmeans: bool = True
) -> plt.Figure:
    """Create a boxplot with optional hue-based grouping."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        palette="Set2" if hue else [sns.color_palette("Set2")[0]],
        boxprops=dict(linewidth=1),
        showmeans=showmeans,
        meanprops=dict(marker='D', markeredgecolor="black", markerfacecolor='white', markersize=5)
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel or x, fontsize=9)
    ax.set_ylabel(ylabel or y, fontsize=9)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    if hue:
        ax.legend(title=None, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=8)

    plt.tight_layout()

    return fig


def merged_layer_plot(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    hue_order: Optional[List[str]] = None,
    fasttext_line: Optional[float] = None,
    prompt_line: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 5)
) -> plt.Figure:
    """Plot accuracy by layer with reference lines."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        marker="o",
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(sorted(plot_df[x].unique()))

    if ylim is not None:
        ax.set_ylim(ylim)

    if fasttext_line is not None:
        ax.axhline(fasttext_line, linestyle="--", color="gray", label="fastText")

    if prompt_line is not None:
        ax.axhline(y=prompt_line, linestyle="--", color="black", label="Qwen2.5 (prompt)")

    ax.legend(title="Model")
    ax.grid(False)

    fig.tight_layout()

    return fig

def model_layer_plot(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str,
    style_by: str,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 5)
) -> plt.Figure:
    """Plot accuracy by layer by dataset and pattern."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=plot_df,
        x=x,
        y=y,
        hue=color_by, 
        style=style_by, 
        markers=True,
        dashes=True,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(sorted(plot_df[x].unique()))

    if ylim is not None:
        ax.set_ylim(ylim)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=9,
        markerscale=0.8,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3
    )
    legend.set_title(None)

    ax.grid(False)

    fig.tight_layout()
    return fig


# ================================================================================
#   Main code
# ================================================================================


def main() -> None:
    """Main pipeline for analyzing MWE classification results."""
    start_time = time.time()

    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting results analysis")
    logging.info(f"Results folder: {args.results_folder}")

    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    figs_dir = Path("figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Results for each dataset
    all_results = []
    baseline_avg = {
        "prompt": {pattern: [] for pattern in ["noun-adj", "verb-obj"]},
        "fasttext": {pattern: [] for pattern in ["noun-adj", "verb-obj"]},
    }

    for dataset in DATASETS:

        dataset_results = {}
        for file in ["regressors", "fasttext", "prompting"]:
            file_path = Path(args.results_folder) / dataset / f"{file}.parquet"
            logging.info(f"Loading {file_path}...")
            dataset_results[file] = (
                pd.read_parquet(file_path)
                .assign(dataset=dataset)
            )

        dataset_results = {
            key: format_dataframes(df) for key, df in dataset_results.items()
        }
        all_results.append(dataset_results["regressors"])

        # Show fasttext and prompting results
        prompt_table = create_summary_table(
            dataset_results["prompting"],
            group_cols=["language", "tested_on", "model"],
            model_order=PROMPT_MODELS_ORDER
        )
        fasttext_table = create_summary_table(
            dataset_results["fasttext"],
            group_cols=["trained_on", "tested_on", "model"],
            model_order=FASTTEXT_MODELS_ORDER
        )

        for baseline_name, summary_table_df in [("prompt", prompt_table), ("fasttext", fasttext_table)]:
            print(f"\n=== {dataset} - {baseline_name} results ===")
            print(summary_table_df)

            if args.save_files:
                tex_path = tables_dir / f"{dataset}_{baseline_name}.tex"
                summary_table_df.to_latex(
                    tex_path,
                    index=True,
                    float_format="%.3f"
                )

        # Save baseline and prompting accuracies for line plots
        for pattern in ["noun-adj", "verb-obj"]:
            baseline_avg["prompt"][pattern].append(
                prompt_table.query(
                    f'language == "gal" and tested_on == "{pattern}" and model == "Qwen2.5-7B-Inst"'
                )["accuracy"].mean()
            )

            baseline_avg["fasttext"][pattern].append(
                fasttext_table.query(
                    f'trained_on == "{pattern}" and tested_on == "{pattern}" and model == "fasttext-sg-300d-w5"'
                )["accuracy"].mean()
            )

        # Best accuracy by model for separate and combined training
        regressors_df = dataset_results["regressors"]
        results_by_training = {
            "separated": regressors_df[regressors_df["trained_on"] != "combined"],
            "combined": regressors_df[regressors_df["trained_on"] == "combined"]
        }
        for training, results_df in results_by_training.items():
            best_acc_df = create_best_metric_table(
                results_df,
                metric="accuracy",
            )

            print(f"\n=== {dataset} - {training} training ===")
            print(best_acc_df)

            if args.save_files:
                best_acc_df.to_latex(
                    tables_dir / f"{dataset}_acc_{training}.tex",
                    index=True,
                    float_format="%.3f"
                )

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Average baseline accuracies across datasets
    baseline_avg = {
        "prompt": {pattern: sum(vals) / len(vals) for pattern, vals in baseline_avg["prompt"].items()},
        "fasttext": {pattern: sum(vals) / len(vals) for pattern, vals in baseline_avg["fasttext"].items()}
    }

    # 2. MWE representation comparison - only separate training, reported models, all datasets merged
    all_results_df = all_results_df[all_results_df["model"].isin(
        ['bert-GL', 'Goldfish-GL', 'Carvalho-8B', 'Qwen2.5-7B']
    )]
    all_results_df = all_results_df[all_results_df["trained_on"] != "combined"]

    mwe_repr_fig = plot_boxplot(
        all_results_df,
        x="comp_type",
        y="accuracy",
        hue="trained_on",
        xlabel="MWE representation",
        ylabel="Accuracy",
        figsize=(6, 3),
        rotation=0,
        ylim=(0.4, 0.9)
    )
   
    if mwe_repr_fig:
        if args.save_files:
            mwe_repr_fig.savefig(figs_dir / "mwe_repr_boxplot.pdf", dpi=300, bbox_inches="tight")
        mwe_repr_fig.show()

    # Plot accuracy by layer - combined datasets, best average method per model
    for pattern in ["noun-adj", "verb-obj"]:
        pattern_df = all_results_df[all_results_df["trained_on"] == pattern].copy()

        model_means = pattern_df.groupby(['model', 'comp_type'])['accuracy'].mean()
        best_pairs = model_means.groupby(level=0).idxmax()
        best_repr_df = pattern_df.set_index(['model', 'comp_type']).loc[best_pairs].reset_index()

        layers_fig = merged_layer_plot(
            best_repr_df,
            x="layer",
            y="accuracy",
            hue="model",
            hue_order=[m for m in RESULTS_MODELS_ORDER if m != "Qwen3-8B"],
            prompt_line=baseline_avg["prompt"][pattern],
            fasttext_line=baseline_avg["fasttext"][pattern],
            ylim=(0.5, 0.8)
        )

        if layers_fig:
            if args.save_files:
                layers_fig.savefig(figs_dir / f"{pattern}_layers_lineplot.pdf", dpi=300, bbox_inches="tight")
            layers_fig.show()

    # Plot accuracy by layer - separate datasets, top method per model and pattern
    for model in ["bert-GL", "Goldfish-GL", "Carvalho-8B", "Qwen2.5-7B"]:
        model_df = all_results_df[all_results_df["model"] == model].copy()

        best_pairs = model_df.loc[
            model_df.groupby('trained_on')['accuracy'].idxmax(),
            ['trained_on', 'comp_type']
        ]

        filtered_model_df = (
            model_df
            .set_index(['trained_on', 'comp_type'])
            .loc[pd.MultiIndex.from_frame(best_pairs)]
            .reset_index()
        )
        filtered_model_df = filtered_model_df.rename(columns={"trained_on": "pattern"})

        model_plot = model_layer_plot(
            filtered_model_df,
            x="layer",
            y="accuracy",
            color_by="dataset",
            style_by="pattern",
            ylim=(0.45, 0.90)
        )

        if model_plot:
            if args.save_files:
                model_plot.savefig(figs_dir / f"{model}_layers_lineplot.pdf", dpi=300, bbox_inches="tight")
            model_plot.show()


    elapsed_time = time.time() - start_time
    logging.info("Pipeline completed successfully!")
    logging.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} minutes)")


if __name__ == "__main__":
    main()