import click
import numpy as np
from typing import List, Tuple
from enum import Enum, auto
from metrics.calculate import *
from metrics.utility import *

class Metric(str, Enum):
    """Enum for supported metrics to ensure type safety and autocompletion"""
    PCC = 'pcc'  # Pearson Correlation Coefficient
    CS = 'cs'    # Cosine Similarity
    ED = 'ed'    # Euclidean Distance

    @classmethod
    def list(cls) -> List[str]:
        """Returns list of supported metric values"""
        return [metric.value for metric in cls]


def calculate_model_kinship(
        delta1: np.ndarray,
        delta2: np.ndarray,
        metrics: List[str]
) -> dict:
    """
    Calculate model kinship using specified metrics.

    Args:
        delta1: Delta parameters for first model
        delta2: Delta parameters for second model
        metrics: List of metrics to calculate

    Returns:
        dict: Dictionary of metric names and their calculated values
    """
    results = {}
    for metric in metrics:
        try:
            if metric not in Metric.list():
                raise ValueError(f"Unsupported metric: {metric}")
            results[metric] = calculate_metric(delta1, delta2, metric)
        except Exception as e:
            results[metric] = f"Error calculating {metric}: {str(e)}"
    return results


@click.command("merge_cal")
@click.argument("model_1_name", type=str)
@click.argument("model_2_name", type=str)
@click.argument("model_base_name", type=str)
@click.argument("metric", type=str)
@click.option(
    "--low-precision",
    is_flag=True,
    default=False,
    help="Use low precision for parameter extraction"
)
def main(
    model_1_name: str,
    model_2_name: str,
    model_base_name: str,
    metric: str,
    low_precision: bool,
):
    """
        This function calculates the model kinship between model_1 and model_2
        relative to a base model, model_base_name.
        """
    # Extract delta parameters between models for calculation
    try:
        # Validate input models
        validate_models(model_1_name, model_2_name, model_base_name)

        # Parse metrics
        metrics = metric.split()
        if not metrics:
            raise click.BadParameter("At least one metric must be specified")

        # Extract parameters
        d1, d2 = extract_delta_parameters(
            model_1_name,
            model_2_name,
            model_base_name,
            low_precision=low_precision
        )

        results = calculate_model_kinship(d1, d2, metrics)
        for metric_name, value in results.items():
            click.echo(f"{metric_name}: {value}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
