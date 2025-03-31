import click
from metrics.calculate import calculate_model_kinship
from metrics.calculate_split import calculate_model_kinship_split
from metrics.utility import validate_models, extract_delta_parameters

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
@click.option(
    "--split-calculation",
    is_flag=True,
    default=False,
    help="Calculate similarity per split instead of full vector"
)
def main(
    model_1_name: str,
    model_2_name: str,
    model_base_name: str,
    metric: str,
    low_precision: bool,
    split_calculation: bool,
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

        if split_calculation:
        # Extract parameters
            results = calculate_model_kinship_split(
                model_1_name,
                model_2_name,
                model_base_name,
                low_precision=low_precision,
                metrics=metrics
            )
        else:
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
