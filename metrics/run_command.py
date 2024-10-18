import click

from metrics.calculate import *

# List of supported metrics
M_LIST = ['pcc', 'cs', 'ed']


@click.command("merge_cal")
@click.argument("model_1_name")  # Name of the first model
@click.argument("model_2_name")  # Name of the second model
@click.argument("model_base_name")  # Name of the base model for comparison
@click.argument("metric")  # Metric(s) to use for the calculation (space-separated)
def main(
        model_1_name: str,
        model_2_name: str,
        model_base_name: str,
        metric: str,
):
    """
        This function calculates the model kinship between model_1 and model_2
        relative to a base model, model_base_name.
        """
    # Extract delta parameters between models for calculation
    d1, d2 = extract_delta_parameters(model_1_name, model_2_name, model_base_name)

    # Iterate over metrics (in case multiple are provided, separated by space)
    for m in metric.split():
        # Check if the provided metric is supported
        if m in M_LIST:
            # Calculate and display the result for the given metric
            click.echo(calculate_metric(d1, d2, m))
        else:
            # Output error if an unsupported metric is provided
            click.echo(f"Error: Metric '{m}' does not exist.")


if __name__ == '__main__':
    main()
