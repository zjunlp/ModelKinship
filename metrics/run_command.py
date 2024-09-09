import click

from metrics.calculate import *

M_LIST = ['pcc', 'ed', 'ed']


@click.command("merge_cal")
@click.argument("model_1_name")
@click.argument("model_2_name")
@click.argument("model_base_name")
@click.argument("metric")
def main(
        model_1_name: str,
        model_2_name: str,
        model_base_name: str,
        metric: str,
):
    # Calculation
    d1, d2 = extract_delta_parameters(model_1_name, model_2_name, model_base_name)
    for m in metric.split():
        click.echo(calculate_metric(
                d1, d2,
                m))
    # else:
    #     click.echo('Metric does not exist')


if __name__ == '__main__':
    main()
