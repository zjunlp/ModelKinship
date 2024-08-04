import click

from metrics.calculate import *

S_LIST = ['pc','ed']


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
    # calculate Pearson Coefficient
    if metric in S_LIST:
        click.echo(calculate_metric(
            model_1_name,
            model_2_name,
            model_base_name,
            metric))
    else:
        click.echo('Metric does not exist')


if __name__ == '__main__':
    main()
