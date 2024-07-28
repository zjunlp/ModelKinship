import click


from calculate import calculate_mk


@click.command("mk_cal")
@click.argument("model_1_name")
@click.argument("model_2_name")
@click.argument("model_base_name")
@click.argument("method")

def main(
    model_1_name: str,
    model_2_name: str,
    model_base_name: str,
    method: str,
):

    if method == 'kinship':
        calculate_mk(
            model_1_name,
            model_2_name,
            model_base_name)
    else:
        click.echo('Method does not exist')

if __name__ == '__main__':
    main()

