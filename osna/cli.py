# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for elevate_osna."""
    click.echo("Replace this message by putting your code into "
               "osna.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
def web(twitter_credentials):
	from .app import app
	app.run(host='0.0.0.0', debug=True)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
