# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import sys

from . import credentials_path, config


@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
	from .app import app
<<<<<<< HEAD
	app.run(host='127.0.0.1', debug=True)
	
=======
	app.run(host='127.0.0.1', debug=True, port=port)

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
	"""
	Read all files in this directory and its subdirectories and print statistics.
	"""
	print('reading from %s' % directory)
	# use glob to iterate all files matching desired pattern (e.g., .json files).
	# recursively search subdirectories.
>>>>>>> 5ffe6275eb2a34deeecd9f88a5aa9373354e223b

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
