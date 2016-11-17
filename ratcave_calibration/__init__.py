import click

@click.group()
def cli():
    """Group of command-line-tools, useful for using the ratcave VR setup. Each subcommand can also be called on its own."""
    pass

from . import arena_scanner
from . import calib_projector
from . import track_rigidbody
from . import dots_test