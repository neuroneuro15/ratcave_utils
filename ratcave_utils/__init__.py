import click

@click.group()
def cli():
    """
    Group of command-line-tools, useful for using the ratcave VR setup. Each subcommand can also be called on its own.

    WARNIING: Don't run these scripts in Powershell on Windows!  For some reason, the calibration results are coming out
    different in this environment.  Strange but true.
    """
    pass

from . import arena_scanner
from . import calib_projector
from . import track_rigidbody
from . import view_mesh
from . import dots_test
from . import view_arenafit
from . import utils
from .logging import Logger