import motive
import click

@click.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
def trackbody(motive_filename, body):

    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)

    body = motive.get_rigid_bodies()[body]
    while True:
        motive.update()
        click.echo(body.location)