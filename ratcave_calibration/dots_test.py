import click
import pyglet
import ratcave as rc
import numpy as np
from itertools import product

display = pyglet.window.get_platform().get_default_display()
screen = display.get_screens()[1]
window = pyglet.window.Window(fullscreen=True, screen=screen)

reader = rc.WavefrontReader(rc.resources.obj_primitives)

SCALE = .01
spheres = []
for row, col in product(*[np.linspace(-1, 1, 20)]*2):
    spheres.append(reader.get_mesh('Sphere', position=(row, col, -1), scale=SCALE,))
    spheres[-1].uniforms['flat_shading'] = True

scene = rc.Scene(meshes=spheres, bgColor=(0,)*3)
scene.camera.ortho_mode = True
scene.camera._update_projection_matrix()
scene.camera.update()


@window.event
def on_draw():
    scene.draw()

def update(dt):
    pass
pyglet.clock.schedule(update)

@click.command()
def run():
    """Displays dot pattern on second screen.  Useful for checking that Motive is detecting the dots."""
    pyglet.app.run()


if __name__ == '__main__':
    run()