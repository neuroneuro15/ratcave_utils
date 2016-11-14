import pyglet
pyglet.options['debug_gl'] = False
from pyglet.window import key
import ratcave as rc
import _transformations as trans
import numpy as np
import natnetclient
import socket
import pickle
import pdb

RIGID_BODY_NAME = 'Rat'
ADJUSTMENT_STEP_SIZE = .001

# Connect to Motive and get tracked object
try:
    tracker = natnetclient.NatClient()
except socket.error as excpt:
    raise IOError("NatNet Server not detected.  Confirm that Motive is streaming data to localhost!")

try:
    rbody = tracker.rigid_bodies[RIGID_BODY_NAME]
    assert bool(rbody), "No rigid body position found.  Make sure rigid bodies are streaming from Motive (off in Motive now by default)"
except KeyError:
    raise KeyError("Rigid Body {} not detected.  Add it in Motive, so we can track it!".format(RIGID_BODY_NAME))

print(tracker.rigid_bodies)


# Load up Projector
with open('projector_data.pickle') as f:
    projdict = pickle.load(f)


modelmat = np.identity(4)
modelmat[:-1, :-1] = -projdict['rotation']
rot  = np.degrees(trans.euler_from_matrix(modelmat, 'rxyz'))
print(modelmat)
print(rot)
projector = rc.Camera(fov_y=projdict['fov_y'], position=projdict['position'])
# projector.rot_x = -90
projector.rot_x = rot[0] - 1.
projector.rot_y = rot[1]
projector.rot_z = 180 - rot[2]
# projector._rot_matrix = projdict['rotation']

display = pyglet.window.get_platform().get_default_display()
screen = display.get_screens()[1]
window = pyglet.window.Window(vsync=True, fullscreen=True, screen=screen)

reader = rc.WavefrontReader(rc.resources.obj_primitives)
sphere = reader.get_mesh('Sphere', scale=.01)
sphere.position = rbody.position
print(sphere.uniforms)
sphere.uniforms['flat_shading'] = 1
scene = rc.Scene(meshes=[sphere], bgColor=(0, .02, 0))
scene.camera = projector
scene.light.position = projector.position
label = pyglet.text.Label('FPS', font_size=20)
label_fps = pyglet.text.Label('FPS', font_size=20)
fps_display = pyglet.clock.ClockDisplay()


@window.event
def on_draw():

    sphere.position = rbody.position
    scene.draw()

    # label.draw()
    # label_fps.draw()
    fps_display.draw()

keys = key.KeyStateHandler()
window.push_handlers(keys)

def update(dt):
    label.text = 'x={:.2f}, y={:.2f}, z={:.2f}'.format(*rbody.position)
    label_fps.text = 'FPS: {}'.format(1. / (dt + .000001))
    if keys[key.UP]:
        projector.z += ADJUSTMENT_STEP_SIZE
    if keys[key.DOWN]:
        projector.z -= ADJUSTMENT_STEP_SIZE
    if keys[key.LEFT]:
        projector.x -= ADJUSTMENT_STEP_SIZE
    if keys[key.RIGHT]:
        projector.x += ADJUSTMENT_STEP_SIZE

pyglet.clock.schedule(update)


pyglet.app.run()
