import pyglet
pyglet.options['debug_gl'] = False
import ratcave as rc
import numpy as np
from collections import deque

#Inputs
OBJ_NAME = 'Monkey'
NUM_OBJECTS = 8
USE_CUBEMAP = True
CUBEMAP_TEXTURE_SIZE = 2048
HAS_UNIFORMS = True
IS_MOVING = True
ROLLING_WINDOW_LEN = 30

# Create Window
window = pyglet.window.Window(resizable=True, vsync=False, fullscreen=True)

# Generate Objects
reader = rc.WavefrontReader(rc.resources.obj_primitives)

player = rc.mesh.EmptyMesh()
screen = reader.get_mesh('Plane')
screen.position = 0, 0, -1
screen.scale = .7
screen.uniforms['diffuse'] = .7, 0, 0

def sphere_factory(reader, n=10):
    for _ in range(n):
        sphere = reader.get_mesh(OBJ_NAME, scale=.1)
        sphere.position = np.append(2 * np.random.random(2) - 1, [-2])
        if not HAS_UNIFORMS:
            sphere.uniforms = rc.UniformCollection()
        else:
            sphere.uniforms['diffuse']= np.random.random(3)
        sphere.update()
        yield sphere

spheres = [sphere for sphere in sphere_factory(reader, NUM_OBJECTS)]
if IS_MOVING:
    for sphere in spheres:
        sphere.rot_velocity = 45 * np.random.random()

# Debug Text
label_fps = pyglet.text.Label('FPS', font_size=20)

# Create Scenes and FBOs to render onto
proj_scene = rc.Scene(meshes=spheres)
proj_scene.camera.aspect = 1.
proj_scene.camera.fov_y = 90.

god_scene = rc.Scene(meshes=[player, screen])
cubetexture = rc.TextureCube(width=CUBEMAP_TEXTURE_SIZE, height=CUBEMAP_TEXTURE_SIZE)
screen.texture = cubetexture
fbo_cube = rc.FBO(cubetexture)

# Run Display and Update Loops
@window.event
def on_resize(width, height):
    # proj_scene.camera.aspect = width / float(height)
    god_scene.camera.aspect = width / float(height)

@window.event
def on_draw():
    # proj_scene.draw()
    window.clear()
    if USE_CUBEMAP:
        with fbo_cube:
            proj_scene.draw360_to_texture(cubetexture)
        god_scene.draw()
    else:
        proj_scene.draw()
    label_fps.draw()


perf_stats = {'msecs': deque(maxlen=ROLLING_WINDOW_LEN),
              'fps': deque(maxlen=ROLLING_WINDOW_LEN)}
def update(dt):
    if IS_MOVING:
        for sphere in spheres:
            sphere.rot_y += sphere.rot_velocity * dt

    # Calculate performance statistics

    fps_instant = 1. / (dt + .0000001)
    if fps_instant < 1000.:
        perf_stats['fps'].append(fps_instant)
        perf_stats['msecs'].append(dt * 1000)

    fps_text = 'msecs: {}\nFPS: {}'.format(np.mean(perf_stats['msecs']), np.mean(perf_stats['fps']))
    label_fps.text = fps_text

pyglet.clock.schedule(update)


pyglet.app.run()


