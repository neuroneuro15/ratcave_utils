
import pyglet
pyglet.options['debug_gl'] = False
import time
import motive
from pyglet.window import key
import click
import ratcave as rc
from . import cli
import numpy as np
import pickle
import pyglet.gl as gl



@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path(exists=True))
@click.argument('arena_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
@click.option('--screen', help='Screen Number to display on', type=int, default=1)
def vr_demo(motive_filename, projector_filename, arena_filename, body, screen):

    motive.initialize()
    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)

    body = motive.get_rigid_bodies()[body]
    for el in range(3):
        body.reset_orientation()
        motive.update()
    arena_body = motive.get_rigid_bodies()['Arena']

    # Load projector's Camera object, created from the calib_projector ratcave_utils CLI tool.
    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    window = pyglet.window.Window(fullscreen=True, screen=screen, vsync=False)


    fbo = rc.FBO(rc.TextureCube(width=4096, height=4096))

    shader3d = rc.Shader.from_file(*rc.resources.genShader)

    reader = rc.WavefrontReader(rc.resources.obj_primitives)
    mesh = reader.get_mesh('Monkey', scale=.1, position=(0, 0, 0))
    mesh.uniforms['ambient'] = .15, .15, .15

    # mesh.rotation = mesh.rotation.to_quaternion()

    arena = rc.WavefrontReader(arena_filename.encode()).get_mesh('Arena')
    arena.rotation = arena.rotation.to_quaternion()
    arena.texture = fbo.texture


    vr_scene = rc.Scene(meshes=[mesh], bgColor=(0., 0, 0))
    vr_scene.camera.projection.fov_y = 90
    vr_scene.camera.projection.aspect = 1.
    vr_scene.camera.projection.z_near = .005
    vr_scene.camera.projection.z_far = 2.

    fbo_aa = rc.FBO(rc.Texture(width=4096, height=4096, mipmap=True))
    quad = rc.gen_fullscreen_quad()
    quad.texture = fbo_aa.texture
    shader_deferred = rc.Shader.from_file(*rc.resources.deferredShader)


    scene = rc.Scene(meshes=[arena], bgColor=(0., 0., .1))


    camera = rc.Camera.from_pickle(projector_filename.encode())
    camera.projection.fov_y = 39
    scene.camera = camera
    scene.light.position.xyz = camera.position.xyz
    vr_scene.light.position.xyz = camera.position.xyz

    @window.event
    def on_draw():
        with shader3d:
            with fbo:
                vr_scene.camera.projection.match_aspect_to_viewport()
                vr_scene.draw360_to_texture(fbo.texture)
            scene.camera.projection.match_aspect_to_viewport()
            with fbo_aa:

                scene.draw()
        with shader_deferred:
            quad.draw()





    def update_body(dt, body):
        motive.update()
        mesh.position.xyz = arena_body.location
        # mesh.position.y -= .5
        mesh.rotation.xyzw = arena_body.rotation_quats
        # mesh.rotation.y += 10 * dt

        arena.position.xyz = arena_body.location
        arena.rotation.xyzw = arena_body.rotation_quats

        vr_scene.camera.position.xyz = body.location
        scene.camera.uniforms['playerPos'] = body.location



    pyglet.clock.schedule(update_body, body)

    pyglet.app.run()


