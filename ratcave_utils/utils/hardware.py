import motive

def motive_camera_vislight_configure():

    for cam in motive.get_cams():

        # All cameras should have frame rate changed.
        cam.frame_rate = 30

        if 'Prime 13' in cam.name:
            cam.settings = motive.CameraSettings(video_mode=2, exposure=33000, threshold=200, intensity=0)
            cam.image_gain = 8  # 8 is the maximum image gain setting
            cam.set_filter_switch(False)
        else:
            cam.settings = motive.CameraSettings(video_mode=0, exposure=cam.exposure, threshold=cam.threshold, intensity=cam.intensity)

