from setuptools import setup, find_packages

setup(
    name='ratcave_calibration',
    version='0.1',
    packages=find_packages(),
    install_requires=['Click', 'numpy', 'pyglet'],  # TODO: Find out how to list OpenCV-2 (cv2 didn't work)
    entry_points={
        'console_scripts': [
            'calibrate_projector = ratcave_calibration.calib_projector:run',
            'track_rigidbody = ratcave_calibration.track_rigidbody:trackbody',
            'scan_arena = ratcave_calibration.arena_scanner:run',
            'dots_test = ratcave_calibration.dots_test:run',
        ],
    }
)
