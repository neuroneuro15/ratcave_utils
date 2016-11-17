from setuptools import setup, find_packages

setup(
    name='ratcave_calibration',
    version='0.1',
    packages=find_packages(),
    install_requires=['Click', 'numpy', 'pyglet'],  # TODO: Find out how to list OpenCV-2 (cv2 didn't work)
    entry_points={
        'console_scripts': [
            'ratcave_utils = ratcave_calibration:cli',
            'calib_projector = ratcave_calibration.calib_projector:calib_projector',
            'track_rigidbody = ratcave_calibration.track_rigidbody:trackbody',
            'scan_arena = ratcave_calibration.arena_scanner:scan_arena',
            'dots_test = ratcave_calibration.dots_test:show_dots',
        ],
    }
)
