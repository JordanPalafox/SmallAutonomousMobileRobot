import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index marker
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        # package manifest
        ('share/' + package_name, ['package.xml']),
        # launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
        # config files (yaml params + reference images)
        (
            os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))
            + glob(os.path.join('config', '*.png')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jordan',
    maintainer_email='jordanpalafoxs@gmail.com',
    description='Vision-based perception for the Puzzlebot AMR',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'camera_calibration = perception.camera_calibration:main',
            'qr_quad_alignment = perception.qr_quad_alignment:main',
            'approach_stop_debug = perception.approach_stop_debug:main',
            'logo_stop_debug = perception.logo_stop_debug:main',
        ],
    },
)
